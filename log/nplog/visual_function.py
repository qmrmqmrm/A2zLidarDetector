import math
import os
import os.path as op
import torch
import numpy as np
import cv2

import config as cfg
import utils.util_function as uf
import model.submodules.model_util as mu
import dataloader.data_util.a2d2_calib_reader as ar


# TODO: rearrange-code-21-11, rename file

def rotated_box2d(bboxes_yxlw, yaw_rads):
    bbox_yx, bbox_lw = bboxes_yxlw[:2], bboxes_yxlw[2:4]
    bbox_xy = np.array([bbox_yx[1], bbox_yx[0]])
    corners = get_box2d_corner(bbox_lw)
    # yaw = -(yaw_rads[0] + (math.pi / 2))
    c, s = np.cos(yaw_rads[0]), np.sin(yaw_rads[0])
    R = np.array([[c, -s],
                  [-s, -c]])  # image coordinates: flip y
    rotated_corners = np.dot(corners, R) + bbox_xy
    return rotated_corners


def get_box2d_corner(bbox_lw):
    return np.array([[- bbox_lw[1] / 2., - bbox_lw[0] / 2.],
                     [+ bbox_lw[1] / 2., - bbox_lw[0] / 2.],
                     [+ bbox_lw[1] / 2., + bbox_lw[0] / 2.],
                     [- bbox_lw[1] / 2., + bbox_lw[0] / 2.]])


def rotated_box3d(bboxes, rads):
    box_yx, box_lw, box_h = bboxes[..., :2], bboxes[..., 2:4], bboxes[4:]
    box_lwh = np.concatenate([box_lw, box_h])
    box_yxz = np.concatenate([box_yx, box_h / 2])
    # calculate unrotated corner point offsets relative to center
    # rotate points
    R = axis_angle_to_rotation_mat(rads)
    points = get_box3d_corner(box_lwh)
    points = np.dot(points, R) + np.asarray(box_yxz)
    return points


def get_box3d_corner(bbox_lwh):
    brl = np.asarray([-bbox_lwh[..., 0] / 2, +bbox_lwh[..., 1] / 2, -bbox_lwh[..., 2] / 2])
    bfl = np.asarray([+bbox_lwh[..., 0] / 2, +bbox_lwh[..., 1] / 2, -bbox_lwh[..., 2] / 2])
    bfr = np.asarray([+bbox_lwh[..., 0] / 2, -bbox_lwh[..., 1] / 2, -bbox_lwh[..., 2] / 2])
    brr = np.asarray([-bbox_lwh[..., 0] / 2, -bbox_lwh[..., 1] / 2, -bbox_lwh[..., 2] / 2])
    trl = np.asarray([-bbox_lwh[..., 0] / 2, +bbox_lwh[..., 1] / 2, +bbox_lwh[..., 2] / 2])
    tfl = np.asarray([+bbox_lwh[..., 0] / 2, +bbox_lwh[..., 1] / 2, +bbox_lwh[..., 2] / 2])
    tfr = np.asarray([+bbox_lwh[..., 0] / 2, -bbox_lwh[..., 1] / 2, +bbox_lwh[..., 2] / 2])
    trr = np.asarray([-bbox_lwh[..., 0] / 2, -bbox_lwh[..., 1] / 2, +bbox_lwh[..., 2] / 2])
    return np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])


def axis_angle_to_rotation_mat(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def obtain_bvbox(obj, rot_angle, bv_img, pv, bvres=0.05):
    bvrows, bvcols, _ = bv_img.shape
    centroid = [round(num, 2) for num in pv[0][:2]]  # Lidar coordinates
    width = obj['size'][1]
    length = obj['size'][0]
    yaw = rot_angle
    # print('lwh')
    # print(length, width, yaw)
    # Compute the four vertexes coordinates
    corners = np.array([[centroid[0] - width / 2., centroid[1] + length / 2.],
                        [centroid[0] + width / 2., centroid[1] + length / 2.],
                        [centroid[0] + width / 2., centroid[1] - length / 2.],
                        [centroid[0] - width / 2., centroid[1] - length / 2.]])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_corners = np.dot(corners - centroid, R) + centroid
    x1 = bvcols / 2 + min(rotated_corners[:, 0]) / bvres
    x2 = bvcols / 2 + max(rotated_corners[:, 0]) / bvres
    y1 = bvrows - max(rotated_corners[:, 1]) / bvres
    y2 = bvrows - min(rotated_corners[:, 1]) / bvres

    roi = bv_img[int(y1):int(y2), int(x1):int(x2)]
    nonzero = np.count_nonzero(np.sum(roi, axis=2))
    if nonzero < 3:  # Detection is doomed impossible with fewer than 3 points
        return -1, -1, -1, -1

    # Remove objects outside the BEV image
    if (x1 <= 0 and x2 <= 0) or \
            (x1 >= bvcols - 1 and x2 >= bvcols - 1) or \
            (y1 <= 0 and y2 <= 0) or \
            (y1 >= bvrows - 1 and y2 >= bvrows - 1):
        return -1, -1, -1, -1  # Out of bounds

    # Clip boxes to the BEV image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bvcols - 1, x2)
    y2 = min(bvrows - 1, y2)

    return x1, y1, x2, y2


def bev_to_orgin_img(bvbox, rad, bvimg_shape):
    global_box = bev_box_to_global(bvbox, rad, bvimg_shape)
    org_box = ar.get_transform_from_global(global_box)


def bev_box_to_global(bvbox, rad, bvimg_shape, bvres=0.05):
    """

    bvbox : (...,5(yxlwh))
    rad : (...,1)
    """
    bvrows, bvcols, _ = bvimg_shape
    bvbox_3d_corner = rotated_box3d(bvbox, rad)
    global_box = np.concatenate([(bvrows - bvbox_3d_corner[0]) * bvres, (bvcols / 2 - bvbox_3d_corner[1]) * bvres])
    return global_box


def convert_img(feature, scale, org_img):
    feature_imge = feature.reshape((scale, scale, 3)) * 255
    feature_imge = org_img + cv2.resize(feature_imge, (640, 640), interpolation=cv2.INTER_NEAREST)
    feature_imge[-1, :] = [255, 255, 255]
    feature_imge[:, -1] = [255, 255, 255]
    return feature_imge
