import os, glob
import numpy as np
import json
import cv2
import math

from config import Config as cfg
from dataloader.data_util.a2d2_calib_reader import get_calibration
from utils.util_function import print_progress


def trim_empty_anno_frames(root_path):
    img_files = sorted(glob.glob(os.path.join(root_path, 'image', '*.png')))
    calib_dict = get_calibration(root_path)
    num_img = len(img_files)
    for i, image_file in enumerate(img_files):
        image = cv2.imread(image_file)
        label_file = image_file.replace('image/', 'label/').replace('.png', '.json')
        cam_file = image_file.replace('image/', 'camera/')
        with open(label_file, 'r') as f:
            label = json.load(f)
        anns = convert_bev(label, image, calib_dict, yaw=True, vp_res=True, bins=12)
        if len(anns) == 0:
            if os.path.isfile(image_file):
                os.remove(image_file)
            if os.path.isfile(label_file):
                os.remove(label_file)
            if os.path.isfile(cam_file):
                os.remove(cam_file)
        print_progress(f"{i}/{num_img}")


def convert_bev(label, image, calib, vp_res, bins, bvres=0.05, yaw=False):
    annotations = list()
    categories = cfg.Datasets.A2D2.CATEGORIES_TO_USE
    for boxes, obj in label.items():
        obj_category = obj['class']
        if obj_category not in categories:
            continue

        location = np.array(obj['center']).reshape((1, 3))
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib["R0"]), np.transpose(location)))
        n = pts_3d_ref.shape[0]
        pts_3d_homo = np.hstack((pts_3d_ref, np.ones((n, 1))))
        pts_3d_velo = np.dot(pts_3d_homo, np.transpose(calib["C2V"]))
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = obtain_bvbox(obj, image, pts_3d_velo, 0.05)
        if bbox_xmin < 0:
            continue

        label = categories.index(obj_category)
        ann = dict()
        ann['category'] = [label]
        ann['bbox2d'] = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]

        # ONLY VALID FOR FRONTAL CAMERA (ONLY_FRONT PARAM)
        velodyne_h = 1.12
        ann['bbox3d'] = [(bbox_xmin + bbox_xmax) / 2., (bbox_ymin + bbox_ymax) / 2.,
                         round(obj['size'][1] / bvres, 3), round(obj['size'][2] / bvres, 3),
                         obj['size'][0] * 255 / 3.,
                         ((pts_3d_velo[0][2] + velodyne_h) + obj['size'][0] * 0.5) * 255 / 3.]
        ann["object"] = [1]
        if yaw:
            ann['yaw'] = [rad2bin(obj['rot_angle'], bins), obj['rot_angle']] if vp_res else [
                rad2bin(obj['rotation_y'], bins)]

        annotations.append(ann)
    return annotations


def obtain_bvbox(obj, bv_img, pv, bvres=0.05):
    bvrows, bvcols, _ = bv_img.shape
    centroid = [round(num, 2) for num in pv[0][:2]]  # Lidar coordinates
    length = obj['size'][1]
    width = obj['size'][0]
    yaw = obj['rot_angle']

    # Compute the four vertexes coordinates
    corners = np.array([[centroid[0] - length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] - width / 2.],
                        [centroid[0] - length / 2., centroid[1] - width / 2.]])

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


def rad2bin(rad, bins):
    bin_dist = np.linspace(-math.pi, math.pi, bins + 1)  # for each class (bins*n_classes)
    bin_res = (bin_dist[1] - bin_dist[0]) / 2.
    bin_dist = [bin - bin_res for bin in
                bin_dist]  # Substracting half of the resolution to each bin it obtains one bin for each direction (N,W,S,E)
    for i_bin in range(len(bin_dist) - 1):
        if bin_dist[i_bin] <= rad and bin_dist[i_bin + 1] >= rad:
            return i_bin
    return 0  # If the angle is above max angle, it won't match so it corresponds to initial bin, initial bin must be from (-pi+bin_res) to (pi-bin_res)


if __name__ == "__main__":
    path = cfg.Datasets.A2D2.PATH
    trim_empty_anno_frames(path)
