import os, glob
import numpy as np
import json
import cv2
import math

import torch

from dataloader.loader_base import DatasetBase
from dataloader.data_util.a2d2_calib_reader import get_calibration
from dataloader.utils_calib import Calibration
import config as cfg
from utils.util_function import print_progress

max_box = 512


# TODO: rearrange-code-21-11, remove kitti related code

class KittiDataset(DatasetBase):
    def __init__(self, path):
        self.max_box = max_box
        self.calib_path = path
        self.camera_path = os.path.join(self.calib_path, 'image')
        self.img_files = sorted(glob.glob(os.path.join(self.camera_path, '*.png')))
        self.anns_list = self.get_img_list(self.img_files)
        self.label_path = self.camera_path.replace("/image", "/label")

        super().__init__(self.anns_list)

    def get_img_list(self, img_files):
        anns_list = list()
        num_img = len(img_files)
        ann_dict = {}
        outfile_name = "/media/dolphin/intHDD/birdnet_data/bv_kitti/result/anno.json"
        if os.path.exists(outfile_name):
            print('File exists: ' + str(outfile_name))
            # ans = input("Do you want to overwrite it? (y/n)")
            # ans = input("y / n  : ")
            ans = 'n'
            if ans is 'n':
                # Return always the same file to match with training script
                with open(outfile_name, 'r') as f:
                    anns_list = json.load(f)
                return anns_list
        for i, img_file in enumerate(img_files):
            label_file = img_file.replace('image/', 'label/').replace('.png', '.txt')
            calib_file = img_file.replace('image/', 'calib/').replace('.png', '.txt')

            calib = Calibration(calib_file)
            label = np.genfromtxt(label_file, delimiter=' ',
                                  names=['type', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin',
                                         'bbox_xmax', 'bbox_ymax', 'dimensions_1', 'dimensions_2', 'dimensions_3',
                                         'location_1', 'location_2', 'location_3', 'rotation_y'], dtype=None)
            # ['2d_bbox', '3d_points', 'alpha', 'axis', 'center', 'class', 'id', 'occlusion', 'rot_angle', 'size', 'truncation']
            anns = self.convert_bev(label, img_file, calib, yaw=True, vp_res=True, bins=12)
            if len(anns) != 0:
                anns_list.append(anns)
            print_progress(f"{i}/{num_img}")

        ann_dict['annotations'] = anns_list
        with open(outfile_name, "w") as outfile:
            outfile.write(json.dumps(ann_dict))
        return ann_dict

    def convert_bev(self, label, img_file, calib, vp_res, bins, bvres=0.05, yaw=False):
        ann_id = 0
        annotations = list()
        if (label.ndim < 1):
            label = np.array(label, ndmin=1)
        img = cv2.imread(img_file)
        only_eval_classes = '0, 3, 5'
        only_eval_classes = only_eval_classes.split(',')
        only_eval_classes = [int(cl) for cl in only_eval_classes]
        categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        categories = [categories[idx] for idx in only_eval_classes]
        category_dict = {k: v for v, k in enumerate(categories)}
        for obj in label:
            o = obj['type']
            print(o)
            if isinstance(o, (bytes, np.bytes_)):
                o = o.decode("utf-8")
            label = category_dict.get(o, 8)  # Default value just in case
            if (label != 7) and (label != 8):
                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = obtain_bvbox(obj, img, calib, bvres)
                if bbox_xmin < 0:
                    continue
                print(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

                cv2.imshow("im", img)
                cv2.waitKey()
                dimg = cv2.rectangle(img, (int(bbox_xmin), int(bbox_xmax)), (int(bbox_ymin), int(bbox_ymax)),
                                     (255, 255, 255), 2)
                cv2.imshow("drow_img", dimg)
                cv2.waitKey()
                ann = {}
                ann['image_file'] = img_file
                ann['bbox_id'] = ann_id
                ann_id += 1
                ann['category'] = label
                boxes = np.empty((0, 4), dtype=np.float32)
                ann['bbox2d'] = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]

                # ONLY VALID FOR FRONTAL CAMERA (ONLY_FRONT PARAM)
                p = calib.project_rect_to_velo(np.array([[obj['location_1'], obj['location_2'], obj['location_3']]]))
                ann['height'] = [obj['dimensions_1'] * 255 / 3.0, ((p[0][2] + 1.73) + obj[
                    'dimensions_1'] * 0.5) * 255 / 3.0]  # (p[0][2]+velodyne_h)]#Not codificated ground
                ann['bbox3d'] = [(bbox_xmin + bbox_xmax) / 2., (bbox_ymin + bbox_ymax) / 2.,
                                 round(obj['dimensions_2'] / bvres, 3), round(obj['dimensions_3'] / bvres, 3)]
                ann["object"] = 1
                if yaw:
                    ann['yaw'] = [rad2bin(obj['rotation_y'], bins), obj['rotation_y']] if vp_res else [
                        rad2bin(obj['rotation_y'], bins)]

                annotations.append(ann)

        return annotations

    def get_anno_data(self, ann):

        fixed_anns = self.gather_elements(ann)
        fixed_anns["box2d_map"], fixed_anns["object_map"], _ = self.gather_featmaps(fixed_anns["gt_bbox2D"])
        # print("fixed_anns.keys() : ", fixed_anns.keys())
        return fixed_anns

    def gather_elements(self, anns):
        """
        :param anns: {'image_file', 'image', 'bbox_id', 'category', 'bbox2d', 'bbox3d', 'object', 'yaw'}
        :return: gathered_anns:
        """
        gathered_anns = dict()
        for i, ann in enumerate(anns):
            for ann_key, ann_val in ann.items():

                if ann_key == "image_file":
                    img = cv2.imread(ann_val)
                    img = torch.tensor(img)
                    # img = img.to(cfg.Model.Structure.DEVICE)
                    gathered_anns["image_file"] = ann_val
                    gathered_anns["image"] = img.permute(2, 0, 1)

                else:
                    if not isinstance(ann_val, int):
                        rank = len(ann_val)
                    else:
                        rank = 1


                    if not ann_key in gathered_anns.keys():
                        gathered_anns[f"gt_{ann_key}"] = torch.tensor([ann_val])
                        gathered_anns[f"num_{ann_key}"] = torch.zeros((self.max_box, rank))
                    gathered_anns[f"num_{ann_key}"][i, :] = torch.tensor(ann_val)

        return gathered_anns

    def gather_featmaps(self, bbox2d):
        """
                 res2(Tensor) : torch.Size([4, 256, 176, 352]) 3.9772727
                 res3(Tensor) : torch.Size([4, 512, 88, 176])  7.95454545
                 res4(Tensor) : torch.Size([4, 1024, 44, 88])  15.9090909
        :param bbox2d: [num_box, channel]
        :return: "box2d_map": [channel, height, width]
        """

        channel_shape = bbox2d.shape[-1]

        center_xs = bbox2d[:, 0].reshape((-1, 1))
        center_ys = bbox2d[:, 1].reshape((-1, 1))
        # center_xs= np.reshape(center_xs,(1,-1))
        heights = [176, 88, 44]
        box2D_map_dict = dict()
        object_map_dict = dict()
        anchor_map_dict = dict()
        # anchor = self.anchor.make_anchor_map(heights)
        feature_names = cfg.Model.Output.FEATURE_ORDER
        for i, (height, feature_name) in enumerate(zip(heights, feature_names)):
            rasio = 700 / height

            bbox2d_map = np.zeros((height, height * 2, channel_shape))
            object_map = np.zeros((height, height * 2, 1))
            # [w*h*a,c]
            # anchor_map = anchor[i].view(height, height * 2, 9, -1)
            # anchor_map = anchor_map.to("cpu")
            for j, (center_x, center_y) in enumerate(zip(center_xs, center_ys)):
                w = center_x / rasio
                h = center_y / rasio
                bbox2d_map[int(h), int(w), :] = bbox2d[j, :]
                object_map[int(h), int(w), :] = 1

            # mask_map = torch.tensor(bbox2d_map[:, :, None, :], device="cuda")
            # anchor_map = anchor_map * mask_map
            # print("test_map : ",anchor_map)
            box2D_map_dict[feature_name] = bbox2d_map
            object_map_dict[feature_name] = object_map
            # anchor_map_dict[feature_name] = anchor_map

        return box2D_map_dict, object_map_dict, anchor_map_dict


def rad2bin(rad, bins):
    bin_dist = np.linspace(-math.pi, math.pi, bins + 1)  # for each class (bins*n_classes)
    bin_res = (bin_dist[1] - bin_dist[0]) / 2.
    bin_dist = [bin - bin_res for bin in
                bin_dist]  # Substracting half of the resolution to each bin it obtains one bin for each direction (N,W,S,E)
    for i_bin in range(len(bin_dist) - 1):
        if bin_dist[i_bin] <= rad and bin_dist[i_bin + 1] >= rad:
            return i_bin

    return 0  # If the angle is above max angle, it won't match so it corresponds to initial bin, initial bin must be from (-pi+bin_res) to (pi-bin_res)


def obtain_bvbox(obj, bv_img, calib, bvres=0.05):
    bvrows, bvcols, _ = bv_img.shape
    pv = calib.project_rect_to_velo(np.array([[obj['location_1'], obj['location_2'], obj['location_3']]]))

    centroid = [round(num, 2) for num in pv[0][:2]]  # Lidar coordinates
    #
    length = obj['dimensions_2']
    width = obj['dimensions_3']
    yaw = obj['rotation_y']

    # # Compute the four vertexes coordinates
    corners = np.array([[centroid[0] - length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] - width / 2.],
                        [centroid[0] - length / 2., centroid[1] - width / 2.]])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    rotated_corners = np.dot(corners - centroid, R) + centroid

    x1 = bvcols / 2 + min(-rotated_corners[:, 1]) / bvres
    x2 = bvcols / 2 + max(-rotated_corners[:, 1]) / bvres
    y1 = bvrows - max(rotated_corners[:, 0]) / bvres
    y2 = bvrows - min(rotated_corners[:, 0]) / bvres
    roi = bv_img[int(y1):int(y2), int(x1):int(x2)]
    nonzero = np.count_nonzero(np.sum(roi, axis=2))
    if nonzero < 3:  # Detection is doomed impossible with fewer than 3 points
        return -1, -1, -1, -1
    # # TODO: Assign DontCare labels to objects with few points?
    #
    # # Remove objects outside the BEV image
    if x1 <= 0 and x2 <= 0 or \
            x1 >= bvcols - 1 and x2 >= bvcols - 1 or \
            y1 <= 0 and y2 <= 0 or \
            y1 >= bvrows - 1 and y2 >= bvrows - 1:
        return -1, -1, -1, -1 # Out of bounds
    #
    # # Clip boxes to the BEV image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bvcols - 1, x2)
    y2 = min(bvrows - 1, y2)

    return x1, y1, x2, y2


def drow_box(img, bbox):
    x0 = int(bbox[0])
    x1 = int(bbox[2])
    y0 = int(bbox[1])
    y1 = int(bbox[3])

    img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imshow("drow_img", img)
    cv2.waitKey()
