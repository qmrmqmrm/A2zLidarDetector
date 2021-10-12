import os, glob
import numpy as np
import json
import cv2
import math

import torch

from dataloader.loader_base import DatasetBase
from dataloader.data_util.a2d2_calib_reader import get_calibration
from dataloader.anchor import Anchor
import config as cfg
import utils.util_function as uf
import model.submodules.model_util as mu


class A2D2Dataset(DatasetBase):
    def __init__(self, root_path, split, max_box=cfg.Datasets.A2D2.MAX_NUM):
        super().__init__(root_path, split)
        self.max_box = max_box
        self.calib_dict = get_calibration(root_path)
        self.categories = cfg.Datasets.A2D2.CATEGORIES_TO_USE
        self.device = cfg.Hardware.DEVICE

        image_shape = torch.tensor(cfg.Model.Structure.IMAGE_SHAPE)
        strides = torch.tensor(cfg.Model.Structure.STRIDE_SHAPE)
        feature_shape = list()
        for stride in strides:
            feature_shape.append((image_shape / stride).to(device=self.device, dtype=torch.int64))
        self.anchors = Anchor('a2d2')()  # yxwh + id

        # self.last_sample = self.__getitem__(91)
        # del self.last_sample['image']

    def list_frames(self, root_dir, split):
        img_files = sorted(glob.glob(os.path.join(root_dir, split, '*/image', '*.png')))

        return img_files

    def __getitem__(self, index):
        """
        :param index: index for self.img_files
        :return: {'image': [height, width, channel], 'category': [fixbox, 1], 'bbox2d': [fixbox, 4](tlbr),
                    'bbox3d': [fixbox, 6], 'object': [fixbox, 1], 'yaw': [fixbox, 2]}
        """
        image_file = self.img_files[index]
        image = cv2.imread(image_file)
        features = dict()

        features['image'] = torch.tensor(image)
        label_file = image_file.replace('image/', 'label/').replace('.png', '.json')
        with open(label_file, 'r') as f:
            label = json.load(f)

        anns = self.convert_bev(label, image, self.calib_dict, bins=12, yaw=True)

        if anns:
            anns = self.gather_and_zeropad(anns)
            self.last_sample = anns
        else:
            anns = self.zero_annotation()
        features['anchors'] = self.anchors
        features.update(anns)
        features['image_file'] = image_file
        features['anchor_id'] = self.matched_anchor(features['anchors'], features['bbox2d'])

        return features

    def convert_bev(self, label, image, calib, bins, bvres=0.05, yaw=False):
        annotations = list()

        for boxes, obj in label.items():
            obj_category = obj['class']
            if obj_category not in self.categories:
                continue

            location = np.array(obj['center']).reshape((1, 3))
            pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib["R0"]), np.transpose(location)))
            n = pts_3d_ref.shape[0]
            pts_3d_homo = np.hstack((pts_3d_ref, np.ones((n, 1))))
            pts_3d_velo = np.dot(pts_3d_homo, np.transpose(calib["C2V"]))
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = self.obtain_bvbox(obj, image, pts_3d_velo, bvres)
            if bbox_xmin < 0:
                continue

            label = self.categories.index(obj_category)
            ann = dict()
            ann['category'] = [label]
            ann['bbox2d'] = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]  # tlbr

            # ONLY VALID FOR FRONTAL CAMERA (ONLY_FRONT PARAM)
            velodyne_h = 1.12
            ann['bbox3d'] = [(bbox_xmin + bbox_xmax) / 2., (bbox_ymin + bbox_ymax) / 2.,
                             round(obj['size'][1] / bvres, 3), round(obj['size'][0] / bvres, 3),
                             obj['size'][2] * 255 / 3.,
                             ((pts_3d_velo[0][2] + velodyne_h) + obj['size'][0] * 0.5) * 255 / 3.]  # xylwzh
            ann["object"] = [1]
            if yaw:
                ann['yaw'] = [rad2bin(obj['rot_angle'], bins)]
                ann['yaw_rads'] = [obj['rot_angle']]
            annotations.append(ann)
        return annotations

    def obtain_bvbox(self, obj, bv_img, pv, bvres=0.05):
        bvrows, bvcols, _ = bv_img.shape
        centroid = [round(num, 2) for num in pv[0][:2]]  # Lidar coordinates
        length = obj['size'][1]
        width = obj['size'][0]
        yaw = obj['rot_angle']
        # print('lwh')
        # print(length, width, yaw)
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

    def gather_and_zeropad(self, anns):
        """
        :param anns: list of {'category': [], 'bbox2d': [](tlbr), 'bbox3d': [], 'object': [], 'yaw': []}
        :return: gathered_anns: {'category': [numbox,], 'bbox2d': [numbox,], 'bbox3d': [numbox,], 'object': [numbox,], 'yaw': [numbox]}
        """
        gathered_anns = {key: [] for key in anns[0].keys()}
        for i, ann in enumerate(anns):
            for ann_key, ann_val in ann.items():
                gathered_anns[ann_key].append(ann_val)

        gathered_anns = {key: torch.tensor(vals, dtype=torch.float32) for key, vals in gathered_anns.items()}
        # zero padding
        for key in gathered_anns:
            numbox, channel = gathered_anns[key].shape
            if self.max_box - numbox > 0:
                if key == 'category':
                    pad = torch.zeros((self.max_box - numbox, channel), dtype=torch.float32)
                elif key == 'yaw':
                    pad = torch.zeros((self.max_box - numbox, channel), dtype=torch.float32)
                elif key == 'yaw_rads':
                    pad = torch.zeros((self.max_box - numbox, channel), dtype=torch.float32)
                elif key == 'object':
                    pad = torch.zeros((self.max_box - numbox, channel), dtype=torch.float32)
                else:
                    pad = torch.zeros((self.max_box - numbox, channel), dtype=torch.float32)

                gathered_anns[key] = torch.cat([gathered_anns[key], pad], dim=0)
            else:
                gathered_anns[key] = gathered_anns[key][:self.max_box, :]
        return gathered_anns

    def zero_annotation(self):
        gathered_anns = {key: torch.zeros(val.shape, dtype=torch.float32) for key, val in self.last_sample.items()}
        return gathered_anns

    def matched_anchor(self, anchors, bbox2d):
        """
        :param anchors: (height,width,anchor,5(yxhw+anchor_id))
        :param bbox2d: (fixed_num(15), 4(tlbr))
        :return: (fixed_num(15), 1)
        """
        feature_anchor = list()
        for anchor in anchors:
            anchor = anchor.view(-1, anchor.shape[-1])
            anchor = mu.convert_box_format_yxhw_to_tlbr(anchor)
            feature_anchor.append(anchor)
        feature_anchor = torch.cat(feature_anchor)
        iou = uf.pairwise_iou(bbox2d, feature_anchor[..., :-1])
        max_iou, max_idx = iou.max(dim=1)
        anchor_id = feature_anchor[max_idx, -1:]
        return anchor_id

    def gather_featmaps(self, bbox2d, objectness):
        """
                 res2(Tensor) : torch.Size([4, 256, 176, 352]) 3.9772727
                 res3(Tensor) : torch.Size([4, 512, 88, 176])  7.95454545
                 res4(Tensor) : torch.Size([4, 1024, 44, 88])  15.9090909
        :param bbox2d: [padded_box, 4]
        :param objectness: [padded_box, 1]
        :return: "box2d_map": [channel, height, width]
        """
        channel_shape = bbox2d.shape[-1]
        center_xs = bbox2d[:, 0].reshape((-1, 1))
        center_ys = bbox2d[:, 1].reshape((-1, 1))
        heights = [176, 88, 44]
        box2D_map_dict = dict()
        object_map_dict = dict()
        feature_names = cfg.Model.Output.FEATURE_ORDER
        for i, (height, feature_name) in enumerate(zip(heights, feature_names)):
            ratio = 700 / height

            bbox2d_map = np.zeros((height, height * 2, channel_shape))
            object_map = np.zeros((height, height * 2, 1))
            for j, (center_x, center_y) in enumerate(zip(center_xs, center_ys)):
                w = center_x / ratio
                h = center_y / ratio
                bbox2d_map[int(h), int(w), :] = bbox2d[j, :]
                object_map[int(h), int(w), :] = 1

            box2D_map_dict[feature_name] = bbox2d_map
            object_map_dict[feature_name] = object_map
        return box2D_map_dict, object_map_dict


def rad2bin(rad, bins):
    bin_dist = np.linspace(-math.pi, math.pi, bins + 1)  # for each class (bins*n_classes)
    bin_res = (bin_dist[1] - bin_dist[0]) / 2.
    bin_dist = [bin - bin_res for bin in
                bin_dist]  # Substracting half of the resolution to each bin it obtains one bin for each direction (N,W,S,E)
    for i_bin in range(len(bin_dist) - 1):
        if bin_dist[i_bin] <= rad and bin_dist[i_bin + 1] >= rad:
            return i_bin
    return 0  # If the angle is above max angle, it won't match so it corresponds to initial bin, initial bin must be from (-pi+bin_res) to (pi-bin_res)


def drow_box(img, bbox):
    bbox = bbox.detach().numpy()
    num, channel = bbox.shape
    for n in range(num):
        x0 = int(bbox[n, 0])
        x1 = int(bbox[n, 2])
        y0 = int(bbox[n, 1])
        y1 = int(bbox[n, 3])
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imshow("drow_img", img)
    cv2.waitKey()
