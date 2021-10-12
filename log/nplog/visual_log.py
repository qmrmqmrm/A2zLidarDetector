import os
import os.path as op
import torch
import numpy as np
import cv2

import config as cfg
from log.nplog.metric import split_true_false
import utils.util_function as uf
import model.submodules.model_util as mu


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        # self.grtr_log_keys = cfg.Train.LOG_KEYS
        # self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Datasets.Standard.CATEGORY_NAMES)}

    def __call__(self, step, grtr, gt_aligned, gt_feature, pred):
        """
        :param step: integer step index
        :param grtr:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4(tlbr)], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
            'gt_feature' : {
                'bbox3d' : list(torch.Size([batch, height/stride* width/stride* anchor, 6]))
                'category' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                'bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4]))
                'yaw' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                'yaw_rads' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                'anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                'object' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                'negative' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                }
        :param pred:
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'category' : torch.Size([batch, 512, class_num, 1])
            'bbox3d' : torch.Size([batch, 512, class_num, 6])
            'yaw' : torch.Size([batch, 512, class_num, 12])
            'yaw_rads' : torch.Size([batch, 512, class_num, 12])
            }
        """
        # grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        splits = split_true_false(grtr, pred, cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["bbox2d"].shape[0]
        feature_bbox2d = gt_feature['bbox2d']
        feature_object = gt_feature['object']

        anchors = []
        for anchor in grtr['anchors']:
            anchor = anchor.reshape(batch, -1, anchor.shape[-1])
            anchor = mu.convert_box_format_yxhw_to_tlbr(anchor) #tlbr
            anchors.append(anchor)
        for scale, anchor,bbox2d, object in zip(cfg.Scales.DEFAULT_FEATURE_SCALES, anchors,feature_bbox2d, feature_object):
            bbox2d = bbox2d.to('cpu').detach().numpy() * object.to('cpu').detach().numpy()

            for i in range(batch):
                org_img = grtr["image"][i].copy()
                bbox2d_per_img = bbox2d[i]
                anchor_per_img = anchor[i]

                valid_mask = bbox2d_per_img[:, 2] > 0  # (N,) h>0
                bbox2d_per_img = bbox2d_per_img[valid_mask, :]
                anchor_per_img = anchor_per_img[valid_mask, :-1]
                image_pred = self.draw_rpn_boxes(org_img,bbox2d_per_img, i, (0, 255, 0))
                image_pred = self.draw_rpn_boxes(image_pred,anchor_per_img, i, (0, 0, 255))
                filename = op.join(self.vlog_path, f"anchor_{step * batch + i :05d}scale{scale}.jpg")
                cv2.imwrite(filename, image_pred)

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_org_file_name = grtr["image_file"][i].replace('image/', 'camera/')
            image_org = cv2.imread(image_org_file_name)
            image_org = cv2.resize(image_org, (1280, 640))

            image_grtr = grtr["image"][i].copy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, (0, 0, 255))

            image_pred = grtr["image"][i].copy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, (0, 0, 255))

            vlog_image = np.concatenate([image_pred, image_grtr], axis=1)
            vlog_image = np.concatenate([image_org, vlog_image], axis=0)

            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            filename_ = op.join(self.vlog_path, f"{step * batch + i:05d}_.jpg")
            cv2.imwrite(filename, vlog_image)
            cv2.imwrite(filename_, vlog_image)

    def draw_boxes(self, image, bboxes, frame_idx, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """
        height, width = image.shape[:2]
        bbox2d = bboxes["bbox2d"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        valid_mask = bbox2d[:, 2] > 0  # (N,) h>0

        bbox2d = bbox2d[valid_mask, :]
        category = category[valid_mask, 0].astype(np.int32)  # (N',)
        for i in range(bbox2d.shape[0]):
            x1, y1, x2, y2 = bbox2d[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            #
            # cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def draw_rpn_boxes(self, image, bbox2d, frame_idx, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """
        for i in range(bbox2d.shape[0]):
            x1, y1, x2, y2 = bbox2d[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            #
            # cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image
