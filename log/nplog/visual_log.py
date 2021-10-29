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
    def __init__(self, ckpt_path, epoch, split):
        # self.grtr_log_keys = cfg.Train.LOG_KEYS
        # self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog", split, f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)

        self.obj_path = op.join(self.vlog_path, 'object')
        if not op.isdir(self.obj_path):
            os.makedirs(self.obj_path)

        self.scale_path = op.join(self.vlog_path, 'scale')
        if not op.isdir(self.scale_path):
            os.makedirs(self.scale_path)

        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Datasets.Standard.CATEGORY_NAMES)}

    def __call__(self, step, grtr, gt_feature, pred, pred_nms):
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
        splits = split_true_false(grtr, pred_nms, cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["bbox2d"].shape[0]
        for batch_idx in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_org_file_name = grtr["image_file"][batch_idx].replace('image/', 'camera/')
            image_org = cv2.imread(image_org_file_name)
            image_org = cv2.resize(image_org, (1280, 640))

            image_grtr = grtr["image"][batch_idx].copy()

            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"]['bbox2d'], splits["grtr_fn"]['category'],
                                         batch_idx, (0, 0, 255))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"]['bbox2d'], splits["grtr_tp"]['category'],
                                         batch_idx, (0, 255, 0))

            image_pred = grtr["image"][batch_idx].copy()

            image_pred = self.draw_boxes(image_pred, splits["pred_fp"]['bbox2d'], splits["pred_fp"]['category'],
                                         batch_idx, (0, 0, 255))
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"]['bbox2d'], splits["pred_tp"]['category'],
                                         batch_idx, (0, 255, 0))



            image_grtr_3d = grtr["image"][batch_idx].copy()
            image_grtr_3d = self.draw_boxes(image_grtr_3d, splits["grtr_fn"]['bbox3d'][..., :-2], splits["grtr_fn"]['category'],
                                            batch_idx, (0, 0, 255))
            image_grtr_3d = self.draw_boxes(image_grtr_3d, splits["grtr_tp"]['bbox3d'][..., :-2], splits["grtr_tp"]['category'],
                                            batch_idx, (0, 255, 0))

            image_pred_3d = grtr["image"][batch_idx].copy()
            image_pred_3d = self.draw_boxes(image_pred_3d, splits["pred_fp"]['bbox3d'][..., :-2], splits["pred_fp"]['category'],
                                            batch_idx, (0, 0, 255))
            image_pred_3d = self.draw_boxes(image_pred_3d, splits["pred_tp"]['bbox3d'][..., :-2], splits["pred_tp"]['category'],
                                            batch_idx, (0, 255, 0))


            #
            # image_grtr_3d_org = image_org.copy()
            # image_grtr_3d_org = self.draw_boxes(image_grtr_3d_org, splits["grtr_fn"]['bbox3d'][..., :-2], splits["grtr_fn"]['category'],
            #                                 batch_idx, (0, 0, 255))
            # image_grtr_3d_org = self.draw_boxes(image_grtr_3d_org, splits["grtr_tp"]['bbox3d'][..., :-2], splits["grtr_tp"]['category'],
            #                                 batch_idx, (0, 255, 0))
            #
            # image_pred_3d_org = image_org.copy()
            # image_pred_3d_org = self.draw_boxes(image_pred_3d_org, splits["pred_fp"]['bbox3d'][..., :-2], splits["pred_fp"]['category'],
            #                                 batch_idx, (0, 0, 255))
            # image_pred_3d_org = self.draw_boxes(image_pred_3d_org, splits["pred_tp"]['bbox3d'][..., :-2], splits["pred_tp"]['category'],
            #                                 batch_idx, (0, 255, 0))


            # image_pred = self.draw_boxes(image_pred, splits["grtr_tp"], batch_idx, (0, 255, 255))
            # image_pred = self.draw_boxes(image_pred, splits["grtr_fn"], batch_idx, (0, 255, 255))

            scale_img = grtr["image"][batch_idx].copy()
            # print('pred_tp')
            scale_img = self.draw_ctgr_boxes(scale_img, splits["pred_tp"]['bbox2d'], splits["pred_tp"]['category'],
                                             batch_idx, 120)
            # print('pred_fp')
            scale_img = self.draw_ctgr_boxes(scale_img, splits["pred_fp"]['bbox2d'], splits["pred_fp"]['category'],
                                             batch_idx, 120)
            # print('grtr_tp')
            scale_img = self.draw_ctgr_boxes(scale_img, splits["grtr_tp"]['bbox2d'], splits["grtr_tp"]['category'],
                                             batch_idx, 255)
            # print('grtr_fn')
            scale_img = self.draw_ctgr_boxes(scale_img, splits["grtr_fn"]['bbox2d'], splits["grtr_fn"]['category'],
                                             batch_idx, 255)

            vlog_image = np.concatenate([image_pred, image_grtr], axis=1)
            vlog_image = np.concatenate([image_org, vlog_image], axis=0)

            vlog_image_3d = np.concatenate([image_pred_3d, image_grtr_3d], axis=1)
            vlog_image_3d = np.concatenate([image_org, vlog_image_3d], axis=0)

            # vlog_image_3d_org = np.concatenate([image_grtr_3d_org, image_pred_3d_org], axis=0)

            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + batch_idx:05d}.jpg")
            cv2.imwrite(filename, vlog_image)

            filename = op.join(self.vlog_path, f"{step * batch + batch_idx:05d}_3d.jpg")
            cv2.imwrite(filename, vlog_image_3d)

            filename = op.join(self.scale_path, f"{step * batch + batch_idx:05d}.jpg")
            cv2.imwrite(filename, scale_img)

            # filename = op.join(self.vlog_path, f"{step * batch + batch_idx:05d}_3d_org.jpg")
            # cv2.imwrite(filename, vlog_image_3d_org)

            gt_obj_imgs = []
            pred_obj_imgs = []
            for scale_idx, (gt_scale_object, pred_scale_object) in enumerate(
                    zip(gt_feature['object'], pred['rpn_feat_objectness'])):
                scale = int(640 / cfg.Scales.DEFAULT_FEATURE_SCALES[scale_idx])
                gt_object_per_image = self.convert_img(gt_scale_object[batch_idx], scale)
                pred_object_per_image = self.convert_img(pred_scale_object[batch_idx], scale)
                gt_obj_imgs.append(gt_object_per_image)
                pred_obj_imgs.append(pred_object_per_image)
            gt_obj_img = np.concatenate(gt_obj_imgs, axis=1)
            pred_obj_img = np.concatenate(pred_obj_imgs, axis=1)
            obj_img = np.concatenate([gt_obj_img, pred_obj_img], axis=0)
            filename = op.join(self.obj_path, f"{step * batch + batch_idx:05d}.jpg")
            cv2.imwrite(filename, obj_img)

    def draw_boxes(self, image, bboxes, category, frame_idx, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """
        bbox2d = bboxes[frame_idx]  # (N, 4)
        valid_mask = bbox2d[:, 2] > 0  # (N,) h>0

        bbox2d = bbox2d[valid_mask, :]
        bbox2d = mu.convert_box_format_yxhw_to_tlbr(bbox2d)
        for i in range(bbox2d.shape[0]):

            y1, x1, y2, x2 = bbox2d[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return image

    def draw_ctgr_boxes(self, image, bboxes, category, frame_idx, val):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """
        shape = category.shape

        if shape[-1] > 1:
            best_cate = np.argmax(category, axis=-1)
            category = np.expand_dims(best_cate, -1)
        bbox2d = bboxes[frame_idx]  # (N, 4)
        category = category[frame_idx]
        one_hot_ctgrs = (self.one_hot(category, 4) * val).astype(np.uint8)

        valid_mask = bbox2d[:, 2] > 0  # (N,) h>0
        bbox2d = bbox2d[valid_mask, :]
        bbox2d = mu.convert_box_format_yxhw_to_tlbr(bbox2d)
        for i in range(bbox2d.shape[0]):
            if one_hot_ctgrs[i][0] == 0:
                y1, x1, y2, x2 = bbox2d[i].astype(np.int32)
                cv2.rectangle(image, (x1, y1), (x2, y2),
                              [int(one_hot_ctgrs[i][1]), int(one_hot_ctgrs[i][2]), int(one_hot_ctgrs[i][3])], 2)

            # annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            #
            # cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape, dtype=np.float32)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data

    def convert_img(self, feature, scale):
        feature_imge = feature.reshape((scale, scale, 3)) * 255
        feature_imge = cv2.resize(feature_imge, (640, 640), interpolation=cv2.INTER_NEAREST)
        feature_imge[-1, :] = [255, 255, 255]
        feature_imge[:, -1] = [255, 255, 255]
        return feature_imge
