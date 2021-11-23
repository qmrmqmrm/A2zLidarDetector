import math
import os
import os.path as op
import torch
import numpy as np
import cv2

import config as cfg
from log.nplog.metric import split_true_false, split_rotated_true_false
import utils.util_function as uf
import model.submodules.model_util as mu


class VisualLog:
    def __init__(self, ckpt_path, epoch, split):
        # self.grtr_log_keys = cfg.Train.LOG_KEYS
        # self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog", split, f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)

        # self.obj_path = op.join(self.vlog_path, 'object')
        # if not op.isdir(self.obj_path):
        #     os.makedirs(self.obj_path)

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
        print('VisualLog')
        splits = split_true_false(grtr, pred_nms, cfg.Validation.TP_IOU_THRESH)
        splits_rot = split_rotated_true_false(grtr, pred_nms, cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["bbox2d"].shape[0]

        for batch_idx in range(batch):
            bev_img = grtr["image"][batch_idx].copy()
            # bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY) * 255
            # bev_img = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)

            # filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_gray.jpg")
            # cv2.imwrite(filename, bev_img)
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_org_file_name = grtr["image_file"][batch_idx].replace('image/', 'camera/')
            image_org = cv2.imread(image_org_file_name)
            image_org = cv2.resize(image_org, (1280, 640))
            #
            for key in ['bbox2d']:
                bbox_image = self.draw_bbox(bev_img, splits, batch_idx, key)
                vlog_image = np.concatenate([image_org, bbox_image], axis=0)
                filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_{key}.jpg")
                cv2.imwrite(filename, vlog_image)
            #
            # # object
            # gt_obj_imgs = []
            # pred_obj_imgs = []
            # for scale_idx, (gt_scale_object, pred_scale_object) in enumerate(
            #         zip(gt_feature['object'], pred['rpn_feat_objectness'])):
            #     org_img = bev_img.copy()
            #     scale = int(640 / cfg.Scales.DEFAULT_FEATURE_SCALES[scale_idx])
            #     gt_object_per_image = self.convert_img(gt_scale_object[batch_idx], scale, org_img)
            #     pred_object_per_image = self.convert_img(pred_scale_object[batch_idx], scale, org_img)
            #     gt_obj_imgs.append(gt_object_per_image)
            #     pred_obj_imgs.append(pred_object_per_image)
            # gt_obj_img = np.concatenate(gt_obj_imgs, axis=1)
            # pred_obj_img = np.concatenate(pred_obj_imgs, axis=1)
            # obj_img = np.concatenate([gt_obj_img, pred_obj_img], axis=0)
            # filename = op.join(self.obj_path, f"{step * batch + batch_idx:05d}.jpg")
            # cv2.imwrite(filename, obj_img)

            # rotation
            rotated_img = self.draw_rotated(bev_img,splits, batch_idx)
            rotated_img = np.concatenate([image_org, rotated_img], axis=0)
            filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_rot.jpg")
            cv2.imwrite(filename, rotated_img)
            # vlog_image_show = cv2.resize(vlog_image, (920,920))
            rotated_img = self.draw_rotated(bev_img,splits_rot, batch_idx)
            rotated_img = np.concatenate([image_org, rotated_img], axis=0)
            filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_rot_iou.jpg")
            cv2.imwrite(filename, rotated_img)
            cv2.imshow("detection_result", rotated_img)
            cv2.waitKey(1)

    def draw_bbox(self, bev_img, splits, batch_idx, key):
        bev_img_pred = bev_img.copy()
        bev_img_pred = self.draw_ctgr_boxes(bev_img_pred, splits["pred_fp"][key],
                                            splits["pred_fp"]['ctgr_probs'],
                                            batch_idx, 2)
        bev_img_pred = self.draw_ctgr_boxes(bev_img_pred, splits["pred_tp"][key],
                                            splits["pred_tp"]['ctgr_probs'],
                                            batch_idx, 1)
        bev_img_grtr = bev_img.copy()
        bev_img_grtr = self.draw_ctgr_boxes(bev_img_grtr, splits["grtr_fn"][key],
                                            splits["grtr_fn"]['category'],
                                            batch_idx, 2, splits["grtr_fn"]['iou'], )
        bev_img_grtr = self.draw_ctgr_boxes(bev_img_grtr, splits["grtr_tp"][key],
                                            splits["grtr_tp"]['category'],
                                            batch_idx, 1, splits["grtr_tp"]['iou'], )
        vlog_image = np.concatenate([bev_img_pred, bev_img_grtr], axis=1)

        return vlog_image

    def draw_rotated(self, bev_img, splits, batch_idx):
        rot_bbox3ds = dict()
        splits_keys = splits.keys()
        for key in splits_keys:
            bbox3d_per_image = splits[key]['bbox3d'][batch_idx]
            yaw_rads_per_image = splits[key]['yaw_rads'][batch_idx]

            rot_bbox3d_per_img = list()

            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                rot_bbox3d = self.rotated_3d(bbox3d[:4], yaw_rads, batch_idx)
                rot_bbox3d_per_img.append(rot_bbox3d)
            rot_bbox3ds[key] = np.stack(rot_bbox3d_per_img, axis=0)
        image_grtr = bev_img.copy()
        image_grtr = self.draw_3D_box(image_grtr, rot_bbox3ds["grtr_fn"],
                                      splits["grtr_fn"]['yaw_cls'][batch_idx],
                                      splits["grtr_fn"]['yaw_rads'][batch_idx],
                                      splits["grtr_fn"]['category'][batch_idx],
                                      val=2)
        image_grtr = self.draw_3D_box(image_grtr, rot_bbox3ds["grtr_tp"],
                                      splits["grtr_tp"]['yaw_cls'][batch_idx],
                                      splits["grtr_tp"]['yaw_rads'][batch_idx],
                                      splits["grtr_tp"]['category'][batch_idx],
                                      val=1)
        image_pred = bev_img.copy()
        image_pred = self.draw_3D_box(image_pred, rot_bbox3ds["pred_fp"],
                                      splits["pred_fp"]['yaw_cls_probs'][batch_idx],
                                      splits["pred_fp"]['yaw_rads'][batch_idx],
                                      splits["pred_fp"]['ctgr_probs'][batch_idx],
                                      splits["pred_fp"]['object'][batch_idx],
                                      val=2)
        image_pred = self.draw_3D_box(image_pred, rot_bbox3ds["pred_tp"],
                                      splits["pred_tp"]['yaw_cls_probs'][batch_idx],
                                      splits["pred_tp"]['yaw_rads'][batch_idx],
                                      splits["pred_tp"]['ctgr_probs'][batch_idx],
                                      splits["pred_tp"]['object'][batch_idx],
                                      val=1)

        vlog_image = np.concatenate([image_pred, image_grtr], axis=1)
        return vlog_image

    def draw_boxes(self, image, bboxes, frame_idx, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """
        bbox2d = bboxes[frame_idx]  # (N, 4)
        valid_mask = bbox2d[:, 2] > 0  # (N,) l>0

        bbox2d = bbox2d[valid_mask, :]
        bbox2d = mu.convert_box_format_yxhw_to_tlbr(bbox2d)
        for i in range(bbox2d.shape[0]):
            y1, x1, y2, x2 = bbox2d[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return image

    def draw_ctgr_boxes(self, image, bboxes, category, frame_idx, val, iou=None):
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
        bbox2d = bboxes[frame_idx][..., :4]  # (N, 4)
        category = category[frame_idx]
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val
        valid_mask = bbox2d[:, 2] > 0  # (N,) h>0
        bbox2d = bbox2d[valid_mask, :]
        category = category[valid_mask, :]
        if iou is not None:
            iou = iou[frame_idx]
            iou = iou[valid_mask, :]
        bbox2d = mu.convert_box_format_yxhw_to_tlbr(bbox2d)
        for i in range(bbox2d.shape[0]):
            y1, x1, y2, x2 = bbox2d[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color[int(category[i])], 2)
            if iou is not None:
                ann = f'{iou[i]}'
                cv2.putText(image, ann, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color[int(category[i])], 2)
        return image

    def rotated_3d(self, bboxes_yxlw, yaw_rads, frame_idx):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param color: box color
        :return: box drawn image
        """

        bbox3d_yx, bbox3d_lw = bboxes_yxlw[:2], bboxes_yxlw[2:4]
        bbox3d_xy = np.array([bbox3d_yx[1], bbox3d_yx[0]])
        corners = np.array([[- bbox3d_lw[1] / 2., - bbox3d_lw[0] / 2.],
                            [+ bbox3d_lw[1] / 2., - bbox3d_lw[0] / 2.],
                            [+ bbox3d_lw[1] / 2., + bbox3d_lw[0] / 2.],
                            [- bbox3d_lw[1] / 2., + bbox3d_lw[0] / 2.]])
        # yaw = -(yaw_rads[0] + (math.pi / 2))
        c, s = np.cos(yaw_rads[0]), np.sin(yaw_rads[0])
        R = np.array([[c, -s],
                      [-s, -c]])  # image coordinates: flip y
        rotated_corners = np.dot(corners, R) + bbox3d_xy
        return rotated_corners

    def draw_3D_box(self, img, corners, bin, rad, category=None, obj_prob=None, val=1):
        """
            draw 3D box in image with OpenCV,
            the order of the corners should be the same with BBox3dProjector
        """
        # color = [(255,255,255),(0,0,255),(0,255,0),(255,0,0)]
        shape = category.shape

        if shape[-1] > 1:
            best_cate = np.argmax(category, axis=-1)
            category = np.expand_dims(best_cate, -1)
        bin = np.argmax(bin, axis=-1)
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val

        for idx, corner in enumerate(corners):

            if int(corner[1][0]) - int(corner[0][0]) == 0 and int(corner[1][1]) - int(corner[0][1]) == 0:
                continue

            for corner_idx in range(corner.shape[0]):
                cv2.line(img, (int(corner[corner_idx][0]), int(corner[corner_idx][1])),
                         (int(corner[((corner_idx + 1) % 4)][0]), int(corner[((corner_idx + 1) % 4)][1])),
                         color[int(category[idx])], 2)
            # ann = f'{bin[idx]},{rad[idx, 0]:1.3f}'
            # cv2.putText(img, ann, (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
            # if obj_prob is not None:
            #     ctgr_idx = np.argmax(category, axis=-1)
            #     ctgr_prob = np.amax(category, axis=-1)
            #     cate_ann = f'{ctgr_prob[idx]:3.3f}, {obj_prob[idx, 0]:3.3f}, {int(ctgr_idx[idx])} '
            #     cv2.putText(img, cate_ann, (int(corner[2][0]), int(corner[2][1])), cv2.FONT_HERSHEY_PLAIN, 1.0,
            #                 color, 2)

        return img

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape, dtype=np.float32)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data

    def convert_img(self, feature, scale, org_img):

        feature_imge = feature.reshape((scale, scale, 3)) * 255
        feature_imge = org_img + cv2.resize(feature_imge, (640, 640), interpolation=cv2.INTER_NEAREST)
        feature_imge[-1, :] = [255, 255, 255]
        feature_imge[:, -1] = [255, 255, 255]
        return feature_imge
