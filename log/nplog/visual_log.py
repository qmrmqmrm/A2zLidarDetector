import math
import os
import os.path as op
import torch
import numpy as np
import cv2

import config as cfg
import utils.util_function as uf
import model.submodules.model_util as mu
from log.nplog.metric import TrueFalseSplitter, RotatedIouEstimator, IouEstimator
from log.nplog.splits import split_rotated_true_false, split_true_false
import log.nplog.visual_function as vf


# TODO: rearrange-code-21-11
class VisualLogNew:
    def __init__(self):
        self.tf_spliter = TrueFalseSplitter(IouEstimator(), cfg.Validation.TP_IOU_THRESH)
        self.tf_spliter_rot = TrueFalseSplitter(RotatedIouEstimator(), cfg.Validation.TP_IOU_THRESH)

    def __call__(self, step, grtr, pred, grtr_feat, pred_feat):
        splits = self.tf_spliter(grtr, pred)
        splits3d = self.tf_spliter_rot(grtr, pred)

        batch = splits["grtr_tp"]["bbox2d"].shape[0]
        for batch_idx in range(batch):
            rot_bbox3ds = self.rotate_box3d(splits3d, batch_idx)
            cam_image = self.get_cam_image()
            bev_image = self.load_image()

            bev_grtr_2d = self.draw_objects(bev_image, splits["grtr_tp"], self.draw_box2d, self.put_text_grtr)
            bev_grtr_2d = self.draw_objects(bev_grtr_2d, splits["grtr_fn"], self.draw_box2d, self.put_text_grtr)

            bev_pred_2d = self.draw_objects(bev_image, splits["pred_tp"], self.draw_box2d, self.put_text_pred)
            bev_pred_2d = self.draw_objects(bev_pred_2d, splits["pred_fp"], self.draw_box2d, self.put_text_pred)
            self.concat_and_save_image(step, cam_image, bev_grtr_2d, bev_pred_2d, "2d")

            bev_grtr_3d = self.draw_objects(bev_image, rot_bbox3ds["grtr_tp"], self.draw_box3d, self.put_text_grtr)
            bev_grtr_3d = self.draw_objects(bev_grtr_3d, rot_bbox3ds["grtr_fn"], self.draw_box3d, self.put_text_grtr)

            bev_pred_3d = self.draw_objects(bev_image, rot_bbox3ds["pred_tp"], self.draw_box3d, self.put_text_pred)
            bev_pred_3d = self.draw_objects(bev_pred_3d, rot_bbox3ds["pred_fp"], self.draw_box3d, self.put_text_pred)
            self.concat_and_save_image(step, cam_image, bev_grtr_3d, bev_pred_3d, "3d")

    def rotate_box3d(self, splits3d, batch_idx):
        splits_keys = splits3d.keys()
        for key in splits_keys:
            bbox3d_per_image = splits3d[key]['bbox3d'][batch_idx]
            yaw_rads_per_image = splits3d[key]['yaw_rads'][batch_idx]

            rot_bbox3d_per_img = list()

            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                rot_bbox3d = vf.rotated_box2d(bbox3d[:4], yaw_rads)
                rot_bbox3d_per_img.append(rot_bbox3d)

            splits3d[key]['rot_bbox3d'] = np.stack(rot_bbox3d_per_img, axis=0)
        return splits3d

    def get_cam_image(self):
        pass

    def load_image(self):
        pass

    def draw_objects(self, img, splits, object, put_text):
        pass

    def draw_box2d(self):
        pass

    def draw_box3d(self, img, corners, category):
        shape = category.shape

        if shape[-1] > 1:
            best_cate = np.argmax(category, axis=-1)
            category = np.expand_dims(best_cate, -1)
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val

        for idx, corner in enumerate(corners):
            if int(corner[1][0]) - int(corner[0][0]) == 0 and int(corner[1][1]) - int(corner[0][1]) == 0:
                continue
            corner_num = corner.shape[0]
            for corner_idx in range(corner.shape[0]):
                cv2.line(img,
                         (int(corner[corner_idx][0]),
                          int(corner[corner_idx][1])),
                         (int(corner[((corner_idx + 1) % corner_num)][0]),
                          int(corner[((corner_idx + 1) % corner_num)][1])),
                         color[int(category[idx])], 2)
        return img

    def put_text_grtr(self):
        pass

    def put_text_pred(self):
        pass

    def concat_and_save_image(self, step, img, grtr, pred, title):
        pass


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
        self.tf_spliter = TrueFalseSplitter(IouEstimator(), cfg.Validation.TP_IOU_THRESH)
        self.tf_spliter_rot = TrueFalseSplitter(RotatedIouEstimator(), cfg.Validation.TP_IOU_THRESH)

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
        splits = self.tf_spliter(grtr, pred)
        splits_rot = self.tf_spliter_rot(grtr, pred)

        # splits = split_true_false(grtr, pred_nms, cfg.Validation.TP_IOU_THRESH)
        # splits_rot = split_rotated_true_false(grtr, pred_nms, cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["bbox2d"].shape[0]

        for batch_idx in range(batch):
            bev_img = grtr["image"][batch_idx].copy()
            # bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY) * 255
            # bev_img = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
            image_org_file_name = grtr["image_file"][batch_idx].replace('image/', 'camera/')
            image_org = cv2.imread(image_org_file_name)
            image_org = cv2.resize(image_org, (1280, 640))

            #
            for key in ['bbox2d']:
                bbox_image = self.draw_bbox(bev_img, splits, batch_idx, key)
                vlog_image = np.concatenate([image_org, bbox_image], axis=0)
                filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_{key}.jpg")
                cv2.imwrite(filename, vlog_image)

            # object
            # obj_img = self.draw_object(bev_img, gt_feature['object'], pred['rpn_feat_objectness'], batch_idx)
            # filename = op.join(self.obj_path, f"{step * batch + batch_idx:05d}.jpg")
            # cv2.imwrite(filename, obj_img)

            # rotation
            # rotated_img = self.draw_rotated(bev_img, splits, batch_idx)
            # rotated_img = np.concatenate([image_org, rotated_img], axis=0)
            # filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_rot.jpg")
            # cv2.imwrite(filename, rotated_img)

            rotated_img = self.draw_split_rotated(bev_img, splits_rot, batch_idx)
            rotated_img = np.concatenate([image_org, rotated_img], axis=0)
            filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_rot_iou.jpg")
            cv2.imwrite(filename, rotated_img)

            img_3d = self.draw_grtr_bbox_3d(image_org, splits_rot, batch_idx)
            filename = op.join(self.vlog_path, f"{step:04d}_{step * batch + batch_idx:05d}_3d.jpg")
            cv2.imwrite(filename, img_3d)
            # cv2.imshow("detection_result", rotated_img)
            # cv2.waitKey(1)

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

    def draw_object(self, bev_img, gt_object_feature, pred_objectness_feat, batch_idx):
        # object
        gt_obj_imgs = []
        pred_obj_imgs = []
        for scale_idx, (gt_scale_object, pred_scale_object) in enumerate(zip(gt_object_feature, pred_objectness_feat)):
            org_img = bev_img.copy()
            scale = int(640 / cfg.Scales.DEFAULT_FEATURE_SCALES[scale_idx])
            gt_object_per_image = self.convert_img(gt_scale_object[batch_idx], scale, org_img)
            pred_object_per_image = self.convert_img(pred_scale_object[batch_idx], scale, org_img)
            gt_obj_imgs.append(gt_object_per_image)
            pred_obj_imgs.append(pred_object_per_image)
        gt_obj_img = np.concatenate(gt_obj_imgs, axis=1)
        pred_obj_img = np.concatenate(pred_obj_imgs, axis=1)
        obj_img = np.concatenate([gt_obj_img, pred_obj_img], axis=0)
        return obj_img

    def convert_img(self, feature, scale, org_img):
        feature_imge = feature.reshape((scale, scale, 3)) * 255
        feature_imge = org_img + cv2.resize(feature_imge, (640, 640), interpolation=cv2.INTER_NEAREST)
        feature_imge[-1, :] = [255, 255, 255]
        feature_imge[:, -1] = [255, 255, 255]
        return feature_imge

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

    def draw_split_rotated(self, bev_img, splits, batch_idx):
        rot_bbox3ds = dict()
        splits_keys = splits.keys()
        for key in splits_keys:
            bbox3d_per_image = splits[key]['bbox3d'][batch_idx]
            yaw_rads_per_image = splits[key]['yaw_rads'][batch_idx]

            rot_bbox3d_per_img = list()

            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                rot_bbox3d = vf.rotated_box(bbox3d[:4], yaw_rads)
                rot_bbox3d_per_img.append(rot_bbox3d)

            rot_bbox3ds[key] = np.stack(rot_bbox3d_per_img, axis=0)
        image_grtr = bev_img.copy()
        print('grtr_fn', splits["grtr_fn"]['iou'][batch_idx])
        image_grtr = self.draw_rotated_box(image_grtr, rot_bbox3ds["grtr_fn"], 2,
                                           splits["grtr_fn"]['category'][batch_idx],
                                           splits["grtr_fn"]['iou'][batch_idx],
                                           )
        print('grtr_tp', splits["grtr_tp"]['iou'][batch_idx])
        image_grtr = self.draw_rotated_box(image_grtr, rot_bbox3ds["grtr_tp"], 1,
                                           splits["grtr_tp"]['category'][batch_idx],
                                           splits["grtr_tp"]['iou'][batch_idx],
                                           )
        image_pred = bev_img.copy()

        image_pred = self.draw_rotated_box(image_pred, rot_bbox3ds["pred_fp"], 2,
                                           splits["pred_fp"]['ctgr_probs'][batch_idx],
                                           )
        image_pred = self.draw_rotated_box(image_pred, rot_bbox3ds["pred_tp"], 1,
                                           splits["pred_tp"]['ctgr_probs'][batch_idx],
                                           )

        vlog_image = np.concatenate([image_pred, image_grtr], axis=1)
        return vlog_image

    def draw_rotated_box(self, img, corners, val=1, category=None, *args):
        shape = category.shape

        if shape[-1] > 1:
            best_cate = np.argmax(category, axis=-1)
            category = np.expand_dims(best_cate, -1)
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val

        for idx, corner in enumerate(corners):
            ann = f'data: '
            if int(corner[1][0]) - int(corner[0][0]) == 0 and int(corner[1][1]) - int(corner[0][1]) == 0:
                continue
            print('corner.shape[0]', corner.shape[0])
            corner_num = corner.shape[0]
            for corner_idx in range(corner.shape[0]):
                cv2.line(img,
                         (int(corner[corner_idx][0]),
                          int(corner[corner_idx][1])),
                         (int(corner[((corner_idx + 1) % corner_num)][0]),
                          int(corner[((corner_idx + 1) % corner_num)][1])),
                         color[int(category[idx])], 2)

            if args is not None:
                for arg in args:
                    ann = ann + f' {arg[idx, 0]:1.3f}'
                cv2.putText(img, ann, (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            color[int(category[idx])], 2)
        return img

    def draw_grtr_bbox_3d(self, org_img, splits, batch_idx):
        bbox_3d = dict()
        for split in ['grtr_fn', 'grtr_tp']:
            bbox3d_per_image = splits[split]['bbox3d'][batch_idx]
            yaw_rads_per_image = splits[split]['yaw_rads'][batch_idx]
            bbox3d_per_img = list()
            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                print('bbox3d, yaw_rads', bbox3d.shape, yaw_rads.shape)
                rot_bbox3d = vf.get_3d_points(bbox3d, yaw_rads)
                bbox3d_per_img.append(rot_bbox3d)
            bbox_3d[split] = np.stack(bbox3d_per_img, axis=0)
        image_grtr = org_img.copy()
        image_grtr = self.draw_rotated_box(image_grtr, bbox_3d["grtr_fn"], 2,
                                           splits["grtr_fn"]['category'][batch_idx],
                                           splits["grtr_fn"]['iou'][batch_idx],
                                           )
        image_grtr = self.draw_rotated_box(image_grtr, bbox_3d["grtr_tp"], 1,
                                           splits["grtr_tp"]['category'][batch_idx],
                                           splits["grtr_tp"]['iou'][batch_idx],
                                           )
        return image_grtr
