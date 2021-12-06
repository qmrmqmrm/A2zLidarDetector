import math
import os
import os.path as op
import torch
import numpy as np
import cv2

import config as cfg
import utils.util_function as uf
import model.submodules.model_util as mu
from log.nplog.metric import TrueFalseSplitter, RotatedIouEstimator, IouEstimator, Rotated3DIouEstimator
from log.nplog.splits import split_rotated_true_false, split_true_false
import log.nplog.visual_function as vf


# TODO: rearrange-code-21-11
class VisualLog:
    def __init__(self, ckpt_path, epoch, split):

        self.vlog_path = op.join(ckpt_path, "vlog", split, f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        self.tf_spliter = TrueFalseSplitter(IouEstimator(), cfg.Validation.TP_IOU_THRESH)
        self.tf_spliter_rot = TrueFalseSplitter(RotatedIouEstimator(), cfg.Validation.TP_IOU_THRESH)
        self.tf_spliter_3D = TrueFalseSplitter(Rotated3DIouEstimator(), cfg.Validation.TP_IOU_THRESH)
        self.put_text_grtr = ['iou', 'yaw_rads']
        self.put_text_pred = ['object', 'yaw_rads']

    def __call__(self, step, grtr, pred, grtr_feat, pred_feat):
        splits = self.tf_spliter(grtr, pred)
        splits_rot = self.tf_spliter_rot(grtr, pred)
        splits3d = self.tf_spliter_3D(grtr, pred)
        batch = splits["grtr_tp"]["bbox2d"].shape[0]
        for batch_idx in range(batch):
            bev_file_name = grtr["image_file"][batch_idx]
            splits_batch = self.get_batch_splits(splits, batch_idx)
            splitsrot_batch = self.get_batch_splits(splits_rot, batch_idx)
            splits3d_batch = self.get_batch_splits(splits3d, batch_idx)

            splitsrot_batch = self.rotate_box2d(splitsrot_batch)
            splits3d_batch = self.rotate_box3d(splits3d_batch)
            cam_image = self.get_cam_image(bev_file_name)
            bev_image = self.load_image(bev_file_name)

            bev_gt_object = bev_image.copy()
            bev_gt_object = self.draw_objects(bev_gt_object, grtr_feat['object'],
                                              self.draw_objectness,
                                              val=batch_idx)
            bev_pred_object = bev_image.copy()
            bev_pred_object = self.draw_objects(bev_pred_object, pred_feat['objectness'],
                                                self.draw_objectness,
                                                val=batch_idx)
            vlog_image = np.concatenate([bev_gt_object, bev_pred_object], axis=0)
            file_dir = op.join(self.vlog_path, 'obj')
            if not op.isdir(file_dir):
                os.makedirs(file_dir)
            filename = op.join(file_dir, f"{step * batch + batch_idx:05d}_obj.jpg")
            cv2.imwrite(filename, vlog_image)
            bev_grtr_2d = bev_image.copy()
            bev_grtr_2d = self.draw_objects(bev_grtr_2d, splits_batch["grtr_fn"],
                                            self.draw_box2d,
                                            self.put_text_grtr, 2)
            bev_grtr_2d = self.draw_objects(bev_grtr_2d, splits_batch["grtr_tp"],
                                            self.draw_box2d,
                                            self.put_text_grtr, 1)
            bev_pred_2d = bev_image.copy()
            bev_pred_2d = self.draw_objects(bev_pred_2d, splits_batch["pred_fp"],
                                            self.draw_box2d,
                                            self.put_text_pred, 2)
            bev_pred_2d = self.draw_objects(bev_pred_2d, splits_batch["pred_tp"],
                                            self.draw_box2d,
                                            self.put_text_pred, 1)
            self.concat_and_save_image(step, cam_image, bev_grtr_2d, bev_pred_2d, batch, batch_idx, "2d")

            bev_grtr_3d = bev_image.copy()
            bev_grtr_3d = self.draw_objects(bev_grtr_3d, splitsrot_batch["grtr_fn"],
                                            self.draw_box3d,
                                            self.put_text_grtr, 2)
            bev_grtr_3d = self.draw_objects(bev_grtr_3d, splitsrot_batch["grtr_tp"],
                                            self.draw_box3d,
                                            self.put_text_grtr, 1)
            bev_pred_3d = bev_image.copy()
            bev_pred_3d = self.draw_objects(bev_pred_3d, splitsrot_batch["pred_fp"],
                                            self.draw_box3d,
                                            self.put_text_pred, 2)
            bev_pred_3d = self.draw_objects(bev_pred_3d, splitsrot_batch["pred_tp"],
                                            self.draw_box3d,
                                            self.put_text_pred, 1)
            self.concat_and_save_image(step, cam_image, bev_grtr_3d, bev_pred_3d, batch, batch_idx, "rot")

            # cam_grtr_3d = cam_image.copy()
            # cam_grtr_3d = self.draw_objects(cam_grtr_3d, splits3d_batch["grtr_fn"],
            #                                 self.draw_box3d,
            #                                 self.put_text_grtr, 2)
            # cam_grtr_3d = self.draw_objects(cam_grtr_3d, splits3d_batch["grtr_tp"],
            #                                 self.draw_box3d,
            #                                 self.put_text_grtr, 1)
            # cam_pred_3d = cam_image.copy()
            # cam_pred_3d = self.draw_objects(cam_pred_3d, splits3d_batch["pred_fp"],
            #                                 self.draw_box3d,
            #                                 self.put_text_pred, 2)
            # cam_pred_3d = self.draw_objects(cam_pred_3d, splits3d_batch["pred_tp"],
            #                                 self.draw_box3d,
            #                                 self.put_text_pred, 1)
            # vlog_image = np.concatenate([cam_grtr_3d, cam_pred_3d], axis=0)
            #
            # filename = op.join(self.vlog_path, f"{step * batch + batch_idx:05d}_3d.jpg")
            # cv2.imwrite(filename, vlog_image)

    def get_batch_splits(self, splits, batch_idx):
        batch_splits = splits.copy()
        for splits_key, split in splits.items():
            for key in split:
                batch_splits[splits_key][key] = splits[splits_key][key][batch_idx]
            catgory = batch_splits[splits_key]['category']
            ctgr_shape = catgory.shape
            best_cate = np.argmax(catgory, axis=-1) if ctgr_shape[-1] > 1 else catgory
            batch_splits[splits_key]['category'] = np.expand_dims(best_cate, -1)
        return batch_splits

    def rotate_box2d(self, splits3d):
        splits_keys = splits3d.keys()
        for key in splits_keys:
            bbox3d_per_image = splits3d[key]['bbox3d']
            yaw_rads_per_image = splits3d[key]['yaw_rads']

            rot_bbox3d_per_img = list()

            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                rot_bbox3d = vf.rotated_box2d(bbox3d[:4], yaw_rads)
                rot_bbox3d_per_img.append(rot_bbox3d)

            splits3d[key]['bbox3d'] = np.stack(rot_bbox3d_per_img, axis=0)
        return splits3d

    def rotate_box3d(self, splits3d):
        splits_keys = splits3d.keys()
        for key in splits_keys:
            bbox3d_per_image = splits3d[key]['bbox3d']
            yaw_rads_per_image = splits3d[key]['yaw_rads']

            rot_bbox3d_per_img = list()

            for bbox3d, yaw_rads in zip(bbox3d_per_image, yaw_rads_per_image):
                rot_bbox3d = vf.rotated_box3d(bbox3d, yaw_rads)
                rot_bbox3d_per_img.append(rot_bbox3d)

            splits3d[key]['bbox3d'] = np.stack(rot_bbox3d_per_img, axis=0)
        return splits3d

    def get_cam_image(self, bev_file_name):
        image_org_file_name = bev_file_name.replace('image/', 'camera/')
        image_org = cv2.imread(image_org_file_name)
        image_org = cv2.resize(image_org, (1280, 640))
        return image_org

    def load_image(self, bev_file_name, white=False):
        bev_img = cv2.imread(bev_file_name)
        if white:
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY) * 255
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
        return bev_img

    def draw_objects(self, img, splits, draw_func, put_text=None, val=1):
        text_splits = {key: splits[key] for key in put_text if put_text is not None} if put_text is not None else put_text
        bbox_image = draw_func(img, splits, val, text_splits)
        return bbox_image

    def draw_objectness(self, bev_img, object_feature, batch_idx, kwargs=None):
        # object
        scale_imgs = []
        for scale_idx, (scale_object) in enumerate(zip(object_feature)):
            org_img = bev_img.copy()
            scale = int(640 / cfg.Scales.DEFAULT_FEATURE_SCALES[scale_idx])
            gt_object_per_image = vf.convert_img(scale_object[batch_idx], scale, org_img)
            scale_imgs.append(gt_object_per_image)
        obj_img = np.concatenate(scale_imgs, axis=1)
        return obj_img

    def draw_box2d(self, img, splits, val, kwargs):
        bbox2d = splits['bbox2d']
        category = splits['category']
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val
        valid_mask = bbox2d[:, 2] > 0  # (N,) h>0
        bbox2d = bbox2d[valid_mask, :]
        category = category[valid_mask, :]
        bbox2d = mu.convert_box_format_yxhw_to_tlbr(bbox2d)
        for idx in range(bbox2d.shape[0]):
            ann = f''
            y1, x1, y2, x2 = bbox2d[idx].astype(np.int32)
            cv2.rectangle(img, (x1, y1), (x2, y2), color[int(category[idx])], 2)
            for key, arg in kwargs.items():
                ann += f'{key}: {arg[idx, 0]:1.3f}'
            cv2.putText(img, ann, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color[int(category[idx])], 2)
        return img

    def draw_box3d(self, img, splits, val, kwargs):
        corners = splits['bbox3d']
        category = splits['category']
        color = np.array([(125, 125, 125), (255, 125, 0), (0, 255, 0), (0, 0, 255)]) / val
        for idx, corner in enumerate(corners):
            ann = f''
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
            for key, arg in kwargs.items():
                ann += f'{key}: {arg[idx, 0]:1.3f}'
            cv2.putText(img, ann, (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        color[int(category[idx])], 2)
        return img

    def concat_and_save_image(self, step, org_img, grtr_img, pred_img, batch, batch_idx, title):
        vlog_image = np.concatenate([grtr_img, pred_img], axis=1)
        vlog_image = np.concatenate([org_img, vlog_image], axis=0)
        file_dir = op.join(self.vlog_path, title)
        if not op.isdir(file_dir):
            os.makedirs(file_dir)
        filename = op.join(file_dir, f"{step * batch + batch_idx:05d}_{title}.jpg")

        cv2.imwrite(filename, vlog_image)
