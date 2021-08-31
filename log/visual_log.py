import os
import os.path as op
import torch
import numpy as np
import cv2

from config import Config as cfg


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.vlog_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        """
        head_proposals = pred['head_proposals']
        gt_bbox2d = torch.cat([p['bbox2d'] for p in head_proposals])
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        proposals = torch.cat([p['proposal_boxes'] for p in head_proposals])

        bg_class_ind = cfg.Model.Structure.NUM_CLASSES
        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)

        for i, image_file in enumerate(grtr['image_file']):
            index = fg_inds[torch.nonzero((fg_inds >= (512 * i)) & (fg_inds < (512 * (i + 1)))).squeeze(1)]
            img_filename = '/'.join(image_file.split('/')[-3:])
            img = cv2.imread(image_file)

            self.draw_boxes(img, img_filename, proposals, index, 'pred', (255, 255, 255))
            self.draw_boxes(img, img_filename, gt_bbox2d, index, 'gt', (255, 255, 255))

    def draw_boxes(self, img, file_name, boxes, index, split, color, rotation='xyxy'):
        draw_img = img.copy()
        draw_boxes = boxes
        if rotation == 'xywh':
            draw_boxes[:, 0] = boxes[:, 0]
            draw_boxes[:, 1] = boxes[:, 1]
            draw_boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            draw_boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        elif rotation == 'cwh':
            draw_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            draw_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            draw_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            draw_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        elif rotation == 'xyxy':
            draw_boxes[:, 0] = boxes[:, 0]
            draw_boxes[:, 1] = boxes[:, 1]
            draw_boxes[:, 2] = boxes[:, 2]
            draw_boxes[:, 3] = boxes[:, 3]
        save_path = os.path.join(self.vlog_path, split, file_name)
        pred_folder = '/'.join(save_path.split('/')[:-1])

        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        for idx in index:
            box = draw_boxes[idx, :].type(torch.int32).cpu().numpy()
            print(box)
            cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(save_path, draw_img)
