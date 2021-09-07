import os
import os.path as op
import torch
import numpy as np
import cv2

from config import Config as cfg
import utils.util_function as uf


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.vlog_path = op.join(ckpt_path, cfg.Train.CKPT_NAME, "vlog", f"ep{epoch:02d}")

        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)

    def __call__(self, step, grtr, pred, title):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        """
        # uf.print_structure('VisualLog pred', pred)
        # uf.print_structure('VisualLog grtr', grtr)

        for i, image_file in enumerate(grtr['image_file']):
            if title == 'pred_nms':
                proposals = pred[0][i]
                gt_bbox2d = grtr['bbox2d'][i]
                gt_classes = grtr['category'][i]
                proposals_box = proposals['pred_boxes']
                proposals_class = proposals['pred_classes']

            elif title == 'gt_proposal':
                uf.print_structure('grtr', grtr)
                uf.print_structure('pred', pred)
                proposals = pred[i]
                gt_bbox2d = grtr['bbox2d'][i]
                gt_classes = grtr['category'][i]
                uf.print_structure('gt_class', gt_classes)
                uf.print_structure('proposals', proposals)
                proposals_box = proposals['gt_proposal_box']
                proposals_class = proposals['gt_proposals_class']

            fg_car_inds = torch.nonzero((gt_classes == 0)).squeeze(1)
            fg_pad_inds = torch.nonzero((gt_classes == 1)).squeeze(1)
            fg_cyc_inds = torch.nonzero((gt_classes == 2)).squeeze(1)
            pre_car_inds = torch.nonzero((proposals_class == 0)).squeeze(1)
            pre_pad_inds = torch.nonzero((proposals_class == 1)).squeeze(1)
            pre_cyc_inds = torch.nonzero((proposals_class == 2)).squeeze(1)

            img = cv2.imread(image_file)

            camera_file = image_file.replace('/image', '/camera')
            camera = cv2.imread(camera_file)
            camera = cv2.resize(camera, dsize=(1400, 700))
            img_filename = '/'.join(image_file.split('/')[-3:])
            save_path = os.path.join(self.vlog_path, 'total_img', img_filename)
            pred_folder = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(pred_folder):
                os.makedirs(pred_folder)

            pred_img = self.draw_boxes(img, img_filename, proposals_box, pre_car_inds, 'pred', (0, 0, 255))
            pred_img = self.draw_boxes(pred_img, img_filename, proposals_box, pre_pad_inds, 'pred', (0, 255, 0))
            pred_img = self.draw_boxes(pred_img, img_filename, proposals_box, pre_cyc_inds, 'pred', (255, 0, 255))
            gt_img = self.draw_boxes(img, img_filename, gt_bbox2d, fg_car_inds, 'gt', (0, 0, 255))
            gt_img = self.draw_boxes(gt_img, img_filename, gt_bbox2d, fg_pad_inds, 'gt', (0, 255, 0))
            gt_img = self.draw_boxes(gt_img, img_filename, gt_bbox2d, fg_cyc_inds, 'gt', (255, 0, 0))
            total_img = cv2.vconcat([camera, pred_img, gt_img])
            cv2.imwrite(save_path, total_img)

    def draw_boxes(self, img, file_name, boxes, idx, split, color, rotation='xyxy'):
        if idx.numel() == 0:
            return img
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
        for box in draw_boxes:
            box = box.type(torch.int32).cpu().numpy()
            cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(save_path, draw_img)
        return draw_img
