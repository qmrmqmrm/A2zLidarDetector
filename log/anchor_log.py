import os
import os.path as op
import torch

from train.loss_util import distribute_box_over_feature_map

class AnchorLog:
    def __init__(self, ckpt_path, epoch):
        self.ckpt_path = ckpt_path
        self.epoch = epoch
        self.anchor_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        if not op.isdir(self.anchor_path):
            os.makedirs(self.anchor_path)

    def __call__(self, step, grtr, pred, loss_by_type):
        anchors = pred['anchors']
        pred_objectness_logits = pred['pred_objectness_logits']
        gt_labels, gt_boxes = distribute_box_over_feature_map(anchors, pred['batched_input'], self.anchor_matcher)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        valid_mask = gt_labels >= 0
        obj_pred = torch.cat(pred_objectness_logits, dim=1)[valid_mask]
        obj_gt = gt_labels[valid_mask].to(torch.float32)
        print(obj_gt.shape)
        print(obj_pred.shape)


    def get_result(self):
        pass
