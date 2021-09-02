import torch
import torch.nn.functional as F
import numpy as np
import math

from config import Config as cfg
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.matcher import Matcher
import train.loss_util as lu
import utils.util_function as uf


class LossBase:
    def __init__(self):
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.anchor_iou_thresh = cfg.Model.RPN.IOU_THRESHOLDS
        self.device = cfg.Model.Structure.DEVICE
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING
        self.anchor_matcher = Matcher(
            cfg.Model.RPN.IOU_THRESHOLDS, cfg.Model.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.weights_height = cfg.Model.Structure.WEIGHTS_HEIGHT

    def __call__(self, features, pred, auxi):
        """

        :param features:
            {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}
        :param pred:
                pred :{'head_class_logits': torch.Size([batch * 512, 4])
                      'bbox_3d_logits': torch.Size([batch * 512, 12])
                      'head_yaw_logits': torch.Size([batch * 512, 36])
                      'head_yaw_residuals': torch.Size([batch * 512, 36])
                      'head_height_logits': torch.Size([batch * 512, 6])

                      'head_proposals': [{'proposal_boxes': torch.Size([512, 4])
                                          'objectness_logits': torch.Size([512])
                                          'gt_category': torch.Size([512, 1])
                                          'bbox2d': torch.Size([512, 4])
                                          'bbox3d': torch.Size([512, 6])
                                          'object': torch.Size([512, 1])
                                          'yaw': torch.Size([512])
                                          'yaw_rads': torch.Size([512])} * batch]

                      'rpn_proposals': [{'proposal_boxes': torch.Size([2000, 4]),
                                        'objectness_logits': torch.Size([2000])} * batch]

                      'pred_objectness_logits' : [torch.Size([batch, 557568(176 * 352 * 9)]),
                                                  torch.Size([batch, 139392(88 * 176 * 9)]),
                                                  torch.Size([batch, 34848(44 * 88 * 9)])]

                      'pred_anchor_deltas' : [torch.Size([batch, 557568(176 * 352 * 9), 4]),
                                              torch.Size([batch, 139392(88 * 176 * 9), 4]),
                                              torch.Size([batch, 34848(44 * 88 * 9), 4])]

                      'anchors' : [torch.Size([557568(176 * 352 * 9), 4])
                                   torch.Size([139392(88 * 176 * 9), 4])
                                   torch.Size([34848(44 * 88 * 9), 4])]
        :return:
        """


class Box2dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        anchors = pred['anchors']
        pred_anchor_deltas = pred['pred_anchor_deltas']
        bbox2d = features['bbox2d']

        gt_labels, gt_boxes = lu.distribute_box_over_feature_map(anchors, bbox2d, self.anchor_matcher)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        anchors = torch.cat(anchors)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, box) for box in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        pos_mask = gt_labels == 1
        loss = F.smooth_l1_loss(torch.cat(pred_anchor_deltas, dim=1)[pos_mask],
                                gt_anchor_deltas[pos_mask],
                                reduction='sum', beta=0.5)
        return loss


class Box3dRegression(LossBase):
    def __call__(self, features, pred, auxi):

        gt_box3d = auxi["gt_matched"]["bbox3d"][:, :4]
        gt_classes = auxi["gt_matched"]["gt_category"]
        pred_box3d = auxi["pred_matched"]["bbox_3d_logits"]
        proposal_boxes = auxi['pred_matched']['bbox2d']
        gt_proposal_deltas = self.box2box_transform.get_deltas(proposal_boxes, gt_box3d, True)
        print('gt_classes', gt_classes.shape)
        box_dim = gt_box3d.size(1)
        category_cols = (box_dim * torch.unsqueeze(gt_classes, 1) + torch.arange(box_dim).to(self.device)).type(torch.int64)
        rows = [[i] * category_cols.size(1) for i in range(category_cols.size(0))]
        pred_box3d_extracted = pred_box3d[rows, category_cols]
        loss = F.smooth_l1_loss(pred_box3d_extracted, gt_proposal_deltas, reduction='sum', beta=0.5)
        return loss / gt_classes.numel()


class HeightRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_height = auxi["gt_matched"]["bbox3d"][:, -2:]
        gt_classes = auxi["gt_matched"]["gt_category"]
        pred_height = auxi["pred_matched"]["height"]
        gt_height_deltas = self.get_h_deltas(gt_height, gt_classes)
        box_dim = gt_height_deltas.size(1)
        category_cols = (box_dim * torch.unsqueeze(gt_classes, 1) + torch.arange(box_dim).to(self.device)).type(torch.int64)
        rows = [[i] * category_cols.size(1) for i in range(category_cols.size(0))]
        pred_height_extracted = pred_height[rows, category_cols]
        loss = F.smooth_l1_loss(pred_height_extracted, gt_height_deltas, reduction='sum', beta=0.5)
        return loss / gt_classes.numel()

    def get_h_deltas(self, gt_height, gt_classes):
        src_heights = torch.tensor([130.05, 149.6, 147.9, 1.0]).to(self.device)  # Mean heights encoded

        target_heights = gt_height[:, 0].to(self.device)
        target_ctr = gt_height[:, 1].to(self.device)

        wh, wg, wz = self.weights_height
        dh = wh * torch.log(target_heights / src_heights[gt_classes.long()])
        dz = wz * (target_ctr - src_heights[gt_classes.long()] / 2.) / src_heights[gt_classes.long()]

        deltas = torch.stack((dh, dz), dim=1, ).to(cfg.Model.Structure.DEVICE)
        return deltas


class YawRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw = auxi["gt_matched"]["yaw"]
        gt_yaw_rads = auxi["gt_matched"]["yaw_rads"]
        gt_classes = auxi["gt_matched"]["gt_category"]
        pred_yaw_residuals = auxi["pred_matched"]["yaw_residuals"]
        gt_yaw_deltas = self.get_vp_deltas(gt_yaw, gt_yaw_rads)

        category_cols = (self.vp_bins * gt_classes + gt_yaw).type(torch.int64)
        rows = list(range(category_cols.size(0)))
        pred_yaw_rads_extracted = pred_yaw_residuals[rows, category_cols]
        loss = F.smooth_l1_loss(pred_yaw_rads_extracted, gt_yaw_deltas, reduction='sum', beta=0.5)
        return loss / gt_classes.numel()

    def get_vp_deltas(self, gt_viewpoint, gt_viewpoint_rads):
        gt_viewpoint = gt_viewpoint.type(torch.int64)
        bin_dist = np.linspace(-math.pi, math.pi, self.vp_bins + 1)
        bin_res = (bin_dist[1] - bin_dist[0]) / 2.

        src_vp_res = torch.tensor(bin_dist - bin_res, dtype=torch.float32).to(self.device)
        target_vp = gt_viewpoint_rads
        src_vp_proposals = src_vp_res[gt_viewpoint]
        src_vp_proposals[target_vp > src_vp_res[self.vp_bins]] = src_vp_res[self.vp_bins]

        wvp = np.trunc(1 / bin_res)
        dvp = wvp * (target_vp - src_vp_proposals - bin_res)
        deltas = dvp
        return deltas


class CategoryClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_classes = auxi["gt_matched"]["gt_category"].type(torch.int64)
        pred_classes = auxi["pred_matched"]["class_logits"]
        loss = F.cross_entropy(pred_classes, gt_classes, reduction="sum")
        return loss


class YawClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw = auxi["gt_matched"]["yaw"].type(torch.int64)
        gt_classes = auxi["gt_matched"]["gt_category"]
        pred_yaw = auxi["pred_matched"]["yaw_logits"]

        category_cols = (self.vp_bins * torch.unsqueeze(gt_classes, 1) + torch.arange(self.vp_bins).to(self.device)).type(torch.int64)
        rows = [[i] * category_cols.size(1) for i in range(category_cols.size(0))]
        pred_yaw_extracted = pred_yaw[rows, category_cols]
        loss = F.cross_entropy(pred_yaw_extracted, gt_yaw, reduction="sum")
        return loss / gt_classes.numel()


class ObjectClassification(LossBase):
    def __call__(self, features, pred, auxi):
        anchors = pred['anchors']
        pred_objectness_logits = pred['pred_objectness_logits']
        bbox2d = features['bbox2d']
        gt_labels, gt_boxes = lu.distribute_box_over_feature_map(anchors, bbox2d, self.anchor_matcher)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        valid_mask = gt_labels >= 0
        loss = F.binary_cross_entropy_with_logits(
            torch.cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        return loss
