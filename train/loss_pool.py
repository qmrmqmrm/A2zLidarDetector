"""
def loss_obj(feature, prediction):
    prediction = {"box2d_map": [batch, channel, height, width],    (channel: y1,x1,y2,x2)
                "object_map": [batch, channel, height, width],    (channel: obj)
                "box3d": [batch, numbox, channel],  (channel: X,Y,Z,length,width,height) 단위??
                "yaw_class": [batch, numbox, channel],  (channel: yaw probabilities)
                "yaw_resid": [batch, numbox, channel],
                "class": [batch, numbox, channel]       (channel: class probabilities)
                }

    loss["box2d"] = smooth_l1(feature["box2d_map"], prediction["box2d_map"], feature["object_map"])
    loss["obj2d"] = binary_crossentropy(feature["obj2d_map"], prediction["object_map"])
    loss["class"] = crossentropy(feature["class"], prediction["class"], feature["object"])
    loss["yaw_class"] = crossentropy(feature["yaw"], prediction["yaw_class"], feature["object"])
    loss["yaw_resid"] = smooth_l1(feature["yaw"], prediction["yaw_resid"], feature["object"])
    loss["box3d"] = smooth_l1(feature["box3d"], prediction["box3d"], feature["object"])


def smooth_l1(feature, prediction, objectness)
    x = feature - prediction
    small_mask = torch.abs(x) < 1
    loss_map = 0.5 * x^2 * objectness * small_mask
                + (torch.abs(x) - 0.5) * objectness * (1 - small_mask)
    loss_scalar = torch.sum(loss_map)
    return loss_scalar
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from config import Config as cfg
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.matcher import Matcher
from model.submodules.sampling import subsample_labels
from utils.util_function import pairwise_iou

DEVICE = cfg.Model.Structure.DEVICE


class LossBase:
    def __init__(self):
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.Model.RPN.IOU_THRESHOLDS, cfg.Model.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.batch_size_per_image = cfg.Model.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.Model.RPN.POSITIVE_FRACTION
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

        self.weights_height = cfg.Model.Structure.WEIGHTS_HEIGHT

    def __call__(self, features, pred):
        """

        :param features:
            {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 2]}
        :param pred:
                {'pred_class_logits': torch.Size([1024, 4])
                'pred_proposal_deltas': torch.Size([1024, 12])
                'viewpoint_logits': torch.Size([1024, 36])
                'viewpoint_residuals': torch.Size([1024, 36])
                'height_logits': torch.Size([1024, 6])

                'head_proposals': [{'proposal_boxes': torch.Size([512, 4])
                                    'objectness_logits': torch.Size([512])
                                    'category': torch.Size([512, 1])
                                    'bbox2d': torch.Size([512, 4])
                                    'bbox3d': torch.Size([512, 6])
                                    'object': torch.Size([512, 1])
                                    'yaw': torch.Size([512, 2])} * batch]
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
                }
        :return:
        """

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def label_and_sample_anchors(self, anchors, feature):
        """

        :param anchors:
                    [torch.Size([557568(176 * 352 * 9), 4])
                    torch.Size([139392(88 * 176 * 9), 4])
                    torch.Size([34848(44 * 88 * 9), 4])]
        :param feature:
            {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 2]}
        :return:
        """

        anchors = torch.cat(anchors)
        image_shape = feature['image'].shape
        image_sizes = [(image_shape[2], image_shape[3]) for i in feature['image'].shape]
        bbox2d_shape = feature['bbox2d'].shape
        bbox2d_batch = list()
        for i in range(bbox2d_shape[0]):
            bbox2d = feature['bbox2d'][i, :]
            weight = bbox2d[:, 2] - bbox2d[:, 0]
            x = torch.where(weight > 0)
            bbox2d = bbox2d[:x[0][-1] + 1, :]
            bbox2d_batch.append(bbox2d)

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, bbox2d_batch):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = self.anchor_matcher(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=DEVICE)
            del match_quality_matrix

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            gt_labels_i = gt_labels_i.to(DEVICE)
            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs]
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes


class SmoothL1(LossBase):
    def __call__(self, features, pred):
        raise NotImplementedError()

    def smooth_l1(self, pred, grtr, beta, reduction):
        if beta < 1e-5:
            # if beta == 0, then torch.where will result in nan gradients when
            # the chain rule is applied due to pytorch implementation details
            # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
            # zeros, rather than "no gradient"). To avoid this issue, we define
            # small values of beta to be exactly l1 loss.
            loss = torch.abs(pred - grtr)
        else:
            n = torch.abs(pred - grtr)
            cond = n < beta
            loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss


class Box2dRegression(SmoothL1):
    def __call__(self, features, pred):
        anchors = pred['anchors']
        pred_anchor_deltas = pred['pred_anchor_deltas']

        gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, features)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        anchors = torch.cat(anchors)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        pos_mask = gt_labels == 1
        loss = self.smooth_l1(torch.cat(pred_anchor_deltas, dim=1)[pos_mask],
                              gt_anchor_deltas[pos_mask],
                              beta=0.0,
                              reduction='sum')

        return loss


class Box3dRegression(SmoothL1):
    def __call__(self, features, pred):
        pred_class_logits = pred['pred_class_logits']
        pred_proposal_deltas = pred['pred_proposal_deltas']
        head_proposals = pred['head_proposals']
        gt_boxes3d = torch.cat([p['bbox3d'] for p in head_proposals])
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        proposals = torch.cat([p['proposal_boxes'] for p in head_proposals])

        gt_proposal_deltas = self.box2box_transform.get_deltas(
            proposals, gt_boxes3d[:, :4], self.rotated_box_training
        )
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim

        bg_class_ind = pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(
            1
        )

        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=DEVICE)
        else:
            fg_gt_classes = gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None].to(DEVICE) + torch.arange(box_dim, device=DEVICE)

        loss_box_reg = self.smooth_l1(
            pred_proposal_deltas[fg_inds[:, None].long(), gt_class_cols.long()],
            gt_proposal_deltas[fg_inds],
            beta=0.0,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / gt_classes.numel()
        return loss_box_reg
        #
        #


class HeightRegression(SmoothL1):
    def __call__(self, features, pred):
        pred_class_logits = pred['pred_class_logits']
        height_logits = pred['height_logits']
        head_proposals = pred['head_proposals']
        gt_boxes3d = torch.cat([p['bbox3d'] for p in head_proposals])
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)

        gt_height = gt_boxes3d[:, -2:]
        gt_height_deltas = self.get_h_deltas(gt_height, gt_classes)
        box_dim = gt_height_deltas.size(1)
        bg_class_ind = pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)

        fg_gt_classes = gt_classes[fg_inds].to(cfg.Model.Structure.DEVICE)
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=DEVICE)
        loss_box_reg = self.smooth_l1(
            height_logits[fg_inds[:, None].long(), gt_class_cols.long()],
            gt_height_deltas[fg_inds],
            0.0,
            reduction="sum",
        )
        # The loss is normalized as in box delta regression task
        loss_box_reg = loss_box_reg / gt_classes.numel()
        return loss_box_reg

    def get_h_deltas(self, gt_height, gt_classes):
        src_heights = torch.tensor([130.05, 149.6, 147.9, 1.0]).to(DEVICE)  # Mean heights encoded

        target_heights = gt_height[:, 0].to(DEVICE)
        target_ctr = gt_height[:, 1].to(DEVICE)

        wh, wg, wz = self.weights_height
        dh = wh * torch.log(target_heights / src_heights[gt_classes.long()])
        dz = wz * (target_ctr - src_heights[gt_classes.long()] / 2.) / src_heights[gt_classes.long()]

        deltas = torch.stack((dh, dz), dim=1, ).to(cfg.Model.Structure.DEVICE)
        return deltas


class YawRegression(SmoothL1):
    def __call__(self, features, pred):
        pred_class_logits = pred['pred_class_logits']
        head_proposals = pred['head_proposals']

        viewpoint_residuals = pred['viewpoint_residuals']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0).to(DEVICE)
        gt_viewpoint = torch.cat([p['yaw'] for p in head_proposals], dim=0).type(torch.int64).to(DEVICE)
        gt_viewpoint_rads = torch.cat([p['yaw_rads'] for p in head_proposals], dim=0).to(DEVICE)
        gt_vp_deltas = self.get_vp_deltas(gt_viewpoint, gt_viewpoint_rads)
        bg_class_ind = pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)
        fg_gt_classes = gt_classes[fg_inds]
        res_index_list = list()

        for idx, logit in enumerate(viewpoint_residuals[fg_inds]):
            res_index_list.append(fg_gt_classes[idx] * self.vp_bins + gt_viewpoint[fg_inds][idx])
        loss_box_reg = self.smooth_l1(
            viewpoint_residuals[fg_inds, res_index_list],
            gt_vp_deltas[fg_inds],
            0.0,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / gt_classes.numel()
        return loss_box_reg

    def get_vp_deltas(self, gt_viewpoint, gt_viewpoint_rads):
        gt_viewpoint = gt_viewpoint
        bin_dist = np.linspace(-math.pi, math.pi, self.vp_bins + 1)
        bin_res = (bin_dist[1] - bin_dist[0]) / 2.

        src_vp_res = torch.tensor(bin_dist - bin_res, dtype=torch.float32).to(DEVICE)
        target_vp = gt_viewpoint_rads
        src_vp_proposals = src_vp_res[gt_viewpoint]
        src_vp_proposals[target_vp > src_vp_res[self.vp_bins]] = src_vp_res[self.vp_bins]

        wvp = np.trunc(1 / bin_res)
        dvp = wvp * (target_vp - src_vp_proposals - bin_res)
        deltas = dvp
        return deltas


class CrossEntropy():
    def __call__(self, grtr, pred, beta, reduction):
        raise NotImplementedError()

    def smooth_l1(self, foo, bar):
        pass


class CategoryClassification(LossBase):
    def __call__(self, features, pred):
        pred_class_logits = pred['pred_class_logits']
        head_proposals = pred['head_proposals']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0).to(DEVICE).type(torch.int64)
        pred_class_logits = pred_class_logits.type(torch.float32)
        loss = F.cross_entropy(pred_class_logits, gt_classes, reduction="mean")
        return loss


class YawClassification(LossBase):
    def __call__(self, features, pred):
        pred_class_logits = pred['pred_class_logits']
        head_proposals = pred['head_proposals']
        viewpoint_logits = pred['viewpoint_logits']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        gt_viewpoint = torch.cat([p['yaw'] for p in head_proposals], dim=0).to(DEVICE)
        bg_class_ind = pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)
        fg_gt_classes = gt_classes[fg_inds]
        vp_list = list()
        for idx, logit in enumerate(viewpoint_logits[fg_inds]):
            gt_num = int(fg_gt_classes[idx]) * self.vp_bins
            vp_list.append(
                logit[gt_num:gt_num + self.vp_bins])  # Theoricatly the last class is background... SEE bg_class_ind
        filtered_viewpoint_logits = torch.cat(vp_list).view(gt_viewpoint[fg_inds].size()[0], self.vp_bins)
        # print(filtered_viewpoint_logits.shape)
        # print(gt_viewpoint[fg_inds].shape)
        loss = F.cross_entropy(filtered_viewpoint_logits, gt_viewpoint[fg_inds].type(torch.int64),
                               reduction="sum")
        loss = loss / gt_classes.numel()
        return loss


class ObjectClassification(LossBase):
    def __call__(self, features, pred):
        anchors = pred['anchors']
        pred_objectness_logits = pred['pred_objectness_logits']
        gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, features)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        valid_mask = gt_labels >= 0
        loss = F.binary_cross_entropy_with_logits(
            torch.cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        return loss
