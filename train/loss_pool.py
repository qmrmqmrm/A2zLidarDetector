import torch
import torch.nn.functional as F
import numpy as np
import math

from config import Config as cfg
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.matcher import Matcher
import train.loss_util as lu
import utils.util_function as uf

DEVICE = cfg.Model.Structure.DEVICE


class LossBase:
    def __init__(self):
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.anchor_iou_thresh = cfg.Model.RPN.IOU_THRESHOLDS
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

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
        pred_anchor_deltas_cat = torch.cat(pred['pred_anchor_deltas'], dim=1)
        anchors_cat = torch.cat(pred['anchors'])
        gt_boxes_batch = features["bbox2d"]
        loss = 0
        for gt_boxes_i, pred_anchor_deltas_i in zip(gt_boxes_batch, pred_anchor_deltas_cat):
            w = gt_boxes_i[:, 2] - gt_boxes_i[:, 0]
            gt_boxes_i = gt_boxes_i[w > 0, :]
            anchor_inds = lu.match_gt_with_anchors(gt_boxes_i, anchors_cat)
            pred_anchor_deltas = pred_anchor_deltas_i[anchor_inds]
            gt_anchor_deltas = self.box2box_transform.get_deltas(anchors_cat[anchor_inds], gt_boxes_i)
            loss += F.smooth_l1_loss(pred_anchor_deltas, gt_anchor_deltas, reduction='sum', beta=0.5)
        print("box2d loss", loss)
        return loss


class Box3dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        head_class_logits = pred['head_class_logits']
        bbox_3d_logits = pred['bbox_3d_logits']
        head_proposals = pred['head_proposals']
        gt_boxes3d = torch.cat([p['bbox3d'] for p in head_proposals])
        gt_bbox2d = torch.cat([p['bbox2d'] for p in head_proposals])
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        proposals = torch.cat([p['proposal_boxes'] for p in head_proposals])

        gt_proposal_deltas = self.box2box_transform.get_deltas(
            proposals, gt_boxes3d[:, :4], True
        )
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = bbox_3d_logits.size(1) == box_dim

        bg_class_ind = head_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)


        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=DEVICE)
        else:
            fg_gt_classes = gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None].to(DEVICE) + torch.arange(box_dim, device=DEVICE)
        # print(bbox_3d_logits[fg_inds[:, None].long(), gt_class_cols.long()])
        loss_box_reg = F.smooth_l1_loss(bbox_3d_logits[fg_inds[:, None].long(), gt_class_cols.long()],
                                        gt_proposal_deltas[fg_inds],
                                        reduction='sum', beta=0.0)

        loss_box_reg = loss_box_reg / gt_classes.numel()
        return loss_box_reg


class HeightRegression(LossBase):
    def __call__(self, features, pred, auxi):
        head_class_logits = pred['head_class_logits']
        head_height_logits = pred['head_height_logits']
        head_proposals = pred['head_proposals']
        gt_boxes3d = torch.cat([p['bbox3d'] for p in head_proposals])
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)

        gt_height = gt_boxes3d[:, -2:]
        gt_height_deltas = self.get_h_deltas(gt_height, gt_classes)
        box_dim = gt_height_deltas.size(1)
        bg_class_ind = head_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)

        fg_gt_classes = gt_classes[fg_inds].to(cfg.Model.Structure.DEVICE)
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=DEVICE)
        loss_box_reg = F.smooth_l1_loss(head_height_logits[fg_inds[:, None].long(), gt_class_cols.long()],
                                        gt_height_deltas[fg_inds],
                                        reduction='sum', beta=0.5)
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


class YawRegression(LossBase):
    def __call__(self, features, pred, auxi):
        head_class_logits = pred['head_class_logits']
        head_proposals = pred['head_proposals']

        head_yaw_residuals = pred['head_yaw_residuals']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0).to(DEVICE)
        gt_viewpoint = torch.cat([p['yaw'] for p in head_proposals], dim=0).type(torch.int64).to(DEVICE)
        gt_viewpoint_rads = torch.cat([p['yaw_rads'] for p in head_proposals], dim=0).to(DEVICE)
        gt_vp_deltas = self.get_vp_deltas(gt_viewpoint, gt_viewpoint_rads)
        bg_class_ind = head_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)
        fg_gt_classes = gt_classes[fg_inds]
        res_index_list = list()

        for idx, logit in enumerate(head_yaw_residuals[fg_inds]):
            res_index_list.append(fg_gt_classes[idx] * self.vp_bins + gt_viewpoint[fg_inds][idx])

        loss = F.smooth_l1_loss(head_yaw_residuals[fg_inds, res_index_list],
                                gt_vp_deltas[fg_inds],
                                reduction='sum', beta=0.5)
        loss = loss / gt_classes.numel()
        return loss

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


class CategoryClassification(LossBase):
    def __call__(self, features, pred, auxi):
        head_class_logits = pred['head_class_logits'].type(torch.float32)
        head_proposals = pred['head_proposals']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0).to(DEVICE).type(torch.int64)
        loss = F.cross_entropy(head_class_logits, gt_classes, reduction="mean")
        return loss


class YawClassification(LossBase):
    def __call__(self, features, pred, auxi):
        head_class_logits = pred['head_class_logits']
        head_proposals = pred['head_proposals']
        head_yaw_logits = pred['head_yaw_logits']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        gt_viewpoint = torch.cat([p['yaw'] for p in head_proposals], dim=0).to(DEVICE)
        bg_class_ind = head_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < bg_class_ind)).squeeze(1)
        fg_gt_classes = gt_classes[fg_inds]
        vp_list = list()
        for idx, logit in enumerate(head_yaw_logits[fg_inds]):
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
    def __call__(self, features, pred, auxi):
        gt_boxes_batch = features["bbox2d"]
        gt_object_batch = torch.squeeze(features["object"])
        anchors_cat = torch.cat(pred['anchors'])
        pred_objectness_logits = torch.cat(pred['pred_objectness_logits'], dim=1)

        loss = 0
        for gt_boxes_i, gt_object_i, pred_objectness_deltas_i in zip(gt_boxes_batch, gt_object_batch, pred_objectness_logits):
            x2 = gt_boxes_i[:, 2]
            gt_boxes_i = gt_boxes_i[x2 > 0, :]
            gt_object_i = gt_object_i[x2 > 0]
            match_quality_matrix = uf.pairwise_iou(gt_boxes_i, anchors_cat)  # [num_gt, sum HWA]
            max_vals, max_inds = match_quality_matrix.max(dim=0)        # max_vals [sum HWA]
            gt_object_map = gt_object_i[max_inds]
            valid_mask = torch.logical_or(max_vals < self.anchor_iou_thresh[0], max_vals > self.anchor_iou_thresh[1])

            print('pred boxes i:', pred_objectness_deltas_i[valid_mask][0:-1:10000])
            print('gt_boxes_i:', gt_object_map[valid_mask][0:-1:10000])

            loss += F.binary_cross_entropy_with_logits(pred_objectness_deltas_i[valid_mask], gt_object_map[valid_mask], reduction="sum")
        print("objectness loss", loss)
        return loss
