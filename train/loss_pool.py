import torch
import torch.nn.functional as F
import numpy as np
import math

import config as cfg
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.matcher import Matcher
import train.loss_util as lu
import utils.util_function as uf


class LossBase:
    def __init__(self):
        self.device = cfg.Hardware.DEVICE

    def __call__(self, features, pred, auxi):
        raise NotImplementedError()


class Box2dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        rpn_bbox2d_logit = torch.cat(pred['rpn_feat_bbox2d'], dim=1)
        gt_bbox2d = torch.cat(auxi['gt_feature']['bbox2d'], dim=1)
        gt_object = torch.cat(auxi['gt_feature']['object'], dim=1)

        loss = F.smooth_l1_loss(rpn_bbox2d_logit * gt_object, gt_bbox2d * gt_object, reduction='sum', beta=0.5)
        return loss


class ObjectClassification(LossBase):
    def __call__(self, features, pred, auxi):
        rpn_object_logit = torch.cat(pred['rpn_feat_objectness'], dim=1)
        gt_object = torch.cat(auxi['gt_feature']['object'], dim=1)
        loss = F.binary_cross_entropy_with_logits(rpn_object_logit, gt_object, reduction="sum")
        return loss


class Box3dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_bbox3d = auxi['gt_aligned']['bbox3d'] * auxi["gt_aligned"]["object"]
        pred_bbox3d = auxi['pred_select']['bbox3d'] * auxi["gt_aligned"]["object"]
        num_gt = torch.sum(auxi["gt_aligned"]["object"])
        loss = F.smooth_l1_loss(pred_bbox3d, gt_bbox3d, reduction='sum', beta=0.0)
        return loss / num_gt


class YawRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw_rads = auxi["gt_aligned"]["yaw_rads"] * auxi["gt_aligned"]["object"]
        pred_yaw_residuals = auxi["pred_select"]["yaw_rads"]
        num_gt = torch.sum(auxi["gt_aligned"]["object"])
        loss = F.smooth_l1_loss(pred_yaw_residuals, gt_yaw_rads, reduction='sum', beta=0.5)
        return loss / num_gt


class CategoryClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_classes = (auxi["gt_aligned"]["category"] * auxi["gt_aligned"]["object"]).type(torch.int64).view(-1)# (batch*512)
        pred_classes = (auxi["pred_select"]["category"] * auxi["gt_aligned"]["object"]).view(-1, 3)  # (batch*512 , 3)
        num_gt = torch.sum(auxi["gt_aligned"]["object"])
        loss = F.cross_entropy(pred_classes, gt_classes, reduction="sum")
        return loss / num_gt


class YawClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw = (auxi['gt_aligned']['yaw'] * auxi["gt_aligned"]["object"]).view(-1).to(torch.int64)
        pred_yaw = (auxi['pred_select']['yaw'] * auxi["gt_aligned"]["object"]).view(-1, 12)
        num_gt = torch.sum(auxi["gt_aligned"]["object"])
        # pred(N,C), gt(N)
        loss = F.cross_entropy(pred_yaw, gt_yaw, reduction="sum")
        return loss / num_gt
