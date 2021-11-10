import torch
import torch.nn.functional as F
import numpy as np
import math

import config as cfg
import model.submodules.model_util as mu
import utils.util_function as uf

np.set_printoptions(precision=6, suppress=True, linewidth=150)


class LossBase:
    def __init__(self):
        self.device = cfg.Hardware.DEVICE
        self.bin_num = cfg.Model.Structure.VP_BINS

    def __call__(self, features, pred, auxi):
        """
        :param features:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4(tlbr)], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
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
        :param auxi:
        {
        'gt_aligned' : {
                        'bbox3d' : torch.Size([batch, 512, 6])
                        'category' : torch.Size([batch, 512, 1])
                        'bbox2d' : torch.Size([batch, 512, 4])
                        'yaw' : torch.Size([batch, 512, 1])
                        'yaw_rads' : torch.Size([batch, 512, 1])
                        'anchor_id' : torch.Size([batch, 512, 1])
                        'object' : torch.Size([batch, 512, 1])
                        'negative' : torch.Size([batch, 512, 1])
                        }
        'gt_feature' : {
                        'bbox3d' : list(torch.Size([batch, height/stride* width/stride* anchor, 6]))
                        'category' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr]))
                        'yaw' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'yaw_rads' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'object' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'negative' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        }
        'pred_select' : {
                        'bbox3d' : torch.Size([batch, 512, 6])
                        'category' : torch.Size([batch, 512, 1])
                        'yaw' : torch.Size([batch, 512, 1])
                        'yaw_rads' : torch.Size([batch,512, 1])
                        }
        }
        :return:
        """
        raise NotImplementedError()


class Box2dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        total_loss = 0
        for scale_idx in range(3):
            loss_per_scale = self.cal_delta2d_loss_per_scale(pred, auxi, scale_idx)
            total_loss += loss_per_scale
        return total_loss

    def cal_delta2d_loss_per_scale(self, pred, auxi, scale_idx):
        gt_object_per_scale = auxi['gt_feature']['object'][scale_idx]
        gt_bbox2d_per_scale = auxi['gt_feature']['bbox2d_delta'][scale_idx] * gt_object_per_scale
        rpn_bbox2d_per_scale = pred['rpn_feat_bbox2d_delta'][scale_idx] * gt_object_per_scale
        return F.smooth_l1_loss(rpn_bbox2d_per_scale, gt_bbox2d_per_scale, reduction='sum', beta=0.5)


class ObjectClassification(LossBase):
    def __call__(self, features, pred, auxi):
        total_loss = 0
        for scale_idx in range(3):
            loss_per_scale = self.cal_obj_loss_per_scale(pred, auxi, scale_idx)
            total_loss += loss_per_scale
        return total_loss

    def cal_obj_loss_per_scale(self, pred, auxi, scale_idx):
        gt_object = auxi['gt_feature']['object'][scale_idx]
        gt_negative = auxi['gt_feature']['negative'][scale_idx]
        rpn_obj_logit = pred['rpn_feat_object_logits'][scale_idx]
        rpn_obj_sig = pred['rpn_feat_objectness'][scale_idx]
        focal_loss = torch.pow(rpn_obj_sig - gt_object, 2)

        ce_loss = F.binary_cross_entropy_with_logits(rpn_obj_logit, gt_object, reduction='none') * focal_loss
        ps_ce = ce_loss * gt_object
        positive_ce = torch.sum(ps_ce) / (torch.sum(gt_object) + 0.00001)
        negative_ce = torch.sum(ce_loss * gt_negative) / (torch.sum(gt_negative) + 0.00001) * 10
        scale_loss = positive_ce + negative_ce
        return scale_loss


class Box3dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_bbox3d = auxi['gt_aligned']['bbox3d_delta'] * auxi["gt_aligned"]["object"]
        pred_bbox3d = auxi['pred_select']['bbox3d_delta'] * auxi["gt_aligned"]["object"]
        loss = F.smooth_l1_loss(pred_bbox3d, gt_bbox3d, reduction='sum', beta=0.5)
        return loss  # / (num_gt + 0.00001)


class YawRegression(LossBase):
    def __init__(self):
        super().__init__()

        bin_edge = np.linspace(-math.pi / 2, math.pi / 2, self.bin_num + 1)
        self.bin_res = (bin_edge[1] - bin_edge[0]) / 2.
        self.bin_edge = torch.tensor(bin_edge - self.bin_res, dtype=torch.float32).to(self.device)

    def __call__(self, features, pred, auxi):
        gt_yaw_res = self.get_deltas_yaw(auxi['gt_aligned']['yaw_cls'], auxi['gt_aligned']['yaw_rads'])
        # yaw residual range: -15deg ~ 15deg = -0.26rad ~ 0.26rad
        # pred_yaw_residuals = torch.sigmoid(auxi["pred_select"]["yaw_res"]) * 0.6 - 0.3
        ce_loss = F.smooth_l1_loss(auxi["pred_select"]["yaw_res"], gt_yaw_res, reduction='none', beta=0.5)
        loss = ce_loss * auxi["gt_aligned"]["object"]
        loss = torch.sum(loss)
        return loss  # / (num_gt + 0.00001)

    def get_deltas_yaw(self, yaw_cls, yaw_rad):
        yaw_cls = yaw_cls.to(dtype=torch.int64)
        bin_begin = self.bin_edge[yaw_cls]
        yaw_rad = torch.where(yaw_rad < self.bin_edge[0], yaw_rad + math.pi, yaw_rad)
        yaw_rad = torch.where(yaw_rad > self.bin_edge[-1], yaw_rad - math.pi, yaw_rad)
        delta = yaw_rad - bin_begin - self.bin_res
        return delta


class CategoryClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_classes = (auxi["gt_aligned"]["category"]).type(torch.int64).view(-1)  # (batch*512)
        pred_classes = (auxi["pred_select"]["ctgr_logit"]).view(-1, 4)  # (batch*512 , 3)
        ce_loss = F.cross_entropy(pred_classes, gt_classes, reduction="none")
        ce_loss = ce_loss * pred['zeropad']

        bgd_ce = ce_loss * (gt_classes == 0) * 0.001
        tr_ce = ce_loss * (gt_classes > 0)
        loss = torch.sum(bgd_ce) + torch.sum(tr_ce)

        return loss  # / (num_gt + 0.00001)


class YawClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw = (auxi['gt_aligned']['yaw_cls']).view(-1).to(torch.int64)
        pred_yaw = (auxi['pred_select']['yaw_cls_logit']).view(-1, self.bin_num)
        ce_loss = F.cross_entropy(pred_yaw, gt_yaw, reduction="none")
        ce_loss = ce_loss * pred['zeropad']
        pos_ce = ce_loss * auxi["gt_aligned"]["object"].view(-1)
        loss = torch.sum(pos_ce)

        return loss  # / (num_gt + 0.00001)
