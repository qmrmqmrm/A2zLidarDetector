import torch
import torch.nn.functional as F
import numpy as np
import config as cfg
import model.submodules.model_util as mu

np.set_printoptions(precision=6, suppress=True, linewidth=150)


class LossBase:
    def __init__(self):
        self.device = cfg.Hardware.DEVICE

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
            loss_per_scale = self.cal_bbox2d_loss_per_scale(pred, auxi, scale_idx)
            total_loss += loss_per_scale
        # rpn_bbox2d_logit = torch.cat(pred['rpn_feat_bbox2d'], dim=1)
        # gt_bbox2d = torch.cat(auxi['gt_feature']['bbox2d'], dim=1)
        # gt_object = torch.cat(auxi['gt_feature']['object'], dim=1)
        # loss = F.smooth_l1_loss(rpn_bbox2d_logit * gt_object, gt_bbox2d * gt_object, reduction='sum', beta=0.5)
        return total_loss

    def cal_bbox2d_loss_per_scale(self, pred, auxi, scale_idx):
        gt_object_per_scale = auxi['gt_feature']['object'][scale_idx]
        gt_bbox2d_per_scale = auxi['gt_feature']['bbox2d'][scale_idx] * gt_object_per_scale
        rpn_bbox2d_per_scale = pred['rpn_feat_bbox2d'][scale_idx] * gt_object_per_scale
        gt_object_mask = torch.where(gt_object_per_scale[..., 0] > 0)
        mask_sum = len(gt_object_mask[0])
        # print('mask_sum',mask_sum)
        if mask_sum > 0:
            gt_bbox2d_mask = gt_bbox2d_per_scale[gt_object_mask]
            rpn_bbox2d_mask = rpn_bbox2d_per_scale[gt_object_mask]
            print(gt_bbox2d_mask.shape)
            gt_mask_yxwh = mu.convert_box_format_tlbr_to_yxhw(gt_bbox2d_mask)
            rpn_mask_yxwh = mu.convert_box_format_tlbr_to_yxhw(rpn_bbox2d_mask)
            print('gt_mask_yxwh', gt_mask_yxwh[0:-1:100])
            print('rpn_mask_yxwh', rpn_mask_yxwh[0:-1:100])
            print('abs',scale_idx, torch.abs(gt_mask_yxwh[0:-1:100] - rpn_mask_yxwh[0:-1:100]))

        return F.smooth_l1_loss(rpn_bbox2d_per_scale, gt_bbox2d_per_scale, reduction='sum', beta=0.5)


class ObjectClassification(LossBase):
    def __call__(self, features, pred, auxi):
        total_loss = 0
        for scale_idx in range(3):
            loss_per_scale = self.cal_obj_loss_per_scale(pred, auxi, scale_idx)
            total_loss += loss_per_scale
        # rpn_object = torch.cat(pred['rpn_feat_objectness'], dim=1)  # (b, hwa,1) * 3
        # gt_object = torch.cat(auxi['gt_feature']['object'], dim=1)
        # loss = F.binary_cross_entropy_with_logits(rpn_object, gt_object, reduction="sum")
        # print('object loss',loss)
        return total_loss

    def cal_obj_loss_per_scale(self, pred, auxi, scale_idx):
        gt_object = auxi['gt_feature']['object'][scale_idx]
        gt_negative = auxi['gt_feature']['negative'][scale_idx]
        rpn_object = pred['rpn_feat_objectness'][scale_idx]
        ce_loss = F.binary_cross_entropy_with_logits(rpn_object, gt_object)
        positive_ce = torch.sum(ce_loss * gt_object) / (torch.sum(gt_object) + 0.00001)
        negative_ce = torch.sum(ce_loss * gt_negative) / (torch.sum(gt_negative) + 0.00001)
        scale_loss = positive_ce + negative_ce
        return scale_loss


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
        gt_classes = (auxi["gt_aligned"]["category"] * auxi["gt_aligned"]["object"]).type(torch.int64).view(-1)# (batch*512) torch.Size([4, 512, 1])
        pred_classes = (auxi["pred_select"]["category"] * auxi["gt_aligned"]["object"]).view(-1, 3)  # (batch*512 , 3) torch.Size([4, 512, 3])
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
