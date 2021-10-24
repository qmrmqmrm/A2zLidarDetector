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
        return total_loss

    def cal_bbox2d_loss_per_scale(self, pred, auxi, scale_idx):
        gt_object_per_scale = auxi['gt_feature']['object'][scale_idx]
        gt_bbox2d_per_scale = auxi['gt_feature']['bbox2d'][scale_idx] * gt_object_per_scale
        rpn_bbox2d_per_scale = pred['rpn_feat_bbox2d'][scale_idx] * gt_object_per_scale
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
        rpn_object = pred['rpn_feat_objectness'][scale_idx]
        focal_loss = torch.pow(rpn_object - gt_object, 2)
        ce_loss = F.binary_cross_entropy_with_logits(rpn_object, gt_object, reduction='none') * focal_loss
        ps_ce = ce_loss * gt_object
        positive_ce = torch.sum(ps_ce) / (torch.sum(gt_object) + 0.00001)
        negative_ce = torch.sum(ce_loss * gt_negative) / (torch.sum(gt_negative) + 0.00001) * 8
        scale_loss = positive_ce + negative_ce
        return scale_loss


class Box3dRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_bbox3d = auxi['gt_aligned']['bbox3d'] * auxi["gt_aligned"]["object"]
        pred_bbox3d = auxi['pred_select']['bbox3d'] * auxi["gt_aligned"]["object"]
        loss = F.smooth_l1_loss(pred_bbox3d, gt_bbox3d, reduction='sum', beta=0.5)
        return loss  # / (num_gt + 0.00001)


class YawRegression(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw_rads = auxi["gt_aligned"]["yaw_rads"] * auxi["gt_aligned"]["object"]
        pred_yaw_residuals = auxi["pred_select"]["yaw_rads"] * auxi["gt_aligned"]["object"]
        loss = F.smooth_l1_loss(pred_yaw_residuals, gt_yaw_rads, reduction='sum', beta=0.5)
        return loss  # / (num_gt + 0.00001)


class CategoryClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_classes = (auxi["gt_aligned"]["category"]).type(torch.int64).view(-1)  # (batch*512) torch.Size([4, 512, 1])
        pred_classes = (auxi["pred_select"]["category"]).view(-1, 4)  # (batch*512 , 3) torch.Size([4, 512, 3])


        pred_classes_numpy = pred_classes.to('cpu').detach().numpy()
        pred_quant = np.quantile(pred_classes_numpy, np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.995, 0.999, 1]))
        print("pred_classes quantile:", pred_quant)
        for i in range(1, pred_classes_numpy.shape[-1]):
            pred_quant = np.quantile(pred_classes_numpy[:, i] - pred_classes_numpy[:, 0],
                                       np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.995, 0.999, 1]))
            print(f"pred_classes quantile {i} :", pred_quant)
        ce_loss = F.cross_entropy(pred_classes, gt_classes, reduction="none")
        bgd_ce = ce_loss * auxi["gt_aligned"]["object"] * (gt_classes == 0) * 0.001
        tr_ce = ce_loss * auxi["gt_aligned"]["object"] * (gt_classes > 0)
        loss = torch.sum(bgd_ce) + torch.sum(tr_ce)
        return loss  # / (num_gt + 0.00001)


class YawClassification(LossBase):
    def __call__(self, features, pred, auxi):
        gt_yaw = (auxi['gt_aligned']['yaw']).view(-1).to(torch.int64)
        pred_yaw = (auxi['pred_select']['yaw']).view(-1, 12)
        # pred(N,C), gt(N)
        ce_loss = F.cross_entropy(pred_yaw, gt_yaw, reduction="none")
        pos_ce = ce_loss * auxi["gt_aligned"]["object"]
        loss = torch.sum(pos_ce)
        return loss  # / (num_gt + 0.00001)
