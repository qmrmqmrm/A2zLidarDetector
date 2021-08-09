import torch
from torch import nn
import copy
import numpy as np
import cv2

from config import Config as cfg
from utils.image_list import ImageList


class ModelBase(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(ModelBase, self).__init__()
        self.device = torch.device("cuda")
        self.backbone = backbone
        self.proposal_generator = rpn
        self.roi_heads = head
        self.to(self.device)

        pass

    def forward(self, batched_inputs):
        # ...
        # return {"backbone_l": backbone_l, "backbone_m": backbone_m, "backbone_s": backbone_s,
        #         "boxreg": boxreg, "category": catetory, "validbox": valid_box}
        pass


class GeneralizedRCNN(ModelBase):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, backbone, rpn, head):
        super().__init__(backbone, rpn, head)
        assert len(cfg.Model.Structure.PIXEL_MEAN) == len(cfg.Model.Structure.PIXEL_STD)
        num_channels = len(cfg.Model.Structure.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.Model.Structure.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.Model.Structure.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

    def forward(self, batched_inputs):
        """

        :param batched_inputs: [list[Dict]] {"num_bbox2D": [512,4], "num_bbox3D": [512,6], "num_yaw": [512,2] ,
                                             "gt_bbox2D": [n,4], "bv_img": [700,1400]}
        :return:
        """
        images = self.preprocess_image(batched_inputs)

        gt_instances = list()
        for batched_input in batched_inputs:
            gt_instance = dict()
            for key, value in batched_input.items():
                if 'gt_' in key:
                    gt_instance.update({key: value})
            gt_instances.append(gt_instance)

        features = self.backbone(images.tensor)
        rpn_proposals, loss_instances = self.proposal_generator(images.image_sizes, features)
        pred, head_proposals = self.roi_heads(features, rpn_proposals, gt_instances)
        pred['rpn_proposals'] = rpn_proposals
        pred.update(loss_instances)
        pred['head_proposals'] = head_proposals
        # pred keys :  dict_keys(['scores', 'proposal_deltas', 'viewpoint_scores', 'viewpoint_residuals', 'height_scores', 'proposals', 'loss_instances'])
        # print("pred_anchor_deltas",loss_instances['pred_anchor_deltas'])
        return pred

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [x.permute(2, 0, 1) for x in images]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class YOLO(ModelBase):
    pass
