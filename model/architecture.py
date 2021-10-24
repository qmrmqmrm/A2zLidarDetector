import glob
import os.path

import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
import cv2

import config as cfg
import model.submodules.model_util as mu
from utils.image_list import ImageList
import utils.util_function as uf


class ModelBase(nn.Module):
    def __init__(self, backbone, neck, rpn, head):
        super(ModelBase, self).__init__()
        self.device = torch.device(cfg.Hardware.DEVICE)
        self.backbone = backbone
        self.neck = neck
        self.proposal_generator = rpn
        self.roi_heads = head
        self.to(cfg.Hardware.DEVICE)

        pass

    def forward(self, batched_input):
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

    def __init__(self, backbone, neck, rpn, head):
        super().__init__(backbone, neck, rpn, head)
        assert len(cfg.Model.Structure.PIXEL_MEAN) == len(cfg.Model.Structure.PIXEL_STD)
        num_channels = len(cfg.Model.Structure.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.Model.Structure.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.Model.Structure.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.use_gt = True
    
    def set_gt_use(self, use):
        self.use_gt = use

    def forward(self, batched_input):
        """
        :param batched_input:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4](tlbr), 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }

        :return: model_output :
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'head_output' : torch.Size([4, 512, 93])
            }
        """
        image = self.preprocess_input(batched_input)
        backbone_features = self.backbone(image)
        # backbone_features /backbone_s2 torch.Size([2, 256, 160, 160])
        # backbone_features /backbone_s3 torch.Size([2, 512, 80, 80])
        # backbone_features /backbone_s4 torch.Size([2, 1024, 40, 40])
        neck_features = self.neck(backbone_features)
        # neck_features /neck_s2 torch.Size([2, 256, 160, 160])
        # neck_features /neck_s3 torch.Size([2, 256, 80, 80])
        # neck_features /neck_s4 torch.Size([2, 256, 40, 40])
        # neck_features /neck_s5 torch.Size([2, 256, 20, 20])
        print('usegt', self.use_gt)
        if self.use_gt:
            rpn_proposals, rpn_aux = self.proposal_generator(neck_features, batched_input['anc_feat'], batched_input)
        else:
            rpn_proposals, rpn_aux = self.proposal_generator(neck_features, batched_input['anc_feat'])
        # rpn_proposals /bbox2d torch.Size([2, 512, 4])
        # rpn_proposals /objectness torch.Size([2, 512, 1])
        # rpn_proposals /anchor_id torch.Size([2, 512, 1])

        head_output = self.roi_heads(neck_features, rpn_proposals)
        model_output = {key: value for key, value in rpn_proposals.items()}
        model_output.update({'rpn_feat_' + key: value for key, value in rpn_aux.items()})
        model_output['head_output'] = head_output
        # head_output  torch.Size([1024, 93]
        uf.print_structure('model_output', model_output)
        return model_output

    def preprocess_input(self, batched_input):  ## image
        image = batched_input['image'].permute(0, 3, 1, 2).to(self.device)
        image = self.normalizer(image)
        return image


class YOLO(ModelBase):
    pass
