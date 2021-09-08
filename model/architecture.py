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
    def __init__(self, backbone,neck, rpn, head):
        super(ModelBase, self).__init__()
        self.device = torch.device(cfg.Model.Structure.DEVICE)
        self.backbone = backbone
        self.neck = neck
        self.proposal_generator = rpn
        self.roi_heads = head
        self.to(cfg.Model.Structure.DEVICE)

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

    def __init__(self, backbone,neck, rpn, head):
        super().__init__(backbone,neck, rpn, head)
        assert len(cfg.Model.Structure.PIXEL_MEAN) == len(cfg.Model.Structure.PIXEL_STD)
        num_channels = len(cfg.Model.Structure.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.Model.Structure.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.Model.Structure.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

    def forward(self, batched_input):
        """
        :param batched_input:
            {'image': [batch, height, width, channel],
            'category': [batch, fixbox, 1],
            'bbox2d': [[numbox, 4]*batch], 'bbox3d': [batch, num, 6], 'object': [batch, num, 1],
            'yaw': [batch, num, 1], 'yaw_rads': [batch, num, 1]}

        :return:
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

                    rpn
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
        """
        image = self.preprocess_input(batched_input)
        backbone_features = self.backbone(image)
        neck_features = self.neck(backbone_features)
        # uf.print_structure('batched_input', batched_input)
        #
        rpn_proposals, auxiliary = self.proposal_generator(image.shape, neck_features, batched_input)
        # pred = self.roi_heads(batched_input, features, rpn_proposals)
        # pred['rpn_proposals'] = rpn_proposals
        # # pred['batched_input'] = batched_input
        # pred.update(auxiliary)
        return rpn_proposals, auxiliary

    def preprocess_input(self, batched_input):  ## image
        image = batched_input['image'].permute(0, 3, 1, 2).to(self.device)
        image = self.normalizer(image)
        # image = F.pad(image,(4,4,2,2), "constant", 0)
        # batched_input['image'] = image
        # batched_input['bbox2d'] += torch.tensor([[[4, 2, 4, 2]]], dtype=torch.float32).to(self.device)
        # batched_input['bbox3d'][:, :, :2] += torch.tensor([[[4, 2]]], dtype=torch.float32).to(self.device)
        # batched_input = mu.remove_padding(batched_input)
        return image


class YOLO(ModelBase):
    pass
