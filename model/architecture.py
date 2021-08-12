import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
import cv2

from config import Config as cfg
from utils.image_list import ImageList


class ModelBase(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(ModelBase, self).__init__()
        self.device = torch.device(cfg.Model.Structure.DEVICE)
        self.backbone = backbone
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

    def __init__(self, backbone, rpn, head):
        super().__init__(backbone, rpn, head)
        assert len(cfg.Model.Structure.PIXEL_MEAN) == len(cfg.Model.Structure.PIXEL_STD)
        num_channels = len(cfg.Model.Structure.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.Model.Structure.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.Model.Structure.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

    def forward(self, batched_input):
        """
        :param batched_input:
            {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 2]}
        :return:
            pred :{'pred_class_logits': torch.Size([1024, 4])
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
        """
        batched_input = self.preprocess_input(batched_input)
        features = self.backbone(batched_input['image'])
        # print('\nGeneralizedRCNN category.shape :', batched_input['category'].shape)
        # print('GeneralizedRCNN category :', batched_input['category'])
        # print('GeneralizedRCNN bbox2d.shape :', batched_input['bbox2d'].shape)
        # print('GeneralizedRCNN bbox2d :', batched_input['bbox2d'])
        # print('GeneralizedRCNN yaw shape :', batched_input['yaw'].shape)
        # print('GeneralizedRCNN yaw :', batched_input['yaw'])
        # print('GeneralizedRCNN yaw_rads shape :', batched_input['yaw_rads'].shape)
        # print('GeneralizedRCNN yaw_rads :', batched_input['yaw_rads'])
        # features {'p2': torch.Size([batch, 256, 176, 352]),'p3': torch.Size([batch, 256, 88, 176]),
        # 'p4': torch.Size([batch, 256, 44, 88]), 'p5': torch.Size([batch, 256, 22, 44])}
        rpn_proposals, auxiliary = self.proposal_generator(batched_input['image'].shape, features)
        # rpn_proposals [{'proposal_boxes': torch.Size([2000, 4]), 'objectness_logits': torch.Size([2000])} * batch]
        # auxiliary {'pred_objectness_logits' : [torch.Size([batch, 557568]),
        #                                        torch.Size([batch, 139392]),
        #                                        torch.Size([batch, 34848])]
        #             'pred_anchor_deltas' : [torch.Size([batch, 557568, 4]),
        #                                     torch.Size([batch, 139392, 4]),
        #                                     torch.Size([batch, 34848, 4])]
        #             'anchors' : [torch.Size([557568, 4])
        #                          torch.Size([139392, 4])
        #                          torch.Size([34848, 4])]}
        # return features, rpn_proposals, auxiliary

        pred = self.roi_heads(batched_input, features, rpn_proposals)
        # pred {'pred_class_logits': torch.Size([1024, 4])
        #       'pred_proposal_deltas': torch.Size([1024, 12])
        #       'viewpoint_logits': torch.Size([1024, 36])
        #       'viewpoint_residuals': torch.Size([1024, 36])
        #       'height_logits': torch.Size([1024, 6])
        #       'head_proposals' : [{'proposal_boxes': torch.Size([512, 4])
        #                           'objectness_logits': torch.Size([512])
        #                           'category': torch.Size([512, 1])
        #                           'bbox2d': torch.Size([512, 4])
        #                           'bbox3d': torch.Size([512, 6])
        #                           'object': torch.Size([512, 1])
        #                           'yaw': torch.Size([512, 2])} * batch]
        #       }

        # return features, rpn_proposals, auxiliary, pred
        pred['rpn_proposals'] = rpn_proposals
        pred.update(auxiliary)
        # # pred keys :  dict_keys(['scores', 'proposal_deltas', 'viewpoint_scores', 'viewpoint_residuals', 'height_scores', 'proposals', 'loss_instances'])
        return pred

    def preprocess_input(self, batched_input):
        image = batched_input['image'].permute(0, 3, 1, 2).to(cfg.Model.Structure.DEVICE)
        image = self.normalizer(image)
        image = F.pad(image,(4,4,2,2), "constant", 0)
        batched_input['image'] = image
        batched_input['bbox2d'] += torch.tensor([[[4, 2, 4, 2]]], dtype=torch.float32)
        batched_input['bbox3d'][:, :, :2] += torch.tensor([[[4, 2]]], dtype=torch.float32)
        return batched_input


class YOLO(ModelBase):
    pass