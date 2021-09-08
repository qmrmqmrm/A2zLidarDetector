# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from utils.util_class import ShapeSpec
import config as cfg
from model.submodules.matcher import Matcher
from model.submodules.model_util import Conv2d
from model.submodules.poolers import ROIPooler
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.proposal_utils import add_ground_truth_to_proposals
from model.submodules.weight_init import c2_msra_fill, c2_xavier_fill
from utils.batch_norm import get_norm
from utils.util_function import pairwise_iou, subsample_labels
from utils.util_function import print_structure


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

    def forward(self, batch_input, features, proposals):
        """


        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, input_shape):
        super(StandardROIHeads, self).__init__(input_shape)
        self._init_box_head(input_shape)

    def _init_box_head(self, input_shape):
        # fmt: off
        self.in_features = cfg.Model.ROI_HEADS.INPUT_FEATURES
        pooler_resolution = cfg.Model.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.Model.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = "ROIAlignV2"
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=int((cfg.Model.RESNET.OUT_FEATURES[-1].split('res'))[1])
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = FastRCNNConvFCHead(
            ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        self.box_predictor = FastRCNNOutputLayers(self.box_head.output_shape)

    def forward(self, batch_input, features, proposals):
        # proposals: [{'proposal_boxes': torch.Size([512, 4]), 'objectness_logits': torch.Size([512])} * batch]
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x["proposal_boxes"] for x in proposals])
        # torch.Size([batch * 512, 256, 7, 7])
        box_features = self.box_head(box_features)
        # torch.Size([batch * 512, 1024])
        pred_instances = self.box_predictor(box_features)
        pred_instances["head_proposals"] = proposals
        return pred_instances


class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.Model.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.Model.ROI_BOX_HEAD.CONV_DIM
        conv_dims = [conv_dim] * num_conv
        num_fc = cfg.Model.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.Model.ROI_BOX_HEAD.FC_DIM
        norm = cfg.Model.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            c2_msra_fill(layer)
        for layer in self.fcs:
            c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_shape):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()
        num_classes = cfg.Model.Structure.NUM_CLASSES
        cls_agnostic_bbox_reg = cfg.Model.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(self.box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.yaw = cfg.Model.Structure.YAW
        viewpoint_bins = cfg.Model.Structure.VP_BINS
        self.yaw_residual = cfg.Model.Structure.YAW_RESIDUAL
        self.height_training = cfg.Model.Structure.HEIGHT_TRAINING
        if self.yaw:
            self.yaw_pred = nn.Linear(input_size, viewpoint_bins * num_classes)
            # torch.nn.init.kaiming_normal_(self.yaw_pred.weight,nonlinearity='relu')
            nn.init.xavier_normal_(self.yaw_pred.weight)
            nn.init.constant_(self.yaw_pred.bias, 0)
        if self.yaw_residual:
            self.yaw_pred_residuals = nn.Linear(input_size, viewpoint_bins * num_classes)
            nn.init.xavier_normal_(self.yaw_pred_residuals.weight)
            # torch.nn.init.kaiming_normal_(self.yaw_pred_residuals.weight,nonlinearity='relu')
            nn.init.constant_(self.yaw_pred_residuals.bias, 0)
        if self.height_training:
            self.height_pred = nn.Linear(input_size, 2 * num_classes)
            nn.init.xavier_normal_(self.height_pred.weight)
            # torch.nn.init.kaiming_normal_(self.height_pred.weight,nonlinearity='relu')
            nn.init.constant_(self.height_pred.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        pred = dict()
        pred["head_class_logits"] = self.cls_score(x)
        # torch.Size([batch * 512, 4(num_classes(3) + 1)])
        pred['bbox_3d_logits'] = self.bbox_pred(x)
        # torch.Size([batch * 512, 12(num_bbox_reg_classes(3) * box_dim(4))])
        if self.yaw:
            pred['head_yaw_logits'] = self.yaw_pred(x)
            # torch.Size([batch * 512, 36(viewpoint_bins(6) * num_classes(3))])
            pred['head_yaw_residuals'] = self.yaw_pred_residuals(x) if self.yaw_residual else None
            # torch.Size([batch * 512, 36(viewpoint_bins(6) * num_classes(3))])
            if self.height_training:
                pred['head_height_logits'] = self.height_pred(x)
                # torch.Size([batch * 512, 6(2 * num_classes(3))])

        return pred
