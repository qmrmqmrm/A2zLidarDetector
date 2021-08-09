# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from utils.util_class import ShapeSpec
from config import Config as cfg
from model.submodules.matcher import Matcher
from model.submodules.model_util import Conv2d
from model.submodules.poolers import ROIPooler
from model.submodules.box_regression import Box2BoxTransform
from model.submodules.proposal_utils import add_ground_truth_to_proposals
from model.submodules.sampling import subsample_labels
from model.submodules.weight_init import c2_msra_fill, c2_xavier_fill
from utils.batch_norm import get_norm
from utils.util_function import pairwise_iou


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image = cfg.Model.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.Model.ROI_HEADS.POSITIVE_FRACTION
        self.num_classes = cfg.Model.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.Model.ROI_HEADS.PROPOSAL_APPEND_GT
        self.cls_agnostic_bbox_reg = cfg.Model.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.weights = cfg.Model.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        self.smooth_l1_beta = cfg.Model.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING
        self.weights_height = cfg.Model.Structure.WEIGHTS_HEIGHT
        self.vp_weight_loss = cfg.Model.Structure.VP_WEIGHT_LOSS
        self.test_score_thresh = cfg.Model.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.Model.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = 100
        self.proposal_matcher = Matcher(
            cfg.Model.ROI_HEADS.IOU_THRESHOLDS,
            cfg.Model.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x["gt_bbox2D"] for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.get("gt_bbox2D"), proposals_per_image.get("proposal_boxes")
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.get("gt_category_id")
            )
            # Set target attributes of the sampled proposals:

            proposals_per_image['proposal_boxes'] = proposals_per_image.get("proposal_boxes")[sampled_idxs]
            proposals_per_image['objectness_logits'] = proposals_per_image.get("objectness_logits")[sampled_idxs]
            proposals_per_image['gt_category_id'] = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.items():
                    if not trg_name.startswith("gt_") in proposals_per_image:
                        proposals_per_image[trg_name] = trg_value[sampled_targets]

            proposals_with_gt.append(proposals_per_image)

        # # Log the number of fg/bg samples that are selected for training ROI heads
        # storage = get_event_storage()
        # storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        # storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, features, proposals, targets=None):
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

    def forward(self, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """

        head_proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        # if self.training:
        #     losses = self._forward_box(features_list, proposals)
        #     # During training the proposals used by the box head are
        #     # used by the mask, keypoint (and densepose) heads.
        #     losses.update(self._forward_mask(features_list, proposals))
        #     losses.update(self._forward_keypoint(features_list, proposals))
        #     return proposals, losses
        # else:
        pred_instances = self._forward_box(features, head_proposals)
        # During inference cascaded prediction is used: the mask and keypoints heads are only
        # applied to the top scoring box detections.
        # pred_instances = self.forward_with_given_boxes(features, pred_instances)
        return pred_instances, head_proposals

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x["proposal_boxes"] for x in proposals])
        box_features = self.box_head(box_features)
        pred = self.box_predictor(box_features)
        del box_features
        return pred

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
        num_classes = cfg.Model.ROI_HEADS.NUM_CLASSES
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

        self.viewpoint = cfg.Model.Structure.VIEWPOINT
        viewpoint_bins = cfg.Model.Structure.VP_BINS
        self.viewpoint_residual = cfg.Model.Structure.VIEWPOINT_RESIDUAL
        self.height_training = cfg.Model.Structure.HEIGHT_TRAINING
        if self.viewpoint:
            self.viewpoint_pred = nn.Linear(input_size, viewpoint_bins * num_classes)
            # torch.nn.init.kaiming_normal_(self.viewpoint_pred.weight,nonlinearity='relu')
            nn.init.xavier_normal_(self.viewpoint_pred.weight)
            nn.init.constant_(self.viewpoint_pred.bias, 0)
        if self.viewpoint_residual:
            self.viewpoint_pred_residuals = nn.Linear(input_size, viewpoint_bins * num_classes)
            nn.init.xavier_normal_(self.viewpoint_pred_residuals.weight)
            # torch.nn.init.kaiming_normal_(self.viewpoint_pred_residuals.weight,nonlinearity='relu')
            nn.init.constant_(self.viewpoint_pred_residuals.bias, 0)
        if self.height_training:
            self.height_pred = nn.Linear(input_size, 2 * num_classes)
            nn.init.xavier_normal_(self.height_pred.weight)
            # torch.nn.init.kaiming_normal_(self.height_pred.weight,nonlinearity='relu')
            nn.init.constant_(self.height_pred.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        pred = dict()
        pred["pred_class_logits"] = self.cls_score(x)
        pred['pred_proposal_deltas'] = self.bbox_pred(x)
        if self.viewpoint:
            pred['viewpoint_logits'] = self.viewpoint_pred(x)
            pred['viewpoint_residuals'] = self.viewpoint_pred_residuals(x) if self.viewpoint_residual else None
            if self.height_training:
                pred['height_logits'] = self.height_pred(x)

        return pred
