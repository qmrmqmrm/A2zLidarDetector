# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
import itertools

from utils.util_class import ShapeSpec
from model.submodules.anchor import DefaultAnchorGenerator
from model.submodules.matcher import Matcher
from model.submodules.box_regression import Box2BoxTransform

from model.submodules.sampling import subsample_labels
from utils.util_function import pairwise_iou
from config import Config as cfg


class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        # anchor_generator = build_anchor_generator(cfg, input_shape)
        anchor_generator = DefaultAnchorGenerator(input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
                len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))

        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len = cfg.Model.RPN.MIN_SIZE
        self.in_features = cfg.Model.RPN.INPUT_FEATURES
        self.nms_thresh = cfg.Model.RPN.NMS_THRESH

        self.smooth_l1_beta = cfg.Model.RPN.SMOOTH_L1_BETA
        self.min_box_size = 0
        self.loss_weight = cfg.Model.RPN.LOSS_WEIGHT
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.Model.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.Model.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.Model.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.Model.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.Model.RPN.BOUNDARY_THRESH  # del

        self.anchor_generator = DefaultAnchorGenerator(
            [input_shape[f] for f in self.in_features]
        )

        self.anchor_matcher = Matcher(
            cfg.Model.RPN.IOU_THRESHOLDS, cfg.Model.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = StandardRPNHead([input_shape[f] for f in self.in_features])

    def forward(
            self,
            image_shape,
            features
    ):
        """

         :param image_shape: (torch.Size): torch.Size([2, 3, 704, 1408])
         :param features: (dict[torch.Tensor]):
                    {'p2': torch.Size([batch, 256, 176, 352]),'p3': torch.Size([batch, 256, 88, 176]),
                    'p4': torch.Size([batch, 256, 44, 88]), 'p5': torch.Size([batch, 256, 22, 44])}

         :return:
             proposals (list[dict[torch.tensor]]): [{'proposal_boxes': torch.Size([2000, 4]),
                                                    'objectness_logits': torch.Size([2000])} * batch]
             auxiliary (dict[list[torch.tensor]]): {'pred_objectness_logits' : [torch.Size([batch, 557568]),
                                                    torch.Size([batch, 139392]),
                                                    torch.Size([batch, 34848])]
                                                    'pred_anchor_deltas' : [torch.Size([batch, 557568, 4]),
                                                                            torch.Size([batch, 139392, 4]),
                                                                            torch.Size([batch, 34848, 4])]
                                                    'anchors' : [torch.Size([557568, 4])
                                                                 torch.Size([139392, 4])
                                                                 torch.Size([34848, 4])]}
        """
        image_sizes = [(image_shape[2], image_shape[3]) for i in range(image_shape[0])]
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        # print(pred_anchor_deltas[0].shape) # torch.Size([2, 557568, 4])
        aux_outputs = {'pred_objectness_logits': pred_objectness_logits, 'pred_anchor_deltas': pred_anchor_deltas,
                       'anchors': anchors}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes
        )
        return proposals, aux_outputs

    def predict_proposals(
            self,
            anchors,
            pred_objectness_logits,
            pred_anchor_deltas,
            image_sizes,
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors, pred_anchor_deltas):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


def find_top_rpn_proposals(
        proposals,
        pred_objectness_logits,
        image_sizes,
        nms_thresh,
        pre_nms_topk,
        post_nms_topk,
        min_box_side_len,
        training
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
            itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        h, w = image_size
        x1 = boxes[:, 0].clamp(min=0, max=w)
        y1 = boxes[:, 1].clamp(min=0, max=h)
        x2 = boxes[:, 2].clamp(min=0, max=w)
        y2 = boxes[:, 3].clamp(min=0, max=h)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        # filter empty boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > 0) & (heights > 0)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]

        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]
        res = dict()
        res['proposal_boxes'] = boxes[keep]
        res['objectness_logits'] = scores_per_img[keep]
        results.append(res)
        # print((res.gt_classes))
    return results


def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep
