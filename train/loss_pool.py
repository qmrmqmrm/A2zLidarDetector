"""
def loss_obj(feature, prediction):
    prediction = {"box2d_map": [batch, channel, height, width],    (channel: y1,x1,y2,x2)
                "object_map": [batch, channel, height, width],    (channel: obj)
                "box3d": [batch, numbox, channel],  (channel: X,Y,Z,length,width,height) 단위??
                "yaw_class": [batch, numbox, channel],  (channel: yaw probabilities)
                "yaw_resid": [batch, numbox, channel],
                "class": [batch, numbox, channel]       (channel: class probabilities)
                }

    loss["box2d"] = smooth_l1(feature["box2d_map"], prediction["box2d_map"], feature["object_map"])
    loss["obj2d"] = binary_crossentropy(feature["obj2d_map"], prediction["object_map"])
    loss["class"] = crossentropy(feature["class"], prediction["class"], feature["object"])
    loss["yaw_class"] = crossentropy(feature["yaw"], prediction["yaw_class"], feature["object"])
    loss["yaw_resid"] = smooth_l1(feature["yaw"], prediction["yaw_resid"], feature["object"])
    loss["box3d"] = smooth_l1(feature["box3d"], prediction["box3d"], feature["object"])


def smooth_l1(feature, prediction, objectness)
    x = feature - prediction
    small_mask = torch.abs(x) < 1
    loss_map = 0.5 * x^2 * objectness * small_mask
                + (torch.abs(x) - 0.5) * objectness * (1 - small_mask)
    loss_scalar = torch.sum(loss_map)
    return loss_scalar
"""

import torch
from torch import nn
import numpy as np
from config import Config as cfg
from utils.util_function import pairwise_iou

class LossBase:
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()


class SmoothL1(LossBase):
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()

    def smooth_l1(self, foo, bar):
        pass


class Box2dRegression(SmoothL1):
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()


class Box3dRegression(SmoothL1):
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()


class YawRegression(SmoothL1):
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()


class CrossEntropy(LossBase):
    def __call__(self, grtr, pred, auxi):
        raise NotImplementedError()

    def smooth_l1(self, foo, bar):
        pass


class CategoryClassification(CrossEntropy):
    pass


class YawClassification(CrossEntropy):
    pass




class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
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
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if self.viewpoint:
            viewpoint_scores = self.viewpoint_pred(x)
            viewpoint_residuals = self.viewpoint_pred_residuals(x) if self.viewpoint_residual else None
            if self.height_training:
                height_scores = self.height_pred(x)
                return scores, proposal_deltas, viewpoint_scores, viewpoint_residuals, height_scores
            return scores, proposal_deltas, viewpoint_scores, viewpoint_residuals, None
        return scores, proposal_deltas, None, None, None


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,  # -1 del
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = self.anchors
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            gt_boxes_i = gt_boxes_i.to('cuda')
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            # if self.boundary_threshold >= 0:
            #     # Discard anchors that go out of the boundaries of the image
            #     # NOTE: This is legacy functionality that is turned off by default in Detectron2
            #     anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
            #     gt_objectness_logits_i[~anchors_inside_image] = -1
            # print(len(gt_boxes_i))
            # if len(gt_boxes_i) == 0:
            #     # These values won't be used anyway since the anchor is labeled as background
            #     gt_anchor_deltas_i = torch.zeros_like(anchors_i)
            # else:
                # TODO wasted computation for ignored boxes
            matched_gt_boxes = gt_boxes_i[matched_idxs]
            gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                anchors_i, matched_gt_boxes
            )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """


        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """

        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Return `num_samples` random samples from `labels`, with a fraction of
    positives no larger than `positive_fraction`.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D indices. The total number of indices is `num_samples` if possible.
            The fraction of positive indices is `positive_fraction` if possible.
    """
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx