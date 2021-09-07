import torch

from utils.util_function import subsample_labels
from utils.util_function import pairwise_iou, print_structure
from config import Config as cfg

DEVICE = cfg.Model.Structure.DEVICE


def distribute_box_over_feature_map(anchors, bbox2d, anchor_matcher):
    """
    :param anchors:
                [torch.Size([557568(176 * 352 * 9), 4])
                torch.Size([139392(88 * 176 * 9), 4])
                torch.Size([34848(44 * 88 * 9), 4])]
    :param feature:
        {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
        'bbox2d': [batch, fixbox, 4], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
        'yaw': [batch, fixbox, 2]}
    :return:
        gt_labels: [[sum(HWA)=731808, 1]*batch]
        matched_gt_boxes: [[sum(HWA)=731808, 4]*batch]
    """
    anchors = torch.cat(anchors)
    gt_labels = []
    matched_gt_boxes = []

    print('anchors shape', anchors.shape)
    for i, box in enumerate(bbox2d):
        """
        image_size_i: (h, w) for the i-th image
        gt_boxes_i: ground-truth boxes for i-th image
        """
        print('box shape', box.shape)
        match_quality_matrix = pairwise_iou(box, anchors)
        matched_idxs, gt_labels_i = anchor_matcher(match_quality_matrix)
        print('gt_labels_i.shape', gt_labels_i.shape)

        # Matching is memory-expensive and may result in CPU tensors. But the result is small
        gt_labels_i = gt_labels_i.to(device=DEVICE)
        del match_quality_matrix

        # A vector of labels (-1, 0, 1) for each anchor
        gt_labels_i = _subsample_labels(gt_labels_i)
        gt_labels_i = gt_labels_i.to(DEVICE)
        if len(box) == 0:
            matched_gt_boxes_i = torch.zeros_like(anchors)
        else:
            matched_gt_boxes_i = box[matched_idxs]
        gt_labels.append(gt_labels_i)  # N,AHW
        matched_gt_boxes.append(matched_gt_boxes_i)

    return gt_labels, matched_gt_boxes


def _subsample_labels(label,
                      batch_size_per_image=cfg.Model.RPN.BATCH_SIZE_PER_IMAGE,
                      positive_fraction=cfg.Model.RPN.POSITIVE_FRACTION):
    """
    Randomly sample a subset of positive and negative examples, and overwrite
    the label vector to the ignore value (-1) for all elements that are not
    included in the sample.

    Args:
        labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
    """
    pos_idx, neg_idx = subsample_labels(
        label, batch_size_per_image, positive_fraction, 0
    )
    # Fill with the ignore label (-1), then set positive and negative labels
    label.fill_(-1)
    label.scatter_(0, pos_idx, 1)
    label.scatter_(0, neg_idx, 0)
    return label


def match_gt_with_anchors(gt, anchor):
    """
    :param gt: [num_gt, 4]
    :param anchor: [sum HWA over scale, 4]
    :return:
    """
    match_quality_matrix = pairwise_iou(anchor, gt)  # [sum HWA, num_gt]
    matched_vals, match_anchor_inds = match_quality_matrix.max(dim=0)
    return match_anchor_inds
