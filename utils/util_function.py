import sys, os
import torch
import cv2
from detectron2.layers import nonzero_tuple

from config import Config as cfg
from model.submodules.matcher import Matcher

DEVICE = cfg.Model.Structure.DEVICE


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def build_optimizer(model: torch.nn.Module, lr) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = dict()
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = 0.0001
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = 0.0
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = lr * 1.0
            weight_decay = 0.0001
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return optimizer


def pairwise_intersection(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1, boxes2):
    # boxes1 = boxes1.to(DEVICE)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter = pairwise_intersection(boxes1, boxes2)

    iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter),
                      torch.zeros(1, dtype=inter.dtype, device=inter.device),
                      )
    return iou


def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


def slice_features(feature, output_composition=cfg.Model.Output.OUT_COMPOSITION):
    index = 0
    slices = dict()
    for name, channel in output_composition:
        slices[name] = feature[..., index:index + channel]
        index += channel
    return slices


def draw_box(img, bboxes_2d):
    draw_img = img.copy()
    for bbox in bboxes_2d:
        x0 = int(bbox[0])
        x1 = int(bbox[2])
        y0 = int(bbox[1])
        y1 = int(bbox[3])
        draw_img = cv2.rectangle(draw_img, (x0, y0), (x1, y1), (255, 255, 255), 2)

    return draw_img


def subsample_labels(
        labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

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
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    # gt_classes : torch.Size([A + gt_num])
    # self.batch_size_per_image : 512
    # self.positive_fraction : 0.25
    # self.num_classes : 3
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

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


def print_structure(title, data, key=""):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif isinstance(data, dict):
        for subkey, datum in data.items():
            print_structure(title, datum, f"{key}/{subkey}")
    elif isinstance(data, str):
        print(title, key, data)
    elif isinstance(data, tuple):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    else:
        print(title, key, data.shape)


def align_gt_with_pred(proposals, targets):
    """
    :param proposals: (list[dict[torch.tensor]])
             [{'proposal_boxes': torch.Size([2000, 4]), 'objectness_logits': torch.Size([2000])} * batch]
    :param targets: {'image': [batch, height, width, channel], 'category': [batch, fixbox, 1],
                    'bbox2d': [batch, num_gt, 4], 'bbox3d': [batch, num_gt, 6], 'object': [batch, num_gt, 1],
                    'yaw': [batch, num_gt, 2]}
    :return:
    """

    proposal_matcher = Matcher(
        cfg.Model.ROI_HEADS.IOU_THRESHOLDS,
        cfg.Model.ROI_HEADS.IOU_LABELS,
        allow_low_quality_matches=False,
    )
    gt_aligned = {'gt_category': [], 'yaw': [], 'yaw_rads': [], 'bbox3d': [], 'bbox2d': []}
    match_result = []
    for bbox2d_per_image, bbox3d_per_image, category_per_image, yaw_par_image, yaw_rads_par_image, proposal_per_image in zip(
            targets['bbox2d'], targets['bbox3d'], targets['category'], targets['yaw'], targets['yaw_rads'], proposals):
        match_quality_matrix = pairwise_iou(bbox2d_per_image, proposal_per_image.get("proposal_boxes"))
        matched_idxs, matched_labels = proposal_matcher(match_quality_matrix)
        # matched_idxs : torch.Size([512])
        # matched_labels : torch.Size([512]) (0: unmatched, -1: ignore, 1: matched)
        # NOTE: here the indexing waste some compute, because heads
        # like masks, keypoints, etc, will filter the proposals again,
        # (by foreground/background, or number of keypoints in the image, etc)
        # so we essentially index the data twice.
        match_result.append(matched_labels)
        gt_aligned['gt_category'].append(category_per_image[matched_idxs])
        gt_aligned['yaw'].append(yaw_par_image[matched_idxs])
        gt_aligned['yaw_rads'].append(yaw_rads_par_image[matched_idxs])
        gt_aligned['bbox3d'].append(bbox3d_per_image[matched_idxs])
        gt_aligned['bbox2d'].append(bbox2d_per_image[matched_idxs])

    for key in gt_aligned:
        gt_aligned[key] = torch.cat(gt_aligned[key])
    match_result = torch.cat(match_result)
    return gt_aligned, match_result


def _sample_proposals(matched_idxs, matched_labels, gt_classes):
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
    num_classes = cfg.Model.Structure.NUM_CLASSES
    batch_size_per_image = cfg.Model.ROI_HEADS.BATCH_SIZE_PER_IMAGE
    positive_fraction = cfg.Model.ROI_HEADS.POSITIVE_FRACTION
    # Get the corresponding GT for each proposal
    if has_gt:
        gt_classes = gt_classes[matched_idxs]
        # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
        gt_classes[matched_labels == 0] = num_classes
        # Label ignore proposals (-1 label)
        gt_classes[matched_labels == -1] = -1
    else:
        gt_classes = torch.zeros_like(matched_idxs) + num_classes
    # gt_classes : torch.Size([2000 + gt_num])
    # self.batch_size_per_image : 512
    # self.positive_fraction : 0.25
    # self.num_classes : 3
    sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        gt_classes, batch_size_per_image, positive_fraction, num_classes
    )
    # sampled_fg_idxs : pos indax
    # sampled_fg_idxs : neg indax
    sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
    # sampled_idxs 512
    return sampled_idxs, gt_classes[sampled_idxs]
