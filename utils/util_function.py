import sys
import torch


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
    print("pairwise_intersection")
    print(boxes1.shape)
    print(boxes1)
    print(boxes1[:, None, 2:].shape)
    print(boxes2.shape)
    print(boxes2)
    print(boxes2[:, 2:].shape)
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1, boxes2):
    print("pairwise_iou")

    print(boxes1[0, :])
    area1 = (boxes1[:, 2] * boxes1[:, 3])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    print("area1.shape, area2.shape")
    print(area1.shape, area2.shape)
    inter = pairwise_intersection(boxes1, boxes2)
    print(inter)
    iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter),
                      torch.zeros(1, dtype=inter.dtype, device=inter.device),
                      )
    return iou
