import sys
import torch
import cv2

from config import Config as cfg


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
    boxes1 = boxes1.to(cfg.Model.Structure.DEVICE)
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

