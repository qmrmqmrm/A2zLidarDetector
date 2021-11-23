import sys, os
import torch
import cv2
import numpy as np

import config as cfg
import model.submodules.model_util as mu


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


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


def pairwise_iou(boxes1_yxlw, boxes2_yxlw):
    boxes1_tlbr = mu.convert_box_format_yxhw_to_tlbr(boxes1_yxlw)
    boxes2_tlbr = mu.convert_box_format_yxhw_to_tlbr(boxes2_yxlw)
    area1 = (boxes1_tlbr[:, 2] - boxes1_tlbr[:, 0]) * (boxes1_tlbr[:, 3] - boxes1_tlbr[:, 1])
    area2 = (boxes2_tlbr[:, 2] - boxes2_tlbr[:, 0]) * (boxes2_tlbr[:, 3] - boxes2_tlbr[:, 1])
    inter = pairwise_intersection(boxes1_tlbr, boxes2_tlbr)
    iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter),
                      torch.zeros(1, dtype=inter.dtype, device=inter.device),
                      )
    return iou


def draw_box(img, bboxes_2d):
    draw_img = img.copy()
    for bbox in bboxes_2d:
        x0 = int(bbox[0])
        x1 = int(bbox[2])
        y0 = int(bbox[1])
        y1 = int(bbox[3])
        draw_img = cv2.rectangle(draw_img, (x0, y0), (x1, y1), (255, 255, 255), 2)

    return draw_img


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
    elif data is None:
        print(f'{title} : None')
    elif isinstance(data, int):
        print(title, key, data)
    elif type(data) == np.ndarray:
        print(title, key, data.shape, type(data))
    else:
        print(title, key, data.shape)


def merge_and_slice_features(features):
    num_classes = cfg.Model.Structure.NUM_CLASSES
    loss_channel = cfg.Model.Structure.LOSS_CHANNEL
    sliced_features = dict()
    last_channel = 0
    for key in features:
        if key == 'head_output':
            for loss, dims in loss_channel.items():
                slice_dim = last_channel + num_classes * dims
                if loss == 'category':
                    slice_dim = slice_dim + 1
                sliced_features[loss] = features[key][..., last_channel:slice_dim]
                last_channel = slice_dim

        else:
            sliced_features[key] = features[key]
    return sliced_features


def slice_class(features):
    num_classes = cfg.Model.Structure.NUM_CLASSES
    loss_channel = cfg.Model.Structure.LOSS_CHANNEL
    sliced_features = features
    for loss, dims in loss_channel.items():
        if loss == 'category':
            num_classes = num_classes + 1
        else:
            num_classes = cfg.Model.Structure.NUM_CLASSES
        sliced_features[loss] = features[loss].reshape(features[loss].shape[0], features[loss].shape[1], num_classes,
                                                       dims)
    return sliced_features


def select_category(pred, ctgr_inds, key_to_select):
    ctgr_inds = ctgr_inds.to(torch.int64).unsqueeze(-1)
    select_pred = dict()
    for key in key_to_select:
        pred_key = pred[key]  # 0 1 2 3
        batch, num, cate, channel = pred_key.shape
        pred_padding = torch.zeros((batch, num, 1, channel), device=cfg.Hardware.DEVICE)
        pred_key = torch.cat([pred_padding, pred_key], dim=-2)  # 512 4 6
        gather_pred = torch.gather(pred_key, dim=2, index=ctgr_inds.repeat(1, 1, 1, pred_key.shape[-1])).squeeze(-2)
        select_pred[key] = gather_pred

    for key in pred:
        if key in key_to_select:
            continue
        select_pred[key] = pred[key]
    return select_pred


def compute_iou_general(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, N1, D1(>4))
    :param pred_yxhw: predicted bounding box in yxhw format (batch, N2, D2(>4))
    :return: iou (batch, N1, N2)
    """
    grtr_yxhw = np.expand_dims(grtr_yxhw, axis=-2)  # (batch, N1, 1, D1)
    pred_yxhw = np.expand_dims(pred_yxhw, axis=-3)  # (batch, 1, N2, D2)

    if grtr_tlbr is None:
        grtr_tlbr = mu.convert_box_format_yxhw_to_tlbr(grtr_yxhw)  # (batch, N1, 1, D1)
    if pred_tlbr is None:
        pred_tlbr = mu.convert_box_format_yxhw_to_tlbr(pred_yxhw)  # (batch, 1, N2, D2)

    inter_tl = np.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])  # (batch, N1, N2, 2)
    inter_br = np.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])  # (batch, N1, N2, 2)
    inter_hw = inter_br - inter_tl  # (batch, N1, N2, 2)
    inter_hw = np.maximum(inter_hw, 0)
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (batch, N1, N2)

    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]  # (batch, 1, N2)
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]  # (batch, N1, 1)
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)  # (batch, N1, N2)
    return iou


def pairwise_batch_iou(boxes1, boxes2):
    batches = boxes1.shape[0]
    batch_iou = list()
    for batch in range(batches):
        iou = pairwise_iou(boxes1[batch], boxes2[batch])
        batch_iou.append(iou)
    ious = torch.stack(batch_iou, dim=0)

    return ious


def rotated_iou_per_frame(grtr, pred, frame):
    print_structure('grtr', grtr)
    print_structure('pred', pred)
    height, wight, channel = grtr['image'][frame].shape
    grtr_bbox3d = grtr['bbox3d'][frame, ..., :4]
    pred_bbox3d = pred['bbox3d'][frame, ..., :4]
    grtr_rad = grtr['yaw_rads'][frame, ..., :4] * 180 / np.pi
    pred_rad = pred['yaw_rads'][frame, ..., :4] * 180 / np.pi
    grtr_rbox = np.concatenate([grtr_bbox3d, grtr_rad], axis=-1)
    pred_rbox = np.concatenate([pred_bbox3d, pred_rad], axis=-1)
    grtr_rbox = rotated_cv2(grtr_rbox, (height, wight))
    pred_rbox = rotated_cv2(pred_rbox, (height, wight))
    iou = jaccard(grtr_rbox, pred_rbox)
    return iou


def rotated_cv2(boxes, img_shape):
    b_boxes = list(map(lambda x: np.ceil(cv2.boxPoints(((x[0], x[1]), (x[2], x[3]), x[4]))), boxes))
    b_imgs = torch.from_numpy(
        np.array([cv2.fillConvexPoly(np.zeros(img_shape, dtype=np.uint8), np.int0(b), 1) for b in b_boxes]).astype(
            float)).float()
    return b_imgs


def jaccard(b_imgs, box_b):
    intersection = torch.FloatTensor()
    summation = torch.FloatTensor()
    for b_img in b_imgs:
        intersection = torch.cat([intersection, (b_img * box_b).sum((1, 2)).unsqueeze(0)])
        summation = torch.cat([summation, (b_img + box_b).sum((1, 2)).unsqueeze(0)])
    return intersection / (summation - intersection + 1.0)


def test():
    box1 = np.array([[100., 100., 50., 50., 45.],[500., 500., 50., 50., 45.]])
    box2 = np.array([[100., 100., 50., 50., 0.], [500., 500., 50., 50., 90.]])

    grtr_rbox = rotated_cv2(box1, (700, 700))
    pred_rbox = rotated_cv2(box2, (700, 700))
    iou = jaccard(grtr_rbox, pred_rbox)
    print(iou)


if __name__ == '__main__':
    test()
