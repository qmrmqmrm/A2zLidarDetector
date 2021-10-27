import torch
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
import numpy as np

import config as cfg
import utils.util_function as uf

device = cfg.Hardware.DEVICE


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def remove_padding(batch_input):
    batch, _, _, _ = batch_input['image'].shape
    bbox2d_batch = list()
    category_batch = list()
    yaw_batch = list()
    yaw_rads_batch = list()
    bbox3d_batch = list()
    for i in range(batch):
        bbox2d = batch_input['bbox2d'][i, :]
        weight = bbox2d[:, 2] - bbox2d[:, 0]
        x = torch.where(weight > 0)
        bbox2d = bbox2d[:x[0][-1] + 1, :]
        bbox2d_batch.append(bbox2d)
        # print('\nhead bbox2d.shape :', bbox2d.shape)
        # print('head bbox2d :', bbox2d)

        category = batch_input['category'][i, :]
        category = category[torch.where(category < 3)]
        category_batch.append(category)

        valid_yaw = batch_input['yaw'][i, :][torch.where(batch_input['yaw'][i, :] < 13)]
        yaw_batch.append(valid_yaw)

        valid_yaw_rads = batch_input['yaw_rads'][i, :][torch.where(batch_input['yaw_rads'][i, :] >= 0)]
        yaw_rads_batch.append(valid_yaw_rads)

        weight_3d = batch_input['bbox3d'][i, :, 2]
        valid_3d = batch_input['bbox3d'][i, :][torch.where(weight_3d > 0)]
        bbox3d_batch.append(valid_3d)

    new_batch_input = {'bbox2d': bbox2d_batch, 'category': category_batch, 'yaw': yaw_batch, 'yaw_rads': yaw_rads_batch,
                       'bbox3d': bbox3d_batch, 'image': batch_input['image']}
    return new_batch_input


def convert_box_format_tlbr_to_yxhw(boxes_tlbr):
    """
    :param boxes_tlbr: any tensor type, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_yx = (boxes_tlbr[..., 0:2] + boxes_tlbr[..., 2:4]) / 2  # center y,x
    boxes_hw = boxes_tlbr[..., 2:4] - boxes_tlbr[..., 0:2]  # y2,x2 = y1,x1 + h,w
    output = [boxes_yx, boxes_hw]
    output = concat_box_output(output, boxes_tlbr)
    return output


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: any tensor type, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_tl = boxes_yxhw[..., 0:2] - (boxes_yxhw[..., 2:4] / 2)  # y1,x1 = cy,cx + h/2,w/2
    boxes_br = boxes_tl + boxes_yxhw[..., 2:4]  # y2,x2 = y1,x1 + h,w
    output = [boxes_tl, boxes_br]
    output = concat_box_output(output, boxes_yxhw)
    return output


def concat_box_output(output, boxes):
    num, dim = boxes.shape[-2:]

    # if there is more than bounding box, append it  e.g. category, distance
    if dim > 4:
        auxi_data = boxes[..., 4:]
        output.append(auxi_data)

    if torch.is_tensor(boxes):
        output = torch.cat(output, dim=-1)
        # output = output.to(dtype=boxes.type)
    else:
        output = np.concatenate(output, axis=-1)
        output = output.astype(boxes.dtype)
    return output


def apply_box_deltas_2d(anchors, deltas, stride=None):
    """
    :param anchors: anchor boxes in image pixel in yxhw format (batch,height,width,4)
    :param deltas: box conv output corresponding to dy,dx,dh,dw
    :return:
    """
    anchor_yx, anchor_hw = anchors[..., :2], anchors[..., 2:4]
    if len(anchor_yx.shape) == 5:
        stride = anchor_yx[0, 1, 0, 0, 0] - anchor_yx[0, 0, 0, 0, 0]
    delta_yx, delta_hw = deltas[..., :2], deltas[..., 2:4]
    anchor_yx = anchor_yx.view(delta_yx.shape)
    anchor_hw = anchor_hw.view(delta_hw.shape)
    # bbox_yx = anchor_yx + torch.sigmoid(delta_yx) * stride * 1.4 - stride * 0.2
    bbox_yx = anchor_yx + delta_yx * stride
    bbox_hw = anchor_hw * torch.exp(delta_hw)
    bbox = torch.cat([bbox_yx, bbox_hw], dim=-1)
    return bbox


def get_deltas_2d(anchors, bboxes, stride=None):
    bboxes_yxlw = convert_box_format_tlbr_to_yxhw(bboxes)
    anchors_yxlw = convert_box_format_tlbr_to_yxhw(anchors)
    anchor_yx, anchor_lw = anchors_yxlw[..., :2], anchors_yxlw[..., 2:4]
    bboxes_yx, bboxes_lw = bboxes_yxlw[..., :2], bboxes_yxlw[..., 2:4]
    anchor_yx = anchor_yx.view(bboxes_yx.shape)
    anchor_lw = anchor_lw.view(bboxes_lw.shape)
    delta_yx = (bboxes_yx - anchor_yx) / stride
    delta_lw = torch.log(bboxes_lw / anchor_lw + 1e-10)
    valid_mask = (bboxes_lw[..., 0:1] > 0)
    delta_bbox = torch.cat([delta_yx, delta_lw], dim=-1) * valid_mask
    return delta_bbox


def apply_box_deltas_3d(anchors, deltas, category, stride):
    """
    :param anchors: anchor boxes in image pixel in yxhw format (batch,height,width,4)
    :param deltas: box conv output corresponding to dy,dx,dh,dw
    :return:
    """
    anchor_yx, anchor_lw = anchors[..., :2], anchors[..., 2:4]
    anchor_h = torch.tensor([149.6, 130.05, 147.9, 1.0])  # Mean heights encoded
    anchor_z = anchor_h / 2
    stride = torch.pow(2, stride + 2).view(anchor_yx.shape[0], -1).unsqueeze(-1).to(device=device)
    delta_yx, delta_lw, delta_h, delta_z = deltas[..., :2], deltas[..., 2:4], deltas[..., 4:5], deltas[..., 5:6]

    bbox_yx = anchor_yx + delta_yx * stride
    bbox_lw = anchor_lw * torch.exp(delta_lw)
    bbox_h = anchor_h[category] * torch.exp(delta_h)
    bbox_z = anchor_z[category] + delta_z * anchor_h[category] / 4
    bbox = torch.cat([bbox_yx, bbox_lw, bbox_z, bbox_h], dim=-1)
    return bbox


def get_deltas_3d(anchors, bboxes, category, stride=None):
    category = category.to(dtype=torch.int64)
    anchor_yx, anchor_lw = anchors[..., :2], anchors[..., 2:4]
    anchor_h = torch.tensor([149.6, 130.05, 147.9, 1.0])  # Mean heights encoded
    anchor_z = anchor_h / 2
    bboxes_yx, bboxes_lw, bboxes_h, bboxes_z = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., 4:5], bboxes[..., 5:6]
    anchor_yx = anchor_yx.view(bboxes_yx.shape)
    anchor_lw = anchor_lw.view(bboxes_lw.shape)
    delta_h = torch.log(bboxes_h / anchor_h[category] + 1e-10)
    delta_z = (bboxes_z - anchor_z[category]) / (anchor_h[category] / 4)
    delta_yx = (bboxes_yx - anchor_yx) / stride
    delta_lw = torch.log(bboxes_lw / anchor_lw + 1e-10)
    valid_mask = (bboxes_lw[..., 0:1] > 0)
    delta_bbox = torch.cat([delta_yx, delta_lw, delta_z, delta_h], dim=-1) * valid_mask
    return delta_bbox


class NonMaximumSuppression:
    def __init__(self, max_out=cfg.NMS.MAX_OUT,
                 iou_thresh=cfg.NMS.IOU_THRESH,
                 score_thresh=cfg.NMS.SCORE_THRESH,
                 category_names=cfg.Datasets.Standard.CATEGORY_NAMES
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.category_names = category_names
        self.device = cfg.Hardware.DEVICE

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False):
        """

        :param pred:
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'category' : torch.Size([batch, 512, class_num, 1])
            'bbox3d' : torch.Size([batch, 512, class_num, 6])
            'yaw' : torch.Size([batch, 512, class_num, 12])
            'yaw_rads' : torch.Size([batch, 512, class_num, 12])
            }
        :param max_out:
        :param iou_thresh:
        :param score_thresh:
        :param merged:
        :return:
        """
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        nms_res = self.pure_nms(pred)
        return nms_res

    def pure_nms(self, pred):
        """
        :param pred:
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'category' : torch.Size([batch, 512, class_num, 1])
            'bbox3d' : torch.Size([batch, 512, class_num, 6])
            'yaw' : torch.Size([batch, 512, class_num, 12])
            'yaw_rads' : torch.Size([batch, 512, class_num, 12])
            }
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        boxes = pred['bbox2d']  # (batch, N, 4(tlbr))
        category = pred["category"]
        categories = torch.argmax(category, dim=-1)  # (batch, N)
        best_probs = torch.amax(category, dim=-1)  # (batch, N)
        objectness = pred["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = category.shape
        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = (categories == ctgr_idx).to(dtype=torch.int64)  # (batch, N)
            score_mask = (scores >= self.score_thresh[ctgr_idx - 1]).squeeze(-1)
            nms_mask = ctgr_mask * score_mask
            ctgr_boxes = boxes * nms_mask[..., None]  # (batch, N, 4)
            ctgr_scores = scores * nms_mask  # (batch, N)
            ctgr_scores_numpy = ctgr_scores.to('cpu').detach().numpy()
            ctgr_scores_quant = np.quantile(ctgr_scores_numpy,
                                            np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.995, 0.999, 1]))
            print(f"ctgr_scores quantile {ctgr_idx}:", ctgr_scores_quant)
            for frame_idx in range(batch):
                # NMS

                selected_indices = box_ops.batched_nms(ctgr_boxes[frame_idx], ctgr_scores[frame_idx],
                                                       categories[frame_idx], self.iou_thresh[ctgr_idx - 1])
                selected_indices = selected_indices[:self.max_out[frame_idx]]
                max_num = self.max_out[frame_idx]

                if selected_indices.numel() <= max_num:
                    zero = torch.ones((max_num - selected_indices.shape[0]), device=self.device) * -1
                selected_indices = torch.cat([selected_indices, zero], dim=0).to(dtype=torch.int64)
                batch_indices[frame_idx].append(selected_indices)
                # print('pred', ctgr_boxes[frame_idx, selected_indices])
                # iou = uf.pairwise_iou(ctgr_boxes[frame_idx, selected_indices], ctgr_boxes[frame_idx, selected_indices])  # (batch, N, M)
                # print('selected_indices iou', iou.shape)
                # iou_where = iou[torch.where((iou > 0) * (iou < 0.1))]
                # iou_where_not = iou[torch.where((iou > 0.1) * (iou <= 1))]
                # print('iou O', iou_where)
                # print('iou X', iou_where_not)
        batch_indices = [torch.cat(ctgr_indices, dim=-1) for ctgr_indices in batch_indices]
        batch_indices = torch.stack(batch_indices, dim=0)  # (batch, K*max_output)
        batch_indices = torch.maximum(batch_indices, torch.zeros(batch_indices.shape, device=self.device)).to(
            dtype=torch.int64)

        result = {'object': [], 'bbox2d': [], 'bbox3d': [], 'anchor_id': [], 'category': [], 'yaw': [],
                  'yaw_rads': [], }
        for i, indices in enumerate(batch_indices.to(dtype=torch.int64)):
            for key in result.keys():
                result[key].append(pred[key][i, indices])
        for key, val in result.items():
            result[key] = torch.stack(val, dim=0)
        return result
