import torch
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
import numpy as np

import config as cfg
import utils.util_function as uf


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


def apply_box_deltas(anchors, deltas):
    """
    :param anchors: anchor boxes in image pixel in yxhw format (batch,height,width,4)
    :param deltas: box conv output corresponding to dy,dx,dh,dw
    :return:
    """
    anchor_yx, anchor_hw = anchors[..., :2], anchors[..., 2:4]
    stride = anchor_yx[0, 1, 0, 0, 0] - anchor_yx[0, 0, 0, 0, 0]
    delta_yx, delta_hw = deltas[..., :2], deltas[..., 2:4]
    delta_hw = torch.clamp(delta_hw, -2, 2)
    anchor_yx = anchor_yx.view(delta_yx.shape)
    anchor_hw = anchor_hw.view(delta_hw.shape)
    bbox_yx = anchor_yx + torch.sigmoid(delta_yx) * stride * 2 - (stride / 2)
    bbox_hw = anchor_hw * torch.exp(delta_hw)
    bbox = torch.cat([bbox_yx, bbox_hw], dim=-1)
    return bbox


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
        categories = torch.argmax(pred["category"], dim=-1)  # (batch, N)
        best_probs = torch.amax(pred["category"], dim=-1)  # (batch, N)
        objectness = pred["objectness"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred["category"].shape
        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(numctgr):
            ctgr_mask = (categories == ctgr_idx).to(dtype=torch.int64)  # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., None]  # (batch, N, 4)

            ctgr_scores = scores * ctgr_mask  # (batch, N)
            for frame_idx in range(batch):
                # NMS
                selected_indices = box_ops.batched_nms(ctgr_boxes[frame_idx], ctgr_scores[frame_idx],
                                                       categories[frame_idx], self.iou_thresh[ctgr_idx])
                print(selected_indices.shape)

                numsel = ctgr_boxes[frame_idx].shape[0]
                zero = torch.ones((numsel - selected_indices.shape[0]), device=self.device) * -1
                selected_indices = torch.cat([selected_indices, zero], dim=0)
                batch_indices[frame_idx].append(selected_indices)
        batch_indices = [torch.cat(ctgr_indices, dim=-1) for ctgr_indices in batch_indices]
        batch_indices = torch.stack(batch_indices, dim=0)  # (batch, K*max_output)
        batch_indices = torch.maximum(batch_indices, torch.zeros(batch_indices.shape, device=self.device)).to(
            dtype=torch.int64)

        result = {'objectness': [], 'bbox2d': [], 'bbox3d': [], 'anchor_id': [], 'category': [], 'yaw': [],
                  'yaw_rads': [], }

        for i, indices in enumerate(batch_indices.to(dtype=torch.int64)):
            result['objectness'].append(pred['objectness'][i, indices])
            result['bbox2d'].append(pred['bbox2d'][i, indices])
            result['bbox3d'].append(pred['bbox3d'][i, indices])
            result['anchor_id'].append(pred['anchor_id'][i, indices])
            result['category'].append(pred['category'][i, indices])
            result['yaw'].append(pred['yaw'][i, indices])
            result['yaw_rads'].append(pred['yaw_rads'][i, indices])
        for key, val in result.items():
            result[key] = torch.stack(val, dim=0)
        return result