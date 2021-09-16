import torch
from torch.nn import functional as F
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
