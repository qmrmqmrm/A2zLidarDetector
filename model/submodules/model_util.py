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


class NonMaximumSuppression:
    def __init__(self, max_out=None,
                 iou_thresh=None,
                 score_thresh=None,
                 category_names=None
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.category_names = category_names

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False):
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        nms_res = self.pure_nms(pred, merged)
        return nms_res

    # @tf.function
    def pure_nms(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred = self.merged_scale(pred)

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])    # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)           # (batch, N)
        if cfg.Model.Output.MINOR_CTGR:
            sign_ctgr = tf.cast(categories == self.category_names.index("Traffic sign"), dtype=tf.float32) * \
                        tf.cast(tf.argmax(pred["sign_ctgr"], axis=-1), dtype=tf.float32)
            mark_ctgr = tf.cast(categories == self.category_names.index("Road mark"), dtype=tf.float32) * \
                        tf.cast(tf.argmax(pred["mark_ctgr"], axis=-1), dtype=tf.float32)
            minor_ctgr = sign_ctgr + mark_ctgr
        else:
            minor_ctgr = tf.zeros_like(categories, dtype=tf.float32)

        best_probs = tf.reduce_max(pred["category"], axis=-1)       # (batch, N)
        objectness = pred["object"][..., 0]                     # (batch, N)
        scores = objectness * best_probs                            # (batch, N)
        batch, numbox, numctgr = pred["category"].shape

        distances = pred["distance"][..., 0]
        anchor_inds = pred["anchor_ind"][..., 0]

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)   # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]                 # (batch, N, 4)

            ctgr_scores = scores * ctgr_mask                                # (batch, N)
            for frame_idx in range(batch):
                selected_indices = tf.image.non_max_suppression(
                    boxes=ctgr_boxes[frame_idx],
                    scores=ctgr_scores[frame_idx],
                    max_output_size=self.max_out[ctgr_idx],
                    iou_threshold=self.iou_thresh[ctgr_idx],
                    score_threshold=self.score_thresh[ctgr_idx],
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)             # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        categories = tf.cast(categories, dtype=tf.float32)
        # "bbox": 4, "object": 1, "category": 1, "minor_ctgr": 1, "distance": 1, "score": 1, "anchor_inds": 1
        result = tf.stack([objectness, categories, minor_ctgr, distances, best_probs, scores, anchor_inds], axis=-1)

        result = tf.concat([pred["yxhw"], result], axis=-1)                # (batch, N, 10)
        result = tf.gather(result, batch_indices, batch_dims=1)     # (batch, K*max_output, 10)
        result = result * valid_mask[..., tf.newaxis]               # (batch, K*max_output, 10)
        return result

    def merged_scale(self, pred):
        scales = [key for key in pred if "feature" in key]
        scale_order = cfg.Model.Output.FEATURE_ORDER
        for scale in scales:
            feature_shape = pred[scale]["object"].shape
            ones_map = tf.ones(feature_shape, dtype=tf.float32)
            if scale == scale_order[0]:
                pred[scale].update(self.anchor_indices(feature_shape, ones_map, range(0, 3)))
            elif scale == scale_order[1]:
                pred[scale].update(self.anchor_indices(feature_shape, ones_map, range(3, 6)))
            elif scale == scale_order[2]:
                pred[scale].update(self.anchor_indices(feature_shape, ones_map, range(6, 9)))
        slice_keys = list(pred[scales[0]].keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
            # list of (batch, HWA in scale, dim)
            scaled_preds = [pred[scale_name][key] for scale_name in scales]
            scaled_preds = tf.concat(scaled_preds, axis=1)  # (batch, N, dim)
            merged_pred[key] = scaled_preds

        return merged_pred

    def anchor_indices(self, feat_shape, ones_map, anchor_list):
        batch, hwa, _ = feat_shape
        num_anchor = cfg.Model.Output.NUM_ANCHORS_PER_SCALE
        anchor_index = tf.cast(anchor_list, dtype=tf.float32)[..., tf.newaxis]
        split_anchor_shape = tf.reshape(ones_map, (batch, hwa // num_anchor, num_anchor, 1))

        split_anchor_map = split_anchor_shape * anchor_index
        merge_anchor_map = tf.reshape(split_anchor_map, (batch, hwa, 1))

        return {"anchor_ind": merge_anchor_map}

    def compete_diff_categories(self, nms_res, foo_ctgr, bar_ctgr, iou_thresh, score_thresh):
        """
        :param nms_res: (batch, numbox, 10)
        :return:
        """
        batch, numbox = nms_res.shape[:2]
        boxes = nms_res[..., :4]
        category = nms_res[..., 5:6]
        score = nms_res[..., -1:]

        foo_ctgr = self.category_names.index(foo_ctgr)
        bar_ctgr = self.category_names.index(bar_ctgr)
        boxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(boxes)
        batch_survive_mask = []
        for frame_idx in range(batch):
            foo_mask = tf.cast(category[frame_idx] == foo_ctgr, dtype=tf.float32)
            bar_mask = tf.cast(category[frame_idx] == bar_ctgr, dtype=tf.float32)
            target_mask = foo_mask + bar_mask
            target_score_mask = foo_mask + (bar_mask * 0.9)
            target_boxes = boxes_tlbr[frame_idx] * target_mask
            target_score = score[frame_idx] * target_score_mask

            selected_indices = tf.image.non_max_suppression(
                boxes=target_boxes,
                scores=target_score[:, 0],
                max_output_size=20,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
            )
            if tf.size(selected_indices) != 0:
                selected_onehot = tf.one_hot(selected_indices, depth=numbox, axis=-1)    # (20, numbox)
                survive_mask = 1 - target_mask + tf.reduce_max(selected_onehot, axis=0)[..., tf.newaxis]    # (numbox,)
                batch_survive_mask.append(survive_mask)

        if len(batch_survive_mask) == 0:
            return nms_res

        batch_survive_mask = tf.stack(batch_survive_mask, axis=0)                 # (batch, numbox)
        nms_res = nms_res * batch_survive_mask
        return nms_res
