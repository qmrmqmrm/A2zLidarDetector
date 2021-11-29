import torch
import numpy as np
import utils.util_function as uf
import model.submodules.model_util as mu
import config as cfg
import parameter_pool as pp


def count_true_positives(grtr, pred, num_ctgr, estimator, tp_iou=cfg.Validation.TP_IOU_THRESH, per_class=False):
    """
    :param grtr: slices of features["bboxes"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param grtr_dontcare: slices of features["dontcare"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    split_key = estimator.split_key
    splits = TrueFalseSplitter(estimator, tp_iou)(grtr, pred)
    # ========== use split instead grtr, pred
    print(split_key)
    print(splits["grtr_tp"][split_key].shape)
    grtr_valid_tp = splits["grtr_tp"][split_key][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"][split_key][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"][split_key][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"][split_key][..., 2:3] > 0

    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_class_pred(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_class_pred(splits["pred_fp"], pred_valid_fp, num_ctgr)
        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def indices_to_binary_mask(indices, mask, max_ind):
    """
    indices: (batch, M)
    mask: (batch, M, 1)
    depth: max of indices + 1
    """
    sel_one_hot = one_hot(indices, max_ind) * mask  # (batch, M, D)
    sel_mask = np.max(sel_one_hot, axis=1)  # (batch, D)
    binary_mask = np.expand_dims(sel_mask, axis=-1)  # (batch, D, 1)
    return binary_mask.astype(np.float32)


def get_iou_thresh_per_class(grtr_ctgr, tp_iou_thresh):
    ctgr_idx = grtr_ctgr.astype(np.int32)
    tp_iou_thresh = np.asarray(tp_iou_thresh, np.float32)
    iou_thresh = numpy_gather(tp_iou_thresh, ctgr_idx)
    return iou_thresh[..., 0]


def count_per_class(boxes, mask, num_ctgr):
    # TODO check numpy test
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    # boxes_ctgr = tf.cast(boxes["category"][..., 0], dtype=tf.int32)  # (batch, N')
    # boxes_onehot = tf.one_hot(boxes_ctgr, depth=num_ctgr) * mask  # (batch, N', K)
    # boxes_count = tf.reduce_sum(boxes_onehot, axis=[0, 1])
    boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr + 1) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def count_per_class_pred(boxes, mask, num_ctgr):
    # TODO check numpy test
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    # boxes_ctgr = tf.cast(boxes["category"][..., 0], dtype=tf.int32)  # (batch, N')
    # boxes_onehot = tf.one_hot(boxes_ctgr, depth=num_ctgr) * mask  # (batch, N', K)
    # boxes_count = tf.reduce_sum(boxes_onehot, axis=[0, 1])
    best_inds = np.argmax(boxes["ctgr_probs"], axis=-1)  # (batch, N')
    boxes_onehot = one_hot(best_inds, num_ctgr + 1) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def one_hot(grtr_category, category_shape):
    one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
    return one_hot_data


def numpy_gather(params, index, dim=0):
    if dim == 1:
        batch_list = []
        for i in range(params.shape[0]):
            batch_param = params[i]
            batch_index = index[i]
            batch_gather = np.take(batch_param, batch_index)
            batch_list.append(batch_gather)
        gathar_param = np.stack(batch_list)
    else:
        gathar_param = np.take(params, index)
    return gathar_param


def test_indices_to_binary_mask():
    indices = np.array([[1, 3, 5], [0, 2, 4]])
    valid = np.expand_dims(np.array([[1, 1, 0], [0, 1, 1]]), axis=-1)
    depth = 6
    mask = indices_to_binary_mask(indices, valid, depth)
    answer = np.array([[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]])
    assert mask[..., 0] == answer


class TrueFalseSplitter:
    def __init__(self, estimator, tp_iou=cfg.Validation.TP_IOU_THRESH):
        self.estimator = estimator
        self.tp_iou = tp_iou

    def __call__(self, grtr, pred):
        batch, M, num_ctgr = pred["ctgr_probs"].shape
        pred_ctgr = np.argmax(pred["ctgr_probs"], axis=-1)
        pred_ctgr = np.expand_dims(pred_ctgr, -1)
        pred_object_mask = (pred_ctgr > 0)
        grtr_object_mask = grtr["object"]
        grtr_ctgr = grtr["category"]

        tpfn_masks = {"pred_tp": 0, "pred_fp": 0, "grtr_tp": 0, "grtr_fn": 0}
        grtr_ious = {"grtr_tp": 0, "grtr_fn": 0}
        for ctgr in range(1, num_ctgr):
            pred_mask = pred_object_mask * (pred_ctgr == ctgr)
            grtr_mask = grtr_object_mask * (grtr_ctgr == ctgr)
            tpfn_masks_ctgr, grtr_iou = self.split_per_category(grtr, grtr_mask, pred, pred_mask, self.tp_iou)
            for mask_key in tpfn_masks:
                tpfn_masks[mask_key] += tpfn_masks_ctgr[mask_key]
            for iou_key in grtr_ious:
                grtr_ious[iou_key] += grtr_iou[iou_key]

        gt_keys = ['category', 'yaw_cls', 'bbox2d', 'bbox3d', 'object', 'yaw_rads', 'anchor_id', ]
        grtr_tp = {key: val * tpfn_masks["grtr_tp"] for key, val in grtr.items() if key in gt_keys}
        grtr_fn = {key: val * tpfn_masks["grtr_fn"] for key, val in grtr.items() if key in gt_keys}
        grtr_tp["iou"] = grtr_ious['grtr_tp']
        grtr_fn["iou"] = grtr_ious['grtr_fn']

        pred_tp = {key: val * tpfn_masks["pred_tp"] for key, val in pred.items()}
        pred_fp = {key: val * tpfn_masks["pred_fp"] for key, val in pred.items()}
        return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}

    def split_per_category(self, grtr, grtr_mask, pred, pred_mask, iou_thresh):
        pred_ctgr = np.argmax(pred["ctgr_probs"], axis=-1)
        pred_ctgr = np.expand_dims(pred_ctgr, -1) * pred_mask
        iou = self.estimator(grtr, grtr_mask, pred, pred_mask)
        best_iou = np.max(iou, axis=-1)  # (batch, N)
        best_idx = np.argmax(iou, axis=-1)  # (batch, N)
        if len(iou_thresh) > 1:
            iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)  # (batch, 15,1) iou_tresh len : 3
        iou_match = np.expand_dims(best_iou >= iou_thresh, -1)
        grtr_tp_mask = iou_match * grtr_mask
        grtr_fn_mask = (1 - iou_match) * grtr_mask

        pred_tp_mask = indices_to_binary_mask(best_idx, iou_match, pred_ctgr.shape[1])
        pred_tp_mask = pred_tp_mask * pred_mask
        pred_fp_mask = (1 - pred_tp_mask) * pred_mask

        best_iou = np.expand_dims(best_iou, -1)
        masks = {"grtr_tp": grtr_tp_mask, "grtr_fn": grtr_fn_mask, "pred_tp": pred_tp_mask, "pred_fp": pred_fp_mask}
        ious = {"grtr_tp": best_iou * grtr_tp_mask, "grtr_fn": best_iou * grtr_fn_mask}
        return masks, ious


class IouEstimator:
    def __init__(self):
        self.split_key = "bbox2d"

    def __call__(self, grtr, grtr_mask, pred, pred_mask):
        grtr_bbox = grtr["bbox2d"] * grtr_mask
        pred_bbox = pred["bbox2d"] * pred_mask
        iou = uf.compute_iou_general(grtr_bbox, pred_bbox)  # (batch, N, M)
        return iou


class RotatedIouEstimator:
    def __init__(self):
        self.split_key = "bbox3d"

    def __call__(self, grtr, grtr_mask, pred, pred_mask):
        grtr_bbox = grtr["bbox3d"] * grtr_mask
        pred_bbox = pred["bbox3d"] * pred_mask
        grtr_rad = (grtr['yaw_rads'] * 180 / np.pi) * grtr_mask
        pred_rad = (pred['yaw_rads'] * 180 / np.pi) * pred_mask
        rotated_ious = list()
        for frame in range(grtr_bbox.shape[0]):
            img_shape = grtr['image'][frame].shape
            rotated_iou = uf.rotated_iou_per_frame(grtr_bbox[frame], pred_bbox[frame], grtr_rad[frame],
                                                   pred_rad[frame], img_shape)  # (N, M)
            rotated_ious.append(rotated_iou)
        rotated_ious = np.stack(rotated_ious, axis=0)
        return rotated_ious


if __name__ == "__main__":
    test_indices_to_binary_mask()
