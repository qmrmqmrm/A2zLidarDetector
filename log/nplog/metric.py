import torch
import numpy as np
import utils.util_function as uf
import model.submodules.model_util as mu
import config as cfg
import parameter_pool as pp


def count_true_positives(grtr, pred, num_ctgr, iou_thresh=cfg.Validation.TP_IOU_THRESH, per_class=False):
    """
    :param grtr: slices of features["bboxes"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param grtr_dontcare: slices of features["dontcare"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    splits = split_true_false(grtr, pred, iou_thresh)
    # ========== use split instead grtr, pred
    grtr_valid_tp = splits["grtr_tp"]["bbox2d"][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"]["bbox2d"][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"]["bbox2d"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["bbox2d"][..., 2:3] > 0
    if per_class:
        print("grtr_tp")
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        print("grtr_fn")
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        print("pred_tp")
        pred_tp_count = count_per_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        print("pred_fp")
        pred_fp_count = count_per_class(splits["pred_fp"], pred_valid_fp, num_ctgr)

        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:

        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)

        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def split_true_false(grtr, pred, iou_thresh):

    splits = split_tp_fp_fn(grtr, pred, iou_thresh)
    return splits


def split_tp_fp_fn(grtr, pred, iou_thresh):
    batch, M, _ = pred["category"].shape
    # best_cate = np.argmax(pred["category"], axis=-1)
    # pred["category"] = np.expand_dims(best_cate,-1)

    valid_mask = grtr["object"]
    # iou = uf.pairwise_batch_iou(grtr["bbox2d"], pred["rpn_bbox2d"])  # (batch, N, M)
    iou = uf.compute_iou_general(grtr["bbox2d"], pred["bbox2d"], grtr_tlbr=True, pred_tlbr=True)  # (batch, N, M)
    best_iou = np.max(iou, axis=-1)  # (batch, N)
    print('best_iou', best_iou)
    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh) # (batch, 15,1) iou_tresh len : 3
    iou_match = best_iou > iou_thresh  # (batch, N)
    print('iou_match', iou_match.shape)
    print('iou_match', iou_match)
    pred_ctgr_aligned = numpy_gather(pred["category"], best_idx, 1)  # (batch, N, 8)
    print('pred_ctgr_aligned', pred_ctgr_aligned)
    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    print('ctgr_match', ctgr_match)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)
    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items() if key in pp.LossComb.BIRDNET}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items() if key in pp.LossComb.BIRDNET}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    # pred_loss_comb = ["objectness", "bbox2d", "category", "bbox3d", "yaw", "yaw_rads"]

    pred_tp = {key: val * pred_tp_mask for key, val in pred.items() if key in pp.LossComb.BIRDNET}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items() if key in pp.LossComb.BIRDNET}

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def indices_to_binary_mask(best_idx, valid_mask, depth):
    best_idx_onehot = one_hot(best_idx, depth) * valid_mask
    binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
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
    print('boxes_ctgr',  boxes_ctgr.shape)
    print('mask',  mask.shape)
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
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

