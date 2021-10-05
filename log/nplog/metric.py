import torch
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
    pred_valid_tp = splits["pred_tp"]["rpn_bbox2d"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["rpn_bbox2d"][..., 2:3] > 0
    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_class(splits["pred_fp"], pred_valid_fp, num_ctgr)

        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:

        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def split_true_false(grtr, pred, iou_thresh):
    # pred_valid, pred_far = split_pred_far(pred)
    # grtr_far, grtr_valid = split_grtr_far(pred_far, grtr, iou_thresh)
    splits = split_tp_fp_fn(grtr, pred, iou_thresh)
    # fp_pred, dc_pred = split_dontcare_pred(splits["pred_fp"])
    # splits["pred_fp"] = fp_pred
    # splits["pred_dc"] = dc_pred
    # splits["grtr_dc"] = grtr_dc
    # splits["pred_far"] = pred_far
    # splits["grtr_far"] = grtr_far
    return splits


def split_tp_fp_fn(grtr, pred, iou_thresh):
    batch, M, _ = pred["category"].shape
    valid_mask = grtr["object"]
    # iou = uf.pairwise_batch_iou(grtr["bbox2d"], pred["rpn_bbox2d"])  # (batch, N, M)
    iou = uf.compute_iou_general(grtr["bbox2d"], pred["rpn_bbox2d"], grtr_tlbr=True, pred_tlbr=True)  # (batch, N, M)

    best_iou = np.max(iou, axis=-1)  # (batch, N)

    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh) # (batch, 15,1) iou_tresh len : 3
    iou_match = best_iou > iou_thresh  # (batch, N)
    pred_ctgr_aligned = numpy_gather(pred["category"], best_idx, 1)  # (batch, N, 8)
    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)
    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items() if key in pp.LossComb.BIRDNET}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items() if key in pp.LossComb.BIRDNET}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_loss_comb = ["rpn_objectness", "rpn_bbox2d", "category", "bbox3d", "yaw", "yaw_rads"]
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items() if key in pred_loss_comb}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items() if key in pred_loss_comb}

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_dontcare_pred(pred_fp, grtr_dc):
    B, M, _ = pred_fp["category"].shape
    iou_dc = uf.compute_iou_general(grtr_dc["yxhw"], pred_fp["yxhw"])
    best_iou_dc = np.max(iou_dc, axis=-1)  # (batch, D)
    grtr_dc["iou"] = best_iou_dc
    dc_match = np.expand_dims(best_iou_dc > 0.5, axis=-1)  # (batch, D)
    best_idx_dc = np.argmax(iou_dc, axis=-1)
    pred_dc_mask = indices_to_binary_mask(best_idx_dc, dc_match, M)  # (batch, M, 1)
    dc_pred = {key: val * pred_dc_mask for key, val in pred_fp.items()}
    fp_pred = {key: val * (1 - pred_dc_mask) for key, val in pred_fp.items()}
    return fp_pred, dc_pred


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


import numpy as np


def test_count_true_positives():
    print("===== start count_true_positives")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N, K = 4, 10, 8  # N: num of TP
    NF = 20  # NF: num of false data

    for i in range(100):
        # create similar grtr and pred boxes that have iou > 0.5
        # preventing zero height or width
        grtr_tp_boxes = np.tile(np.random.uniform(0, 1, (1, N, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_tp_boxes[..., 0] = np.linspace(0, 0.3 * (N - 1), N).reshape((1, N)) + 0.1
        grtr_tp_boxes[..., 4] = np.random.randint(0, K, N).reshape((1, N))  # set category indices
        pred_tp_boxes = grtr_tp_boxes.copy()
        # add small noise to boxes
        pred_tp_boxes[..., :4] += np.tile(grtr_tp_boxes[..., 2:4], 2) * (np.random.uniform(0, 1, (B, N, 4)) - 0.5) * 0.1
        grtr_tp_boxes = tf.convert_to_tensor(grtr_tp_boxes[..., :5])
        pred_tp_boxes = tf.convert_to_tensor(pred_tp_boxes[..., :8])
        iou = uf.compute_iou_general(grtr_tp_boxes, pred_tp_boxes)
        if (tf.linalg.diag_part(iou).numpy() <= 0.5).any():
            print("grtr vs pred (aligned) iou:", tf.linalg.diag_part(iou).numpy())
            print("grtr_boxes", grtr_tp_boxes[0])
            print("pred_boxes", pred_tp_boxes[0])
        assert (tf.linalg.diag_part(iou).numpy() > 0.5).all()

        # create different grtr and pred boxes
        grtr_fn_boxes = np.tile(np.random.uniform(0, 1, (1, NF, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_fn_boxes[..., 0] = np.linspace(0, 0.3 * (NF - 1), NF).reshape((1, NF)) + 5.1
        grtr_fn_boxes[..., 4] = np.random.randint(0, K, NF).reshape((1, NF))  # set category indices
        pred_fp_boxes = grtr_fn_boxes.copy()
        pred_fp_boxes[:, :5, :2] += 2  # zero iou
        pred_fp_boxes[:, 5:10, 4] = (pred_fp_boxes[:, 5:10, 4] + 1) % K  # different category
        pred_fp_boxes[:, 10:15, :] = 0  # zero pred box
        grtr_fn_boxes[:, 15:20, :] = 0  # zero gt box

        # grtr_boxes, pred_boxes: N similar boxes, NF different boxes
        grtr_boxes = tf.cast(tf.concat([grtr_tp_boxes, grtr_fn_boxes[..., :5]], axis=1), dtype=tf.float32)
        pred_boxes = tf.cast(tf.concat([pred_tp_boxes, pred_fp_boxes[..., :8]], axis=1), dtype=tf.float32)
        grtr_boxes = uf.slice_bbox(grtr_boxes, True)
        pred_boxes = uf.slice_bbox(pred_boxes, False)

        # EXECUTE
        result = count_true_positives(grtr_boxes, pred_boxes, K)
        # true positive is supposed to be 40
        print("result", result)
        assert result["trpo"] == B * N
        assert result["grtr"] == B * (N + NF - 5)
        assert result["pred"] == B * (N + NF - 5)

    print("!!! pass test_count_true_positives")


if __name__ == "__main__":
    test_count_true_positives()
