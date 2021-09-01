import log.util_function as uf

def count_true_positives(grtr, grtr_dontcare, pred, num_ctgr, per_class=False, category_key="category", suffix=""):
    """
    :param grtr: slices of GT object info {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    suffix = suffix if suffix == "" else "_" + suffix
    splits = split_true_false(grtr, pred, grtr_dontcare, category_key)
    # ========== use split instead grtr, pred
    valid_grtr_tp = tf.cast(splits["grtr_tp"]["yxhw"][..., 2:3] > 0, tf.float32)
    valid_grtr_fn = tf.cast(splits["grtr_fn"]["yxhw"][..., 2:3] > 0, tf.float32)
    valid_pred_tp = tf.cast(splits["pred_tp"]["yxhw"][..., 2:3] > 0, tf.float32)
    valid_pred_fp = tf.cast(splits["pred_fp"]["yxhw"][..., 2:3] > 0, tf.float32)

    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], valid_grtr_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], valid_grtr_fn, num_ctgr)
        pred_tp_count = count_per_class(splits["pred_tp"], valid_pred_tp, num_ctgr)
        pred_fp_count = count_per_class(splits["pred_fp"], valid_pred_fp, num_ctgr)

        return {"trpo" + suffix: pred_tp_count.numpy(),
                "grtr" + suffix: (grtr_tp_count + grtr_fn_count).numpy(),
                "pred" + suffix: (pred_tp_count + pred_fp_count).numpy()}
    else:
        grtr_count = tf.reduce_sum(valid_grtr_tp + valid_grtr_fn)
        pred_count = tf.reduce_sum(valid_pred_tp + valid_pred_fp)
        trpo_count = tf.reduce_sum(valid_pred_tp)
        return {"trpo" + suffix: trpo_count.numpy(),
                "grtr" + suffix: grtr_count.numpy(),
                "pred" + suffix: pred_count.numpy()}


def split_true_false(grtr, pred, category_key="category"):
    """
    1. split pred -> valid pred, far pred
    2. far pred vs grtr -> far grtr, valid grtr
    3. valid pred vs valid grtr -> tp pred, tp grtr, fp pred, fn grtr
    4. fp pred vs dc grtr -> dc pred, fp pred
    """
    # valid_pred, far_pred = split_far_pred(pred)
    # far_grtr, valid_grtr = split_far_grtr(far_pred, grtr)
    # split_true_false에서 dontcare 부분만 제거하면 될듯
    splits = split_tp_fp_fn(grtr, pred, category_key)

    fp_pred, dc_pred = split_dontcare_pred(splits["pred_fp"], grtr_dc)
    splits["pred_fp"] = fp_pred
    splits["pred_dc"] = dc_pred
    splits["grtr_dc"] = grtr_dc
    splits["pred_far"] = far_pred
    splits["grtr_far"] = far_grtr
    return splits


def split_far_pred(pred):
    far_pred_mask = tf.cast(pred["distance"] > cfg.Log.DIST_LIM, dtype=tf.float32)
    valid_pred = {key: val * (1 - far_pred_mask) for key, val in pred.items()}
    far_pred = {key: val * far_pred_mask for key, val in pred.items()}
    return valid_pred, far_pred


def split_far_grtr(far_pred, grtr):
    iou_far = uf.compute_iou_general(grtr["yxhw"], far_pred["yxhw"])
    best_iou_far = tf.reduce_max(iou_far, axis=-1, keepdims=True)
    iou_thresh = get_iou_thresh_per_class(grtr["category"])
    # iou_thresh = 0.5
    iou_match = tf.cast(best_iou_far > iou_thresh, dtype=tf.float32)
    valid_grtr = {key: val * (1 - iou_match) for key, val in grtr.items()}
    far_grtr = {key: val * iou_match for key, val in grtr.items()}
    far_grtr["iou"] = best_iou_far
    return far_grtr, valid_grtr


def split_tp_fp_fn(valid_pred, valid_grtr, category_key):
    valid_mask = valid_grtr["object"] # gt (batch , fixed_num , 1)
    batch, M, _ = valid_pred[category_key].shape  # (batch * 512 , 4)
    iou = uf.compute_iou_general(valid_grtr["yxhw"], valid_pred["yxhw"])  # (batch, N, M)
    best_iou = tf.reduce_max(iou, axis=-1, keepdims=True)  # (batch, N)
    best_idx = tf.cast(tf.argmax(iou, axis=-1), dtype=tf.int32)  # (batch, N)

    iou_thresh = get_iou_thresh_per_class(valid_grtr["category"])

    iou_match = tf.cast(best_iou > iou_thresh, dtype=tf.float32)  # (batch, N)

    pred_ctgr_aligned = tf.gather(valid_pred[category_key], best_idx, batch_dims=1)  # (batch, N, 8)

    ctgr_match = tf.cast(valid_grtr[category_key] == pred_ctgr_aligned, dtype=tf.float32)  # (batch, N)

    grtr_tp_mask = (iou_match * ctgr_match)  # (batch, N, 1)
    grtr_fn_mask = (tf.convert_to_tensor(1, dtype=tf.float32) - grtr_tp_mask) * valid_mask  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in valid_grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in valid_grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask
    grtr_fn["iou"] = best_iou * grtr_fn_mask
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    best_idx_onehot = tf.one_hot(best_idx, depth=M) * grtr_tp_mask  # (batch, N, M)
    pred_tp_mask = tf.reduce_max(best_idx_onehot, axis=1)[..., tf.newaxis]  # (batch, M, 1)
    pred_fp_mask = tf.convert_to_tensor(1, dtype=tf.float32) - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in valid_pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in valid_pred.items()}

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_dontcare_pred(pred_fp, grtr_dc):
    B, M, _ = pred_fp["category"].shape
    iou_dc = uf.compute_iou_general(grtr_dc["yxhw"], pred_fp["yxhw"])
    best_iou_dc = tf.reduce_max(iou_dc, axis=-1, keepdims=True)  # (batch, D)
    best_iou_dc_idx = tf.argmax(iou_dc, axis=-1)
    # iou_thresh = get_iou_thresh_per_class(pred_fp, grtr_dc, best_iou_dc_idx)
    iou_thresh = 0.5
    grtr_dc["iou"] = best_iou_dc
    dc_match = tf.cast(best_iou_dc > iou_thresh, dtype=tf.float32)  # (batch, D)
    best_idx_dc = tf.cast(tf.argmax(iou_dc, axis=-1), dtype=tf.int32)
    best_idx_onehot_dc = tf.one_hot(best_idx_dc, depth=M) * dc_match  # (batch, D, M)
    pred_dc_mask = tf.reduce_max(best_idx_onehot_dc, axis=1)[..., tf.newaxis]  # (batch, M, 1)
    dc_pred = {key: val * pred_dc_mask for key, val in pred_fp.items()}
    fp_pred = {key: val * (1 - pred_dc_mask) for key, val in pred_fp.items()}
    return fp_pred, dc_pred


def get_iou_thresh_per_class(grtr_ctgr):
    ctgr_idx = tf.cast(grtr_ctgr, dtype=tf.int32)
    tp_iou_thresh = tf.convert_to_tensor(cfg.Log.TP_IOU_THRESH, tf.float32)
    iou_thresh = tf.gather(tp_iou_thresh, ctgr_idx)
    return iou_thresh


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = tf.cast(boxes["category"][..., 0], dtype=tf.int32)  # (batch, N')
    boxes_onehot = tf.one_hot(boxes_ctgr, depth=num_ctgr) * mask  # (batch, N', K)
    boxes_count = tf.reduce_sum(boxes_onehot, axis=[0, 1])
    return boxes_count


import numpy as np


def test_count_true_positives():
    print("===== start count_true_positives")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N, K = 4, 10, 8  # N: num of TP
    NF = 20             # NF: num of false data

    for i in range(100):
        # create similar grtr and pred boxes that have iou > 0.5
        # preventing zero height or width
        grtr_tp_boxes = np.tile(np.random.uniform(0, 1, (1, N, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_tp_boxes[..., 0] = np.linspace(0, 0.3*(N-1), N).reshape((1, N)) + 0.1
        grtr_tp_boxes[..., 4] = np.random.randint(0, K, N).reshape((1, N))    # set category indices
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
        grtr_fn_boxes[..., 4] = np.random.randint(0, K, NF).reshape((1, NF))    # set category indices
        pred_fp_boxes = grtr_fn_boxes.copy()
        pred_fp_boxes[:, :5, :2] += 2           # zero iou
        pred_fp_boxes[:, 5:10, 4] = (pred_fp_boxes[:, 5:10, 4] + 1) % K         # different category
        pred_fp_boxes[:, 10:15, :] = 0          # zero pred box
        grtr_fn_boxes[:, 15:20, :] = 0          # zero gt box

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


def test_get_iou_thresh():
    print("TEST test_get_iou_thresh")
    grtr_ctgr = tf.convert_to_tensor([[1, 2, 3, 4], [0, 4, 3, 2]], dtype=tf.int32)
    iou_thresh = tf.gather(cfg.Log.TP_IOU_THRESH, grtr_ctgr)
    print("IOU_THRESH in categor order:", cfg.Log.TP_IOU_THRESH)
    print("gather output:", iou_thresh)
    print("by numpy indexing0", np.array(cfg.Log.TP_IOU_THRESH)[grtr_ctgr[0].numpy()])
    print("by numpy indexing1", np.array(cfg.Log.TP_IOU_THRESH)[grtr_ctgr[1].numpy()])


if __name__ == "__main__":
    # test_count_true_positives()
    test_get_iou_thresh()

