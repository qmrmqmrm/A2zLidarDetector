

def split_true_false(grtr, pred, iou_thresh):
    batch, M, num_ctgr = pred["category"].shape
    pred_ctgr = np.argmax(pred["category"], axis=-1)
    pred_ctgr = np.expand_dims(pred_ctgr, -1)
    pred_object_mask = (pred_ctgr > 0)
    grtr_object_mask = grtr["object"]
    grtr_ctgr = grtr["category"]

    tpfn_masks = {"pred_tp": 0, "pred_fp": 0, "grtr_tp": 0, "grtr_fn": 0}
    grtr_ious = {"grtr_tp": 0, "grtr_fn": 0}
    for ctgr in range(1, num_ctgr):
        pred_mask = pred_object_mask * (pred_ctgr == ctgr)
        grtr_mask = grtr_object_mask * (grtr_ctgr == ctgr)
        tpfn_masks_ctgr, grtr_iou = split_per_category(grtr, grtr_mask, pred, pred_mask, iou_thresh)
        for mask_key in tpfn_masks:
            tpfn_masks[mask_key] += tpfn_masks_ctgr[mask_key]
        for iou_key in grtr_ious:
            grtr_ious[iou_key] += grtr_iou[iou_key]

    # mask_counts = {key: np.sum(mask) for key, mask in tpfn_masks.items()}
    # mask_counts["grtr"] = mask_counts["grtr_tp"] + mask_counts["grtr_fn"]
    # mask_counts["pred"] = mask_counts["pred_tp"] + mask_counts["pred_fp"]
    # mask_counts = {key: int(val) for key, val in mask_counts.items()}
    # print("mask counts:", mask_counts)

    gt_keys = ['category', 'yaw_cls', 'bbox2d', 'bbox3d', 'object', 'yaw_rads', 'anchor_id', ]
    grtr_tp = {key: val * tpfn_masks["grtr_tp"] for key, val in grtr.items() if key in gt_keys}
    grtr_fn = {key: val * tpfn_masks["grtr_fn"] for key, val in grtr.items() if key in gt_keys}
    grtr_tp["iou"] = grtr_ious['grtr_tp']
    grtr_fn["iou"] = grtr_ious['grtr_fn']

    pred_tp = {key: val * tpfn_masks["pred_tp"] for key, val in pred.items()}
    pred_fp = {key: val * tpfn_masks["pred_fp"] for key, val in pred.items()}
    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_per_category(grtr, grtr_mask, pred, pred_mask, iou_thresh):
    grtr_bbox = grtr["bbox2d"] * grtr_mask
    pred_bbox = pred["bbox2d"] * pred_mask
    pred_ctgr = np.argmax(pred["category"], axis=-1)
    pred_ctgr = np.expand_dims(pred_ctgr, -1) * pred_mask
    iou = uf.compute_iou_general(grtr_bbox, pred_bbox)  # (batch, N, M)
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


def split_rotated_true_false(grtr, pred, iou_thresh):
    batch, M, num_ctgr = pred["category"].shape
    pred_ctgr = np.argmax(pred["category"], axis=-1)
    pred_ctgr = np.expand_dims(pred_ctgr, -1)
    pred_object_mask = (pred_ctgr > 0)
    grtr_object_mask = grtr["object"]
    grtr_ctgr = grtr["category"]

    tpfn_masks = {"pred_tp": 0, "pred_fp": 0, "grtr_tp": 0, "grtr_fn": 0}
    grtr_ious = {"grtr_tp": 0, "grtr_fn": 0}
    for ctgr in range(1, num_ctgr):
        pred_mask = pred_object_mask * (pred_ctgr == ctgr)
        grtr_mask = grtr_object_mask * (grtr_ctgr == ctgr)
        tpfn_masks_ctgr, grtr_iou = split_rotated_per_category(grtr, grtr_mask, pred, pred_mask, iou_thresh)
        for mask_key in tpfn_masks:
            tpfn_masks[mask_key] += tpfn_masks_ctgr[mask_key]
        for iou_key in grtr_ious:
            grtr_ious[iou_key] += grtr_iou[iou_key]

    # mask_counts = {key: np.sum(mask) for key, mask in tpfn_masks.items()}
    # mask_counts["grtr"] = mask_counts["grtr_tp"] + mask_counts["grtr_fn"]
    # mask_counts["pred"] = mask_counts["pred_tp"] + mask_counts["pred_fp"]
    # mask_counts = {key: int(val) for key, val in mask_counts.items()}
    # print("mask counts:", mask_counts)

    gt_keys = ['category', 'yaw_cls', 'bbox2d', 'bbox3d', 'object', 'yaw_rads', 'anchor_id', ]
    grtr_tp = {key: val * tpfn_masks["grtr_tp"] for key, val in grtr.items() if key in gt_keys}
    grtr_fn = {key: val * tpfn_masks["grtr_fn"] for key, val in grtr.items() if key in gt_keys}
    grtr_tp["iou"] = grtr_ious['grtr_tp']
    grtr_fn["iou"] = grtr_ious['grtr_fn']

    pred_tp = {key: val * tpfn_masks["pred_tp"] for key, val in pred.items()}
    pred_fp = {key: val * tpfn_masks["pred_fp"] for key, val in pred.items()}
    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_rotated_per_category(grtr, grtr_mask, pred, pred_mask, iou_thresh):
    grtr_bbox = grtr["bbox3d"] * grtr_mask
    pred_bbox = pred["bbox3d"] * pred_mask
    grtr_rad = (grtr['yaw_rads'] * 180 / np.pi) * grtr_mask
    pred_rad = (pred['yaw_rads'] * 180 / np.pi) * pred_mask
    pred_ctgr = np.argmax(pred["category"], axis=-1)
    pred_ctgr = np.expand_dims(pred_ctgr, -1) * pred_mask
    rotated_ious = list()
    for frame in range(grtr_bbox.shape[0]):
        img_shape = grtr['image'][frame].shape
        rotated_iou = uf.rotated_iou_per_frame(grtr_bbox[frame], pred_bbox[frame], grtr_rad[frame], pred_rad[frame],
                                               img_shape)  # (N, M)
        rotated_ious.append(rotated_iou)
    rotated_ious = np.stack(rotated_ious, axis=0)
    best_iou = np.max(rotated_ious, axis=-1)  # (batch, N)
    best_idx = np.argmax(rotated_ious, axis=-1)  # (batch, N)
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