import os
import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import torch
import math

import settings
import config as cfg
from model.model_factory import ModelFactory
from dataloader.loader_factory import get_dataset
import utils.util_function as uf
import model.submodules.model_util as mu
from log.nplog.metric import count_true_positives


def optimize_nms_params():
    dataset_name = "a2d2"
    valid_category = cfg.get_valid_category_mask(dataset_name)
    ckpt_path = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    test_data_loader, model = load_dataset_model(dataset_name, ckpt_path)
    print('optimize_nms_params')
    perf_data = collect_recall_precision(test_data_loader, model, valid_category)
    save_results(perf_data, ckpt_path)


def load_dataset_model(dataset_name, ckpt_path):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    tfrd_path = cfg.Paths.TFRECORD
    valid_category = cfg.get_valid_category_mask(dataset_name)

    test_data_loader = get_dataset(dataset_name, 'test', batch_size, False)
    model_factory = ModelFactory(dataset_name)
    model = model_factory.make_model()
    model = try_load_weights(ckpt_path, model)
    return test_data_loader, model


# def get_dataset(tfrd_path, dataset_name, shuffle, batch_size, split):
#     tfrpath = op.join(tfrd_path, f"{dataset_name}_{split}")
#     reader = TfrecordReader(tfrpath, shuffle, batch_size, 1)
#     dataset = reader.get_dataset()
#     frames = reader.get_total_frames()
#     tfr_cfg = reader.get_tfr_config()
#     image_shape = tfr_cfg["image"]["shape"]
#     # anchor sizes per scale in pixel
#     anchors_per_scale = {key: np.array(val) / np.array([image_shape[:2]]) for key, val in tfr_cfg.items() if key.startswith("anchor")}
#     anchors_per_scale = {key: val.astype(np.float32) for key, val in anchors_per_scale.items()}
#     print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
#           f"frames={frames},\n\tanchors={anchors_per_scale}")
#     return dataset, frames // batch_size, image_shape, anchors_per_scale


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model.load_state_dict(torch.load(ckpt_file))
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def collect_recall_precision(dataset, model, valid_category):
    results = {"max_box": [], "iou_thresh": [], "score_thresh": []}
    ap_results = {"max_box": [], "iou_thresh": [], "score_thresh": []}
    for max_box in cfg.NMS.MAX_OUT_CANDIDATES:
        for iou_thresh in cfg.NMS.IOU_CANDIDATES:
            for score_thresh in cfg.NMS.SCORE_CANDIDATES[::-1]:
                results["max_box"].append(max_box)
                results["iou_thresh"].append(iou_thresh)
                results["score_thresh"].append(score_thresh)

    results = {key: np.array(val) for key, val in results.items()}
    num_params = results["max_box"].shape[0]

    num_categories = len(valid_category)
    accum_keys = ["trpo", "grtr", "pred"]
    init_data = {key: np.zeros((num_params, num_categories), dtype=np.float32) for key in accum_keys}
    results.update(init_data)
    nms = mu.NonMaximumSuppression()
    train_loader_iter = iter(dataset)
    steps = len(train_loader_iter)
    for step in range(steps):
        start = timer()
        features = next(train_loader_iter)
        features = to_device(features)
        # grtr_slices = uf.merge_and_slice_features(grtr, True)
        grtr_slices = convert_tensor_to_numpy(features)
        pred = model(features)
        # pred_slices = uf.merge_and_slice_features(pred, False)
        pred = select_best_ctgr_pred(pred)
        for i in range(num_params):
            max_box = np.ones((num_categories,), dtype=np.int64) * results["max_box"][i]
            iou_thresh = np.ones((num_categories,), dtype=np.float32) * results["iou_thresh"][i]
            score_thresh = np.ones((num_categories,), dtype=np.float32) * results["score_thresh"][i]
            pred_bboxes = nms(pred, max_box, iou_thresh, score_thresh)
            # pred_bboxes = uf.slice_feature(pred_bboxes, cfg.Model.Output.get_bbox_composition(False))
            pred_bboxes = convert_tensor_to_numpy(pred_bboxes)
            count_per_class = count_true_positives(grtr_slices, pred_bboxes,
                                                   num_categories, iou_thresh=cfg.Validation.MAP_TP_IOU_THRESH,
                                                   per_class=True)
            for key in accum_keys:
                results[key][i] += count_per_class[key][1:]
        uf.print_progress(f"=== step: {step}/{steps}, took {timer() - start:1.2f}s")
        # if step > 50:
        #     break
    results["recall"] = np.divide(results["trpo"], results["grtr"], out=np.zeros_like(results["trpo"]),
                                  where=(results["grtr"] != 0))
    results["precision"] = np.divide(results["trpo"], results["pred"], out=np.zeros_like(results["trpo"]),
                                     where=(results["pred"] != 0))
    results["min_perf"] = np.minimum(results["recall"], results["precision"])
    results["avg_perf"] = (results["recall"] + results["precision"]) / 2.
    for key, val in results.items():
        print(f"results: {key}\n{val[:10]}")
    return results


def to_device(features):
    device = cfg.Hardware.DEVICE
    for key in features:
        if isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(device=device)
        if isinstance(features[key], list):
            data = list()
            for feature in features[key]:
                if isinstance(feature, torch.Tensor):
                    feature = feature.to(device=device)
                data.append(feature)
            features[key] = data
    return features


def select_best_ctgr_pred(pred):
    best_ctgr = torch.argmax(pred['ctgr_probs'], dim=-1).unsqueeze(-1)
    need_key = ['bbox3d', 'yaw_cls_probs', 'yaw_cls_logit', 'yaw_res', 'bbox3d_delta']
    select_pred = uf.select_category(pred, best_ctgr, need_key)
    best_yaw_cls_idx = torch.argmax(select_pred['yaw_cls_probs'], dim=-1).unsqueeze(-1)
    select_pred['yaw_res'] = torch.gather(select_pred['yaw_res'], dim=-1,
                                          index=best_yaw_cls_idx)
    select_pred['yaw_rads'] = (best_yaw_cls_idx * (math.pi / cfg.Model.Structure.VP_BINS) - (math.pi / 2) +
                               select_pred['yaw_res'])
    return select_pred


def save_results(perf_data, ckpt_path):
    param_path = op.join(ckpt_path, "nms_param")
    os.makedirs(param_path, exist_ok=True)
    intkeys = ["trpo", "grtr", "pred"]
    # summary = summarize_data(perf_data)
    specific_summary, param_summary, ap_summary, mean_ap_summary = specific_data(perf_data)
    for key, data in perf_data.items():
        filename = op.join(param_path, key + ".txt")
        if key in intkeys:
            np.savetxt(filename, data, fmt="%1d")
        else:
            np.savetxt(filename, data, fmt="%1.4f")
    np.savez(op.join(ckpt_path, "nms_param_perf_data.csv"), **perf_data)
    specific_summary.to_csv(op.join(param_path, "specific_summary.csv"), index=False, float_format="%1.4f")
    param_summary.to_csv(op.join(param_path, "nms_param_best.csv"), index=False, float_format="%1.4f")
    ap_summary.to_csv(op.join(param_path, "ap_summary.csv"), index=False, float_format="%1.4f")
    mean_ap_summary.to_csv(op.join(param_path, "mean_ap_summary.csv"), index=False, float_format="%1.4f")


def specific_data(data):
    num_categories = cfg.Model.Structure.NUM_CLASSES
    specific_summary = dict()
    class_order_data = {"class": np.tile(np.arange(0, num_categories, 1), data["score_thresh"].shape[0])}
    thresh_dict, re_pr_dict = change_data_form(data, num_categories)
    for update_data in [thresh_dict, class_order_data, re_pr_dict]:
        specific_summary.update(update_data)
    columns = specific_summary.keys()
    specific_values = np.stack(specific_summary.values(), axis=0).transpose()
    specific_summary = pd.DataFrame(specific_values, columns=columns)
    param_summary = param_summarize(specific_summary, num_categories)
    ap_summary, mean_ap_summary = ap_data(data, num_categories)

    return specific_summary, param_summary, ap_summary, mean_ap_summary


def param_summarize(data, num_categories):
    params = pd.DataFrame()
    for n_class in range(num_categories):
        class_data = data.loc[data["class"] == n_class]
        max_index = np.argmax(class_data["min_perf"])
        select_param = class_data.iloc[max_index]
        params = params.append(select_param, ignore_index=True)
    del params["min_perf"]
    return params


def ap_data(data, num_categories):

    class_ap, mean_ap, iou_thresh, max_box = compute_mAP(data)
    # class ap : iou * max * cls, mean_ap : iou * max
    # [0 1 2 0 1 2 0 1 2].num == max.num
    ap_suammry = {"max_box": np.repeat(max_box, num_categories),
                  "iou_thresh": np.repeat(iou_thresh, num_categories),
                  "class": np.tile(np.arange(0, num_categories, 1), class_ap.shape[0]//num_categories),
                  "class_ap": class_ap}

    # mean_ap_summary = extract_need_keys(data, ["max_box", "iou_thresh"], num_categories, True)
    # ap_summary = extract_need_keys(data, ["max_box", "iou_thresh"], num_categories)
    mean_ap_summary = {"max_box": max_box, "iou_thresh": iou_thresh, "mean_ap": mean_ap}

    ap_df = make_df(ap_suammry)
    mean_ap_df = make_df(mean_ap_summary)

    return ap_df, mean_ap_df


def extract_need_keys(data, need_key, num_categories, adjust_number=False):
    summary = dict()
    for key, value in data.items():
        if key in need_key:
            summary[key] = value
    for key in summary.keys():
        append_data = []
        for idx, value in enumerate(summary[key]):
            if adjust_number:
                if idx % num_categories == 0:
                    append_data.append(value)
            else:
                append_data.append(value)
        summary[key] = np.asarray(append_data, dtype=np.float32)
    return summary


def make_df(summary):
    columns = summary.keys()
    # max_out 220
    values = np.stack(summary.values(), axis=0).transpose()
    summary_df = pd.DataFrame(values, columns=columns)
    return summary_df


def change_data_form(data, num_categories):
    dim_one_data = dict()
    dim_two_data = dict()
    need_key = ["max_box", "iou_thresh", "score_thresh", "recall", "precision", "average_prec", "mean_ap", "min_perf"]
    for key, data in data.items():
        if data.ndim == 1:
            if key in need_key:
                dim_one_data[key] = np.repeat(data, num_categories, axis=0)
        elif data.ndim == 2:
            if key in need_key:
                dim_two_data[key] = data.flatten()
    return dim_one_data, dim_two_data


def convert_tensor_to_numpy(features):
    numpy_feature = dict()
    for key in features:
        if isinstance(features[key], torch.Tensor):
            numpy_feature[key] = features[key].to(device='cpu').detach().numpy()
        if isinstance(features[key], list):
            data = list()
            for feature in features[key]:
                if isinstance(feature, torch.Tensor):
                    feature = feature.to(device='cpu').detach().numpy()
                data.append(feature)
            numpy_feature[key] = data
    return numpy_feature


def compute_mAP(result):
    recalls = result["recall"]  # list(num_box * iou * score)
    precisions = result["precision"]  # list(num_box * iou * score)
    max_out = result['max_box']
    iou_thresh = result['iou_thresh']
    num_score = len(cfg.NMS.SCORE_CANDIDATES)
    start_idx = 0
    end_idx = num_score
    total_length = recalls.shape[0] // num_score
    temp_data = {'temp_ap': [], 'temp_map': [], 'temp_max': [], 'temp_iou': []}
    for i in range(total_length):
        ap = 0.
        recall = recalls[start_idx:end_idx, :]

        precision = precisions[start_idx: end_idx, :]
        max_pre = precision.copy()
        max_rec = recall.copy()

        max_param = max_out[start_idx]
        iou_param = iou_thresh[start_idx]

        for score in range(num_score - 2, -1, -1):
            max_pre[score] = np.maximum(max_pre[score], max_pre[score + 1])
            print("test", max_pre[score])
        for score in range(1, num_score):
            ap += ((max_rec[score] - max_rec[score - 1]) * max_pre[score])
        mean_ap = np.mean(ap, axis=0)
        # draw_ap_curve(recall, max_pre, True)

        start_idx += len(cfg.NMS.SCORE_CANDIDATES)
        end_idx += len(cfg.NMS.SCORE_CANDIDATES)
        temp_data['temp_ap'].append(ap)
        temp_data['temp_map'].append(mean_ap)
        temp_data['temp_iou'].append(iou_param)
        temp_data['temp_max'].append(max_param)
    class_ap = np.asarray(temp_data['temp_ap'], dtype=np.float32).flatten()
    mean_ap = np.asarray(temp_data['temp_map'], dtype=np.float32)
    iou_ = np.asarray(temp_data['temp_iou'], dtype=np.float32)
    box_ = np.asarray(temp_data['temp_max'], dtype=np.float32)
    return class_ap, mean_ap, iou_, box_


def draw_ap_curve(recall, max_pre, draw=False):
    if draw:
        plt.plot(recall[:, 1], max_pre[:, 1], '-o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.show()
        plt.cla()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    optimize_nms_params()
