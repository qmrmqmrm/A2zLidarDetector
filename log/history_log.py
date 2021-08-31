import numpy as np
import tensorflow as tf
import pandas as pd
from timeit import default_timer as timer

from log.metric import count_true_positives
from config import Config as model_cfg


class HistoryLog:
    def __init__(self):
        self.batch_data_table = pd.DataFrame()
        self.start = timer()
        self.summary = dict()
        self.sign_idx = model_cfg.CATEGORIES.index("Traffic sign")
        self.mark_idx = model_cfg.CATEGORIES.index("Road mark")

    def __call__(self, step, grtr, pred, total_loss, loss_by_type):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param total_loss:
        :param loss_by_type:
        """
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name].numpy() for loss_name in loss_list}
        batch_data["total_loss"] = total_loss.numpy()
        box_objectness = self.analyze_box_objectness(grtr, pred)
        if model_cfg.TRAIN_LANE:
            lane_objectness = self.analyze_lane_objectness(grtr, pred)
            batch_data.update(lane_objectness)
        batch_data.update(box_objectness)
        num_ctgr = pred["feature_l"]["category"].shape[-1]

        metric = self.compute_metrics(grtr["bboxes"], grtr["dontcare"], pred["bboxes"], num_ctgr)
        batch_data.update(metric)

        batch_data = self.set_precision(batch_data, 5)
        col_order = list(batch_data.keys())
        self.batch_data_table = self.batch_data_table.append(batch_data, ignore_index=True)
        self.batch_data_table = self.batch_data_table.loc[:, col_order]

        if step % 200 == 10:
            print("\n--- batch_data:", batch_data)
        #     self.check_pred_scales(pred)

    def compute_metrics(self, grtr_bbox, grtr_dc, pred_bbox, num_ctgr):
        metric = {}
        major_metric = count_true_positives(grtr_bbox, grtr_dc, pred_bbox, num_ctgr, per_class=False)
        metric.update(major_metric)
        if
            sign_metric = self.get_minor_metric(grtr_bbox, grtr_dc, pred_bbox, num_ctgr, self.sign_idx, "sign")
            mark_metric = self.get_minor_metric(grtr_bbox, grtr_dc, pred_bbox, num_ctgr, self.mark_idx, "mark")
            metric.update(sign_metric)
            metric.update(mark_metric)
        return metric

    def get_minor_metric(self, grtr_bbox, grtr_dc, pred_bbox, num_ctgr, target_idx, suffix):
        grtr_valid_mask = tf.cast(grtr_bbox["category"] == target_idx, dtype=tf.float32)
        pred_valid_mask = tf.cast(pred_bbox["category"] == target_idx, dtype=tf.float32)
        grtr_minor = {key: val * grtr_valid_mask for key, val in grtr_bbox.items()}
        pred_minor = {key: val * pred_valid_mask for key, val in pred_bbox.items()}
        minor_metric = count_true_positives(grtr_minor, grtr_dc, pred_minor, num_ctgr, per_class=False,
                                            category_key="minor_ctgr", suffix=suffix)
        return minor_metric

    def analyze_box_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        scales = ["feature_s", "feature_m", "feature_l"]
        for scale_name in scales:
            pos_obj_sc, neg_obj_sc = self.pos_neg_obj(grtr[scale_name]["object"], pred[scale_name]["object"])
            pos_obj += pos_obj_sc
            neg_obj += neg_obj_sc
        objectness = {"pos_obj": pos_obj.numpy() / len(scales), "neg_obj": neg_obj.numpy() / len(scales)}
        return objectness

    def analyze_lane_objectness(self, grtr, pred):
        pos_obj, neg_obj = self.pos_neg_obj(grtr["feat_lane"]["object"], pred["feat_lane"]["object"])
        objectness = {"pos_obj": pos_obj.numpy(), "neg_obj": neg_obj.numpy()}
        return objectness

    def pos_neg_obj(self, grtr_obj_mask, pred_obj_prob):
        obj_num = tf.maximum(tf.reduce_sum(grtr_obj_mask), 1)
        pos_obj = tf.reduce_sum(grtr_obj_mask * pred_obj_prob) / obj_num
        # average top 50 negative objectness probabilities per frame
        neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
        batch, hwa, _ = neg_obj_map.shape
        neg_obj_map = tf.reshape(neg_obj_map, (batch * hwa, -1))
        neg_obj_map = tf.sort(neg_obj_map, axis=-1, direction="DESCENDING")
        neg_obj_map = neg_obj_map[:, :50]
        neg_obj = tf.reduce_mean(neg_obj_map)
        return pos_obj, neg_obj

    def analyze_box_ctgr(self, grtr, pred):
        scales = [key for key in grtr if "feature_" in key]
        category_probs = {"major_pos_prob": 0, "major_neg_prob": 0}

        for scale_name in scales:
            pred_major_ctgr = pred[scale_name]["category"]
            grtr_major_ctgr = grtr[scale_name]["category"]
            ignore_bgd = tf.cast(grtr_major_ctgr[..., 0:1] > 0, dtype=tf.float32)
            grtr_major_mask = tf.one_hot(tf.cast(grtr_major_ctgr, dtype=tf.int32), depth=pred_major_ctgr.shape[-1], axis=-1) * ignore_bgd

            major_positive_ctgr_prob = pred_major_ctgr * grtr_major_mask
            valid_major_count = tf.reduce_sum(grtr_major_mask, axis=[0, 1])
            valid_major_count = tf.maximum(valid_major_count, 1)
            major_positive_ctgr_prob = tf.reduce_sum(major_positive_ctgr_prob, axis=[0, 1]) / valid_major_count

            major_negative = pred_major_ctgr * (1 - grtr_major_mask)
            major_negative = tf.reshape(major_negative, (major_negative.shape[0] * major_negative.shape[1], -1))
            major_top_50_negative = tf.sort(major_negative, axis=0)[-50:, ...]
            major_top_50_negative = tf.reduce_mean(major_top_50_negative, axis=0)

            category_probs["major_pos_prob"] += major_positive_ctgr_prob
            category_probs["major_neg_prob"] += major_top_50_negative

        for category in category_probs:
            category_probs[category] /= len(scales)
        return category_probs

    def check_pred_scales(self, pred):
        raw_features = {key: tensor for key, tensor in pred.items() if key.endswith("raw")}
        pred_scales = dict()
        for key, feat in raw_features.items():
            pred_scales[key] = np.quantile(feat.numpy(), np.array([0.05, 0.5, 0.95]))
        print("--- pred_scales:", pred_scales)

    def set_precision(self, logs, precision):
        new_logs = {key: np.around(val, precision) for key, val in logs.items()}
        return new_logs

    def make_summary(self):
        mean_result = self.batch_data_table.mean(axis=0).to_dict()
        sum_result = self.batch_data_table.sum(axis=0).to_dict()
        metric_result = dict()
        metric_keys = []
        for suffix in ["", "_sign", "_mark"]:
            metric_result["recall" + suffix] = sum_result["trpo" + suffix] / (sum_result["grtr" + suffix] + 1e-5)
            metric_result["precision" + suffix] = sum_result["trpo" + suffix] / (sum_result["pred" + suffix] + 1e-5)
            metric_keys += ["trpo" + suffix, "grtr" + suffix, "pred" + suffix]
        print("metric keys", metric_result)
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(metric_result)
        summary["time_m"] = round((timer() - self.start ) /60., 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary
