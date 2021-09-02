import numpy as np
import torch
import pandas as pd
from timeit import default_timer as timer

from log.metric import count_true_positives


class HistoryLog:
    def __init__(self):
        self.batch_data_table = pd.DataFrame()
        self.start = timer()
        self.summary = dict()

    def __call__(self, step, grtr, pred, total_loss, loss_by_type):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param total_loss:
        :param loss_by_type:
        """
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name].cpu().detach().numpy() for loss_name in loss_list}
        batch_data["total_loss"] = total_loss.cpu().detach().numpy()

        # box_objectness = self.analyze_box_objectness(grtr, pred)
        # batch_data.update(box_objectness)
        # 
        # num_ctgr = pred["feature_l"]["category"].shape[-1]
        # metric = self.compute_metrics(grtr["bboxes"], grtr["dontcare"], pred["bboxes"], num_ctgr)
        # batch_data.update(metric)

        batch_data = self.set_precision(batch_data, 5)
        col_order = list(batch_data.keys())
        self.batch_data_table = self.batch_data_table.append(batch_data, ignore_index=True)
        self.batch_data_table = self.batch_data_table.loc[:, col_order]

        if step % 200 == 10:
            print("\n--- batch_data:", batch_data)

    def compute_metrics(self, grtr_bbox, grtr_dc, pred_bbox, num_ctgr):
        metric = {}
        major_metric = count_true_positives(grtr_bbox, grtr_dc, pred_bbox, num_ctgr, per_class=False)
        metric.update(major_metric)
        return metric

    def analyze_box_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        scales = ["feature_s", "feature_m", "feature_l"]
        for scale_name in scales:
            pos_obj_sc, neg_obj_sc = self.pos_neg_obj(grtr[scale_name]["object"], pred[scale_name]["object"])
            pos_obj += pos_obj_sc
            neg_obj += neg_obj_sc
        objectness = {"pos_obj": pos_obj.numpy() / len(scales), "neg_obj": neg_obj.numpy() / len(scales)}
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

    def analyze_objectness(self, grtr, pred):
        anchors = pred['anchors']
        pred_objectness_logits = pred['pred_objectness_logits']
        gt_labels, gt_boxes = distribute_box_over_feature_map(anchors, grtr, self.anchor_matcher)
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        num_pos_anchors = num_pos_anchors / num_images
        num_neg_anchors = num_neg_anchors / num_images
        objectness = {"pos_obj": num_pos_anchors, "neg_obj": num_neg_anchors}
        return objectness

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
        summary = mean_result
        if "trpo" in sum_result:
            metric_result = dict()
            metric_result["recall"] = sum_result["trpo"] / (sum_result["grtr"] + 1e-5)
            metric_result["precision"] = sum_result["trpo"] / (sum_result["pred"] + 1e-5)
            metric_keys = ["trpo", "grtr", "pred"]
            print("metric keys", metric_result)
            summary = {key: val for key, val in summary.items() if key not in metric_keys}
            summary.update(metric_result)

        summary["time_m"] = round((timer() - self.start) / 60., 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary
