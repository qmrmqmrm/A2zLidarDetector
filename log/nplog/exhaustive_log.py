import os.path as op
import numpy as np
import torch
import pandas as pd

import utils.util_function as uf
import model.submodules.model_util as mu
from log.nplog.metric import count_true_positives
import config as cfg


class ExhaustiveLogger:
    def __init__(self, log_cols):
        if cfg.Model.Output.MINOR_CTGR:
            log_cols += ["min_cat_loss"]
        if cfg.Model.Output.LANE_DET:
            log_cols += ["pos_lane_obj", "neg_lane_obj"]
        self.analyzer = ExhaustiveAnalyzer(log_cols)
        self.data = pd.DataFrame()
        self.metric_data = pd.DataFrame()
        self.num_ctgr = None
        self.summary = dict()

    def __call__(self, step, grtr, pred, loss_by_type, epoch, use_anchor):
        data_list = []
        self.num_ctgr = pred["feature_l"]["category"].shape[-1]
        if use_anchor:
            for category in range(1, self.num_ctgr):
                for anchor in range(len(cfg.Tfrdata.ANCHORS_PIXEL)):
                    result = self.analyzer(grtr, pred, loss_by_type, epoch, step, anchor, category)
                    data_list.append(result)
                grtr_match_box = self.box_category_match(grtr["bboxes"], category)
                pred_match_box = self.box_category_match(pred["bboxes"], category)
                metric = count_true_positives(grtr_match_box, pred_match_box, grtr["dontcare"], self.num_ctgr)
                metric.update({"anchor": -1, "category": category})
                self.metric_data = self.metric_data.append(metric, ignore_index=True)

        else:
            result = self.analyzer(grtr, pred, loss_by_type, epoch, step)
            data_list.append(result)
        self.data = self.data.append(data_list, ignore_index=True)

    def box_category_match(self, bbox, category):
        match_bbox = dict()
        valid_mask = bbox["category"] == category
        for key in bbox.keys():
            match_bbox[key] = bbox[key] * valid_mask

        return match_bbox

    def get_summary(self):
        return self.summary

    def make_summary(self):
        mean_data = self.data[cfg.Logging.COLUMNS_TO_MEAN]
        mean_category_data = mean_data.groupby("category", as_index=False).mean()
        mean_category_data["anchor"] = -1

        mean_anchor_data = mean_data.groupby("anchor", as_index=False).mean()
        mean_anchor_data["category"] = -1

        mean_category_anchor_data = mean_data.groupby(["anchor", "category"], as_index=False).mean()

        mean_epoch_data = pd.DataFrame([mean_data.mean(axis=0)])
        mean_epoch_data["anchor"] = -1
        mean_epoch_data["category"] = -1

        mean_summary = pd.concat([mean_epoch_data, mean_anchor_data, mean_category_data, mean_category_anchor_data],
                                 join='outer', ignore_index=True)

        sum_data = self.metric_data[cfg.Logging.COLUMNS_TO_SUM]
        sum_category_data = sum_data.groupby("category", as_index=False).sum()
        sum_category_data = pd.DataFrame({"anchor": sum_data["anchor"][:self.num_ctgr-1],
                                          "category": sum_data["category"][:self.num_ctgr-1],
                                          "recall": sum_category_data["trpo"] / (sum_category_data["grtr"] + 1e-5),
                                          "precision": sum_category_data["trpo"] / (sum_category_data["pred"] + 1e-5),
                                          "grtr": sum_category_data["grtr"], "pred": sum_category_data["pred"],
                                          "trpo": sum_category_data["trpo"]})

        sum_epoch_data = pd.DataFrame([sum_data.sum(axis=0).to_dict()])
        sum_epoch_data = pd.DataFrame({"anchor": -1, "category": -1,
                                       "recall": sum_epoch_data["trpo"] / (sum_epoch_data["grtr"] + 1e-5),
                                       "precision": sum_epoch_data["trpo"] / (sum_epoch_data["pred"] + 1e-5),
                                       "grtr": sum_epoch_data["grtr"], "pred": sum_epoch_data["pred"],
                                       "trpo": sum_epoch_data["trpo"]})
        sum_summary = pd.concat([sum_epoch_data, sum_category_data],
                                join='outer', ignore_index=True)
        sum_box_data = self.data[["anchor", "category", "grtr", "pred"]]
        sum_box_anchor = sum_box_data.groupby("anchor", as_index=False).sum()
        sum_box_anchor["category"] = -1
        sum_box_anchor_ctgr = sum_box_data.groupby(["anchor", "category"], as_index=False).sum()
        box_summary = pd.concat([sum_box_anchor, sum_box_anchor_ctgr], ignore_index=True)
        sum_summary = pd.concat([sum_summary, box_summary], ignore_index=True)
        summary = pd.merge(left=mean_summary, right=sum_summary, how="outer", on=["anchor", "category"])

        self.summary = summary.to_dict()


class ExhaustiveAnalyzer:
    def __init__(self, log_cols):
        self.log_cols = log_cols
        self.exhaust_func = ExhaustivePrepare()
        self.nms = mu.NonMaximumSuppression()

    def __call__(self, grtr, pred, loss_by_type, epoch, step, anchor=None, category=None):
        grtr_map = self.exhaust_func.extract_feature_map(grtr, anchor, is_loss=False)
        pred_map = self.exhaust_func.extract_feature_map(pred, anchor, is_loss=False)
        loss_map = self.exhaust_func.extract_feature_map(loss_by_type, anchor, is_loss=True)
        cate_mask = self.exhaust_func.create_category_mask(grtr_map, category)
        valid_num = np.sum(cate_mask, dtype=np.float32)
        result = dict()
        result["anchor"] = anchor
        result["category"] = category

        if valid_num == 0:
            default_result = dict()
            for key in self.log_cols:
                default_result[key] = 0
            result.update(default_result)
            num_grtr = np.sum(grtr_map["category"] == category, dtype=np.float32)
            result.update({"grtr": num_grtr})
            pred_box = self.collect_anchor_category_match_pred(pred["bboxes"], anchor, category)
            num_pred = np.sum(pred_box["object"] > 0, dtype=np.float32)
            result.update({"pred": num_pred})
            return result

        if "ciou_loss" in self.log_cols:
            result["ciou_loss"] = self.mean_map(loss_map["ciou"], cate_mask, valid_num)
        if "object_loss" in self.log_cols:
            result["object_loss"] = self.mean_map(loss_map["object"], None, valid_num)
        if "maj_cat_loss" in self.log_cols:
            result["maj_cat_loss"] = self.mean_map(loss_map["category"], cate_mask, valid_num)
        if "min_cat_loss" in self.log_cols:
            result["min_cat_loss"] = self.mean_map(loss_map["minor_ctgr"], cate_mask, valid_num)
        if "dist_loss" in self.log_cols:
            result["dist_loss"] = self.mean_map(loss_map["distance"], cate_mask, valid_num)
        if "pos_obj" in self.log_cols:
            result["pos_obj"], result["neg_obj"] = self.pos_neg_obj(grtr_map["object"], pred_map["object"])
        if "iou" in self.log_cols:
            result["iou"] = self.iou_mean(grtr_map["yxhw"], pred_map["yxhw"], valid_num)
        if "box_hw" in self.log_cols:
            result["box_hw"], result["box_yx"] = self.box_mean(grtr_map["yxhw"], pred_map["yxhw"], cate_mask, valid_num)
        if "true_class" in self.log_cols:
            result["true_class"], result["false_class"] = self.tf_class(grtr_map["category"], pred_map["category"],
                                                                        cate_mask)

        num_grtr = np.sum(grtr_map["category"] == category, dtype=np.float32)
        result.update({"grtr": num_grtr})
        pred_box = self.collect_anchor_category_match_pred(pred["bboxes"], anchor, category)
        num_pred = np.sum(pred_box["object"] > 0, dtype=np.float32)
        result.update({"pred": num_pred})
        return result

    def mean_map(self, feature, mask, valid_num):
        if mask is None:
            return np.sum(feature, dtype=np.float32) / valid_num
        return np.sum(feature * mask[..., 0], dtype=np.float32) / valid_num

    def sum_map(self, feature, mask):
        return np.sum(feature * mask, dtype=np.float32)

    def pos_neg_obj(self, grtr, pred):
        p_obj_num = np.sum(grtr, dtype=np.float32)
        pos_obj = np.sum(grtr * pred, dtype=np.float32) / p_obj_num
        neg_obj_map = (1. - grtr) * pred
        test = np.sort(neg_obj_map, axis=1)[::-1]
        neg_obj_map = np.sort(neg_obj_map, axis=1)[:, ::-1]
        neg_obj_map = neg_obj_map[..., :10, :]
        neg_obj = np.mean(neg_obj_map, dtype=np.float32)
        return pos_obj, neg_obj

    def iou_mean(self, grtr, pred, valid_num):
        iou = uf.compute_iou_aligned(grtr, pred).numpy()
        iou = np.sum(iou, dtype=np.float32) / valid_num
        return iou

    def box_mean(self, grtr_yxhw, pred_yxhw, mask, valid_num):
        yx_diff = (np.abs(grtr_yxhw[..., :2] - pred_yxhw[..., :2], dtype=np.float32)) / (grtr_yxhw[..., :2] + 1e-6)
        hw_diff = (np.abs(grtr_yxhw[..., 2:] - pred_yxhw[..., 2:], dtype=np.float32)) / (grtr_yxhw[..., 2:] + 1e-6)

        yx_mean = self.sum_map(yx_diff, mask) / valid_num
        hw_mean = self.sum_map(hw_diff, mask) / valid_num

        return yx_mean, hw_mean

    def tf_class(self, grtr_category, pred_category, grtr_object_ctgr):
        grtr_category_mask = self.one_hot(grtr_category, pred_category.shape[-1])

        grtr_mask = grtr_category_mask * grtr_object_ctgr
        valid_num = np.sum(grtr_mask, dtype=np.float32)
        category_prob = pred_category * grtr_mask
        true_class_per_box = np.max(category_prob, axis=-1)
        true_class = np.sum(true_class_per_box, dtype=np.float32) / valid_num

        invalid_grtr_mask = 1 - grtr_mask
        invalid_num = np.sum(invalid_grtr_mask, dtype=np.float32)
        invalid_prob = pred_category * invalid_grtr_mask
        false_class_per_box = np.max(invalid_prob, axis=-1)
        false_class = np.sum(false_class_per_box, dtype=np.float32) / invalid_num

        return true_class, false_class

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data

    def generate_feature_to_box(self, grtr_map, category):
        valid_grtr = dict()
        for gt_key, gt_val in grtr_map.items():
            grtr_box_single_list = []
            for batch in range(gt_val.shape[0]):
                grtr_box_per_batch = gt_val[batch, ...]
                grtr_batch_box = grtr_box_per_batch[grtr_map["category"][batch, :, 0] == category]
                grtr_box_single_list.append(grtr_batch_box.numpy())
            grtr_box = np.array(grtr_box_single_list, dtype=object)
            grtr = self.box_pad(grtr_box)
            valid_grtr[gt_key] = grtr

        return valid_grtr

    def box_pad(self, grtr):
        grtr_batch_stacked = []
        for batch in range(grtr.shape[0]):
            grtr_box_per_batch = grtr[batch]
            grtr_pad_data = np.zeros((1, grtr_box_per_batch.shape[-1]), np.float32)

            grtr_num_pad = cfg.Tfrdata.MAX_BBOX_PER_IMAGE - grtr_box_per_batch.shape[0]
            grtr_pad_data = np.repeat(grtr_pad_data, grtr_num_pad)
            grtr_pad_data = np.reshape(grtr_pad_data, (grtr_num_pad, -1))

            grtr_padded = np.concatenate([grtr_box_per_batch, grtr_pad_data], axis=0)
            grtr_batch_stacked.append(grtr_padded)
        # grtr_batch_stacked = tf.convert_to_tensor(np.stack(grtr_batch_stacked, axis=0), dtype=tf.float32)
        return grtr_batch_stacked

    def collect_anchor_category_match_pred(self, pred_bbox, anchor, category):
        anchor_mask = np.expand_dims(pred_bbox["anchor_ind"][..., -1] == anchor, axis=-1)
        category_mask = np.expand_dims(pred_bbox["category"][..., -1] == category, axis=-1)
        pred_bbox_per_anchor = dict()
        for key in pred_bbox.keys():
            pred_bbox_per_anchor[key] = pred_bbox[key] * anchor_mask * category_mask
        return pred_bbox_per_anchor


class ExhaustivePrepare:
    def __init__(self):
        # auto generate with config
        self.anchor_to_scale_suffix = {0: "s", 1: "s", 2: "s", 3: "m", 4: "m", 5: "m", 6: "l", 7: "l", 8: "l"}
        self.num_scales = len(cfg.Model.Output.FEATURE_ORDER)

    def extract_feature_map(self, features, anchor, is_loss):
        if is_loss is False:
            features = self.gather_feature_maps_to_level1_dict(features, "feature")
        if anchor is None:
            out_feats = self.merge_over_scales(features, is_loss)
        else:
            out_feats = self.extract_single_anchor(features, anchor, is_loss)
        return out_feats

    def gather_feature_maps_to_level1_dict(self, features, target_key):
        out_feats = dict()
        for key, feat_dict in features.items():
            if key.startswith(target_key):
                scale_suffix = key[-2:]
                for subkey, feat_map in feat_dict.items():
                    out_feats[subkey + scale_suffix] = feat_map
        return out_feats

    def merge_over_scales(self, features, is_loss):
        out_feats = dict()
        for key, feat_dict in features.items():
            if key[-2:] not in ["_l", "_m", "_s"]:
                continue
            new_key = key[:-2]
            if new_key not in out_feats:
                out_feats[new_key] = []
            out_feats[new_key].append(feat_dict)

        for key, feat_maps in out_feats.items():
            if is_loss:
                out_feats[key] = np.concatenate(feat_maps, axis=-1)
            else:
                out_feats[key] = np.concatenate(feat_maps, axis=-2)
        return out_feats

    def extract_single_anchor(self, features, anchor, is_loss):
        out_feats = dict()
        scale_suffix = self.anchor_to_scale_suffix[anchor]
        anchor_idx = anchor % self.num_scales
        for key, feat_map in features.items():
            if key[-1:] == scale_suffix:
                new_key = key[:-2]
                if is_loss:
                    batch, hwa = feat_map.shape
                    channel = False
                else:
                    batch, hwa, channel = feat_map.shape
                hw = hwa // cfg.Model.Output.NUM_ANCHORS_PER_SCALE
                num_anchor = cfg.Model.Output.NUM_ANCHORS_PER_SCALE
                if channel:
                    new_feat = np.reshape(feat_map, (batch, hw, num_anchor, channel))
                    out_feats[new_key] = new_feat[..., anchor_idx, :]
                else:
                    new_feat = np.reshape(feat_map, (batch, hw, num_anchor))
                    out_feats[new_key] = new_feat[..., anchor_idx]
                    # new_feat = new_feat[..., tf.newaxis]
                # new_feat = feat_map[..., 0::hw, :]

        return out_feats

    def create_category_mask(self, grtr_map, category):
        if category is None:
            return grtr_map["object"]
        else:
            return grtr_map["object"] * grtr_map["category"] == category


