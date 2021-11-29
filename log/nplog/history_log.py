import numpy as np
import torch
import pandas as pd
from timeit import default_timer as timer

from log.nplog.metric import count_true_positives, IouEstimator, RotatedIouEstimator
import utils.util_function as uf
import config as cfg


class HistoryLog:
    def __init__(self):
        self.batch_data_table = pd.DataFrame()
        self.start = timer()
        self.summary = dict()
        self.num_categs = cfg.Model.Structure.NUM_CLASSES

    def __call__(self, step, grtr, gt_aligned, gt_feature, pred, pred_nms, total_loss, loss_by_type):
        """
        :param step: integer step index
        :param grtr:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4](tlbr), 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
        :param pred:
            {'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'head_output' : torch.Size([4, 512, 93])
            }
        :param total_loss:
        :param loss_by_type:
        """
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name] for loss_name in loss_list}
        batch_data["total_loss"] = total_loss
        box_objectness = self.analyze_box_objectness(gt_feature, pred)
        batch_data.update(box_objectness)

        num_ctgr = pred_nms["ctgr_probs"].shape[-1] - 1
        print()
        print('img', grtr['image_file'])
        metric = count_true_positives(grtr, pred_nms, num_ctgr, IouEstimator(), per_class=True)
        metric_rot = count_true_positives(grtr, pred_nms, num_ctgr, RotatedIouEstimator(), per_class=True)
        metric_rota = dict()
        for key in metric_rot:
            new_key = key + "_rot"
            metric_rota[new_key] = metric_rot[key]

        uf.print_structure('metric_rota', metric_rota)
        batch_data.update(metric)
        batch_data.update(metric_rota)
        # pred_cate = torch.softmax(torch.tensor(pred["category"]), dim=-1).to('cpu').detach().numpy()
        batch_data["true_cls"] = self.logtrueclass(gt_aligned, pred["ctgr_probs"])
        batch_data["false_cls"] = self.logfalseclass(gt_aligned, pred["ctgr_probs"])

        batch_data = self.set_precision(batch_data, 5)
        col_order = list(batch_data.keys())
        self.batch_data_table = self.batch_data_table.append(batch_data, ignore_index=True)
        self.batch_data_table = self.batch_data_table.loc[:, col_order]


        if step % 20 == 10:
            print("\n--- batch_data:", batch_data)
            self.check_pred_scales(pred_nms)

    def analyze_box_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        pos_obj_sc, neg_obj_sc = self.pos_neg_obj(np.concatenate(grtr["object"], axis=1),
                                                  np.concatenate(pred["rpn_feat_objectness"], axis=1))
        pos_obj += pos_obj_sc
        neg_obj += neg_obj_sc
        objectness = {"pos_obj": pos_obj, "neg_obj": neg_obj}
        return objectness

    def pos_neg_obj(self, grtr_obj_mask, pred_obj_prob):
        obj_num = np.maximum(np.sum(grtr_obj_mask, dtype=np.float32), 1)
        pos_obj = np.sum(grtr_obj_mask * pred_obj_prob, dtype=np.float32) / obj_num
        # average top 50 negative objectness probabilities per frame
        neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
        batch, hwa, _ = neg_obj_map.shape
        neg_obj_map = np.reshape(neg_obj_map, (batch * hwa))
        neg_obj_map = np.sort(neg_obj_map, axis=0)[::-1]
        neg_obj_map = neg_obj_map[:50]
        neg_obj = np.mean(neg_obj_map, dtype=np.float32)
        return pos_obj, neg_obj

    def check_pred_scales(self, pred):
        for key, feat in pred.items():
            if isinstance(feat, dict):
                for subkey, subfeat in feat.items():
                    if np.max(np.abs(subfeat)) > 1e4:
                        print(f"[pred scale] {key}/{subkey}:", np.quantile(subfeat.numpy(), np.linspace(0, 1, 6)))

    def set_precision(self, logs, precision):
        new_logs = {key: np.around(val, precision) for key, val in logs.items()}
        return new_logs

    def make_summary(self):
        mean_result = self.batch_data_table.mean(axis=0).to_dict()
        sum_result = self.batch_data_table.sum(axis=0).to_dict()
        sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
                      "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5),
                      "trpo_num": sum_result["trpo"],
                      "grtr_num": sum_result["grtr"],
                      "pred_num": sum_result["pred"],
                      "recall_rot": sum_result["trpo_rot"] / (sum_result["grtr_rot"] + 1e-5),
                      "precision_rot": sum_result["trpo_rot"] / (sum_result["pred_rot"] + 1e-5),
                      "trpo_rot_num": sum_result["trpo_rot"],
                      "grtr_rot_num": sum_result["grtr_rot"],
                      "pred_rot_num": sum_result["pred_rot"]
                      }
        metric_keys = ["trpo", "grtr", "pred"]

        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round((timer() - self.start) / 60., 5)
        self.summary = summary

    def get_summary(self):
        return self.summary

    def logtrueclass(self, grtr, pred_cate):
        grtr_ctgr_mask = self.one_hot(grtr["category"], self.num_categs + 1)
        grtr_target_mask = grtr["object"]
        category_prob = grtr_target_mask * grtr_ctgr_mask * pred_cate
        true_class = np.sum(category_prob) / (np.sum(grtr_target_mask) + 0.00001)
        return true_class

    def logfalseclass(self, grtr, pred_cate):
        grtr_ctgr_mask = self.one_hot(grtr["category"], self.num_categs + 1)
        grtr_target_mask = grtr["object"]
        category_prob = grtr_target_mask * pred_cate * (1. - grtr_ctgr_mask)
        false_prob = np.max(category_prob, axis=-1)
        false_class = np.sum(false_prob) / (np.sum(grtr_target_mask) + 0.00001)
        return false_class

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape, dtype=np.float32)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data

