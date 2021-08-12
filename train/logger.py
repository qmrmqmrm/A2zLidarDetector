import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import torch

import utils.util_function as uf
from config import Config as cfg


class LogFile:
    def save_log(self, epoch, train_log, val_log):
        summary = self.merge_logs(epoch, train_log, val_log)
        filename = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME, "history.csv")
        if op.isfile(filename):
            history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(summary, ignore_index=True)
        else:
            history = pd.DataFrame([summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        history.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')

    def merge_logs(self, epoch, train_log, val_log):
        summary = dict()
        summary["epoch"] = epoch
        train_summary = train_log.get_summary()
        train_summary = {"!" + key: val for key, val in train_summary.items()}
        summary.update(train_summary)
        summary["|"] = 0
        val_summary = val_log.get_summary()
        val_summary = {"`" + key: val for key, val in val_summary.items()}
        summary.update(val_summary)
        return summary


class LogData:
    def __init__(self):
        self.batch = pd.DataFrame()
        self.start = timer()
        self.summary = dict()
        self.nan_grad_count = 0

    def append_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name].cpu().detach().numpy() for loss_name in loss_list}
        batch_data["total_loss"] = total_loss.cpu().detach().numpy()
        # objectness = self.analyze_objectness(grtr, pred)
        # batch_data.update(objectness)

        # self.check_nan(batch_data, grtr, pred)
        batch_data = self.set_precision(batch_data, 5)
        self.batch = self.batch.append(batch_data, ignore_index=True)
        # print("\n--- batch_data:", batch_data)
        if step % 200 == 10:
            print("\n--- batch_data:", batch_data)

        #     self.check_pred_scales(pred)

    def analyze_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        print(type(grtr))
        print(len(grtr))
        print(type(grtr[0]))
        print((grtr[0].keys()))
        print(type(pred))
        print((pred.keys()))
        print(len(pred['pred_objectness_logits']))
        # grtr_slices = uf.slice_features(grtr)
        # pred_slices = uf.slice_features(pred)

        grtr_objs = [gt["gt_object"] for gt in grtr]
        grtr_obj_mask = torch.cat(grtr_objs,dim=0)

        pred_obj = [obj for obj in pred["pred_objectness_logits"]]
        print(grtr_objs)
        print(pred_obj)
        pred_obj_prob = torch.cat(pred_obj,dim=1)
        print(grtr_obj_mask.shape)
        print(pred_obj_prob.shape)
        obj_num = torch.maximum(torch.sum(grtr_obj_mask))
        # average positive objectness probability
        pos_obj += torch.sum(grtr_obj_mask * pred_obj_prob) / obj_num
        # average top 50 negative objectness probabilities per frame
        neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
        batch, grid_h, grid_w, anchor, _ = neg_obj_map.shape
        neg_obj_map = torch.reshape(neg_obj_map, (batch, grid_h * grid_w * anchor))
        neg_obj_map = torch.sort(neg_obj_map, axis=-1)
        neg_obj_map = neg_obj_map[:, :50]
        neg_obj += torch.mean(neg_obj_map)
        print(pos_obj)
        print(neg_obj)
        objectness = {"pos_obj": pos_obj.numpy() / len(scales), "neg_obj": neg_obj.numpy() / len(scales)}
        return objectness

    def check_nan(self, losses, grtr, pred):
        valid_result = True
        for name, loss in losses.items():
            if np.isnan(loss) or np.isinf(loss) or loss > 100:
                print(f"nan loss: {name}, {loss}")
                valid_result = False
        for name, tensor in pred.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, tensor in grtr.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False

        assert valid_result

    def check_pred_scales(self, pred):
        raw_features = {key: tensor for key, tensor in pred.items() if key.endswith("raw")}
        pred_scales = dict()
        for key, feat in raw_features.items():
            pred_scales[key] = np.quantile(feat.numpy(), np.array([0.05, 0.5, 0.95]))
        print("--- pred_scales:", pred_scales)

    def set_precision(self, logs, precision):
        new_logs = {key: np.around(val, precision) for key, val in logs.items()}
        return new_logs

    def finalize(self):
        self.summary = self.batch.mean(axis=0).to_dict()
        self.summary["time_m"] = round((timer() - self.start)/60., 5)
        print("finalize:", self.summary)
    
    def get_summary(self):
        return self.summary
