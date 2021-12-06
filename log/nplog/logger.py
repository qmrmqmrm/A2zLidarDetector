import os
import os.path as op
import numpy as np
import math
import torch
import pandas as pd

from log.nplog.exhaustive_log import ExhaustiveLogger
from log.nplog.history_log import HistoryLog
from log.nplog.visual_log import VisualLog
import utils.util_function as uf
import model.submodules.model_util as mu
import config as cfg


class LogFile:
    def __init__(self, ckpt_path):
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.val_filename = op.join(ckpt_path, "history_val.csv")
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)

    def save_log(self, epoch, train_log, val_log):
        history_summary = self.merge_logs(epoch, train_log.get_history_summary(), val_log.get_history_summary())
        # if val_log.exhuastive_logger is not None:
        #     exhaust_summary = val_log.get_exhuastive_summary()
        #     exhaust_filename = self.exhaust_path + f"/{epoch}.csv"
        #     exhaust = pd.DataFrame(exhaust_summary)
        #     exhaust.to_csv(exhaust_filename, encoding='utf-8', index=False, float_format='%.4f')

        if op.isfile(self.history_filename):
            history = pd.read_csv(self.history_filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(history_summary, ignore_index=True)
        else:
            history = pd.DataFrame([history_summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        history.to_csv(self.history_filename, encoding='utf-8', index=False, float_format='%.4f')

    def merge_logs(self, epoch, train_summary, val_summary):
        summary = dict()
        summary["epoch"] = epoch
        if train_summary is not None:
            train_summary = {"!" + key: val for key, val in train_summary.items()}
            summary.update(train_summary)
            summary["|"] = 0
        if "anchor" in val_summary:
            del val_summary["anchor"]
            del val_summary["category"]
        val_summary = {"`" + key: val for key, val in val_summary.items()}
        summary.update(val_summary)
        return summary

    def save_val_log(self, val_log):
        print("validation summary:", val_log.get_history_summary())
        history_summary = self.merge_logs(0, None, val_log.get_history_summary())
        val_filename = self.history_filename[:-4] + "_val.csv"
        if op.isfile(val_filename):
            history = pd.read_csv(val_filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(history_summary, ignore_index=True)
        else:
            history = pd.DataFrame([history_summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        print("val_filename", val_filename)
        history.to_csv(val_filename, encoding='utf-8', index=False, float_format='%.4f')


class Logger:
    def __init__(self, visual_log, exhuastive_log, ckpt_path, epoch, split):
        self.split = split
        self.history_logger = HistoryLog()
        self.visual_logger = VisualLog(ckpt_path, epoch, split) if visual_log else None
        self.aligned_iou_threshold = cfg.Loss.ALIGN_IOU_THRESHOLD
        self.anchor_iou_threshold = cfg.Loss.ANCHOR_IOU_THRESHOLD
        self.device = cfg.Hardware.DEVICE
        # self.exhuastive_logger = ExhaustiveLogger(cfg.Logging.COLNUMS) if exhuastive_log else None
        self.nms = mu.NonMaximumSuppression()

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type, auxi):
        """

        :param step:
        :param grtr:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4](tlbr),
            'bbox3d': [batch, fixbox, 6],
            'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1],
            'yaw_rads': [batch, fixbox, 1]},
            'anchor_id': [batch, fixbox, 1]
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
        :param epoch:
        :return:
        """
        pred_select = self.select_best_ctgr_pred(pred)
        pred_slices_nms = self.nms(pred_select)
        gt_aligned = auxi['gt_aligned']
        gt_feature = auxi['gt_feature']
        grtr = self.convert_tensor_to_numpy(grtr)
        pred['inst'] = self.convert_tensor_to_numpy(pred['inst'])
        pred['fmap'] = self.convert_tensor_to_numpy(pred['fmap'])
        pred_slices_nms = self.convert_tensor_to_numpy(pred_slices_nms)
        loss_by_type = self.convert_tensor_to_numpy(loss_by_type)
        gt_feature = self.convert_tensor_to_numpy(gt_feature)
        gt_aligned = self.convert_tensor_to_numpy(gt_aligned)
        total_loss = total_loss.to('cpu').detach().numpy()
        self.history_logger(step, grtr, gt_aligned, gt_feature, pred, pred_slices_nms, total_loss, loss_by_type)
        if self.visual_logger:
            self.visual_logger(step, grtr, pred_slices_nms, gt_feature, pred['fmap'])
        # if self.exhuastive_logger:
        #     self.exhuastive_logger(step, grtr, gt_aligned, pred_slices, loss_by_type, epoch, cfg.Logging.USE_ANCHOR)

    def select_best_ctgr_pred(self, pred):
        best_ctgr = torch.argmax(pred['inst']['category'], dim=-1).unsqueeze(-1)
        need_key = ['bbox3d', 'yaw_cls', 'yaw_cls_logit', 'yaw_res', 'bbox3d_delta']
        select_pred = uf.select_category(pred['inst'], best_ctgr, need_key)
        best_yaw_cls_idx = torch.argmax(select_pred['yaw_cls'], dim=-1).unsqueeze(-1)
        select_pred['yaw_res'] = torch.gather(select_pred['yaw_res'], dim=-1,
                                              index=best_yaw_cls_idx)
        select_pred['yaw_rads'] = (best_yaw_cls_idx * (math.pi / cfg.Model.Structure.VP_BINS) - (math.pi / 2) +
                                   select_pred['yaw_res'])
        return select_pred

    def convert_tensor_to_numpy(self, features, key=''):
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

    def finalize(self):
        self.history_logger.make_summary()

    def get_history_summary(self):
        return self.history_logger.get_summary()

    # def get_exhuastive_summary(self):
    #     return self.exhuastive_logger.get_summary()
