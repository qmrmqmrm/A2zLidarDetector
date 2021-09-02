import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import torch
import torch.nn.functional as F

from config import Config as cfg
from train.loss_util import distribute_box_over_feature_map
from model.submodules.matcher import Matcher
from log.history_log import HistoryLog
from log.visual_log import VisualLog
from log.anchor_log import AnchorLog

DEVICE = cfg.Model.Structure.DEVICE


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
    def __init__(self, visual_log, ckpt_path, epoch):
        self.start = timer()
        self.history_logger = HistoryLog()
        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.anchor_logger = AnchorLog(ckpt_path, epoch) if visual_log else None
        self.summary = dict()
        self.nan_grad_count = 0
        self.anchor_matcher = Matcher(
            cfg.Model.RPN.IOU_THRESHOLDS, cfg.Model.RPN.IOU_LABELS, allow_low_quality_matches=True
        )

    def append_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        # nms_boxes = self.nms(pred)
        self.history_logger(step, grtr, pred, total_loss, loss_by_type)
        if self.visual_logger:
            self.visual_logger(step, grtr, pred)

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

    def set_precision(self, logs, precision):
        new_logs = {key: np.around(val, precision) for key, val in logs.items()}
        return new_logs

    def finalize(self):
        self.history_logger.make_summary()

    def get_summary(self):
        return self.history_logger.get_summary()

    def prediction_nms(self, pred):
        print('prediction_nms')
        pred_objectness_logits = pred['pred_objectness_logits']

        gt_class = torch.cat(pred['batched_input']['category'])
        num_box = gt_class.shape[0]

        max_ctgr_probs = torch.max(pred['head_class_logits'][:, :-1], dim=-1)

        pred_proposal_deltas = pred['bbox_3d_logits']
        print(pred_proposal_deltas.shape)
        filter_mask = scores > 3  # R x K
        filter_inds = filter_mask.nonzero()
        scores = scores[filter_mask]

        print(scores.shape)

    def recall_precision(self, gt, pred):
        gt_class = torch.cat(pred['batched_input']['category'])
        gt_shape = gt_class.shape
        scores = pred['head_class_logits']
        head_proposals = pred['head_proposals']
        gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0).type(torch.int64).to(DEVICE)

        gt_onehot = F.one_hot(gt_classes)
        max_scores = torch.argmax(scores, dim=1)
        score_onehot = F.one_hot(max_scores)

        score_slice = scores[:gt_shape[0]]
        print('score_slice')
        print(score_slice)

        gt_classes_false = torch.nonzero(gt_onehot[:, -1] > 0)
        gt_classes_true = torch.nonzero(gt_onehot[:, -1] == 0)
        score_true = torch.nonzero(score_onehot[:, -1] == 0)
        score_false = torch.nonzero(score_onehot[:, -1] > 0)
        print('gt')
        print(gt_classes_true.shape)
        print(gt_classes_false.shape)
        print('score')
        print(score_true.shape)
        print(score_false.shape)
        # print(score_true)

        # filter_mask = scores > 0.05  # R x K
        # print(filter_mask)
        # scores = scores[filter_mask]
        # gt_onehot = gt_onehot[filter_mask]
        # print(scores.shape)
        # print(scores)
        # print(gt_onehot.shape)
        # print(gt_onehot)

        # gt_ture =
        # pred_ture =
        # TP = gt_true and pred_true
        # TN = gt_true and pred_false
        # FP = gt_false and pred_true
        # FN = gt_false and pred_false
        # precision = TP / (FP + TP) pre->1024(not 0
        # recall = TP / (TN + TP) gt
