import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

import config as cfg
# from train.inference import Inference
from model.submodules.matcher import Matcher
from model.nms import NonMaximumSuppresion
from log.history_log import HistoryLog
from log.visual_log import VisualLog
from log.anchor_log import AnchorLog
import utils.util_function as uf

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
    def __init__(self, visual_log, ckpt_path, epoch, training):
        self.start = timer()
        self.training = training
        self.history_logger = HistoryLog(self.training)

        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.anchor_logger = AnchorLog(ckpt_path, epoch) if visual_log else None
        self.nms = NonMaximumSuppresion()

        self.summary = dict()
        self.nan_grad_count = 0
        self.anchor_matcher = Matcher(cfg.Model.RPN.IOU_THRESHOLDS, cfg.Model.RPN.IOU_LABELS,
                                      allow_low_quality_matches=True)
        self.score_thresh = cfg.Model.ROI_HEADS.NMS_SCORE_THRESH
        self.iou_thresh = cfg.Model.ROI_HEADS.NMS_IOU_THRESH

    def append_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        pred_nms = None
        if not self.training:
            out = Inference(pred)
            pred_nms = out.inference(self.score_thresh, self.iou_thresh, 100)

        self.history_logger(step, grtr, pred, total_loss, loss_by_type, pred_nms)
        if self.visual_logger:
            self.visual_logger(step, grtr, pred_nms, 'pred_nms')

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

    def matched_nms(self, grtr, pred):
        gt_proposal = list()
        for i, image_file in enumerate(grtr['image_file']):
            proposals = pred[0][i]
            gt_bbox2d = grtr['bbox2d'][i]
            gt_classes = grtr['category'][i]

            proposals_box = proposals['pred_boxes']
            proposals_class = proposals['pred_classes']

            gt_proposal_box, gt_proposals_class = uf.matched_category_gt_with_pred(proposals_box, gt_bbox2d, gt_classes,
                                                                                   proposals_class)
            gt_proposal_per_image = {'gt_proposal_box': gt_proposal_box, 'gt_proposals_class': gt_proposals_class}
            gt_proposal.append(gt_proposal_per_image)
        return gt_proposal
