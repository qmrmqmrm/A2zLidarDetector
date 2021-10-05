import os
import os.path as op
import numpy as np
import torch
import pandas as pd

from log.nplog.exhaustive_log import ExhaustiveLogger
from log.nplog.history_log import HistoryLog
from log.nplog.visual_log import VisualLog
import utils.util_function as uf
import model.submodules.model_util as mu
import config as cfg


class Logger:
    def __init__(self, visual_log, exhuastive_log, ckpt_path, epoch):
        self.history_logger = HistoryLog()
        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.exhuastive_logger = ExhaustiveLogger(cfg.Logging.COLNUMS) if exhuastive_log else None
        self.nms = mu.NonMaximumSuppression()

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type, epoch):
        # self.check_nan(grtr, pred, loss_by_type)
        pred_slices = uf.merge_and_slice_features(pred)
        # nms_boxes = self.nms(pred_slices)
        gt_aligned = self.matched_gt(grtr, pred['rpn_bbox2d'])
        grtr = self.convert_tensor_to_numpy(grtr)
        gt_aligned = self.convert_tensor_to_numpy(gt_aligned)
        pred_slices = self.convert_tensor_to_numpy(pred_slices)
        loss_by_type = self.convert_tensor_to_numpy(loss_by_type)
        total_loss = total_loss.to('cpu').detach().numpy()

        # self.numeric_logger(step, grtr, gt_aligned, pred_slices, loss_by_type, total_loss)
        # if self.visual_logger:
        #     self.visual_logger(step, gt_aligned, pred_slices)
        self.history_logger(step, grtr, gt_aligned, pred_slices, total_loss, loss_by_type)
        if self.visual_logger:
            self.visual_logger(step, grtr, pred_slices)
        if self.exhuastive_logger:
            self.exhuastive_logger(step, grtr, pred_slices, loss_by_type, epoch, cfg.Logging.USE_ANCHOR)


    def check_nan(self, grtr, pred, loss_by_type):
        valid_result = True
        for name, tensor in grtr.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, tensor in pred.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, loss in loss_by_type.items():
            if loss.ndim == 0 and (np.isnan(loss) or np.isinf(loss) or loss > 100000):
                print(f"nan loss: {name}, {loss}")
                valid_result = False
        assert valid_result

    # def convert_tensor_to_numpy(self, feature):
    #     numpy_feature = dict()
    #     uf.print_structure('feature', feature)
    #     for key, value in feature.items():
    #         if isinstance(value, dict):
    #             sub_feature = dict()
    #             for sub_key, sub_value in value.items():
    #                 sub_feature[sub_key] = sub_value.numpy()
    #             numpy_feature[key] = sub_feature
    #         else:
    #             numpy_feature[key] = value.numpy()
    #     return numpy_feature

    def convert_tensor_to_numpy(self, features):
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

    def matched_gt(self, grtr, pred_box):
        matched = {key: [] for key in
                   ['bbox3d', 'category', 'bbox2d', 'yaw', 'yaw_rads', 'anchor_id', 'object', 'negative']}
        for i in range(4):
            iou_matrix = uf.pairwise_iou(grtr['bbox2d'][i], pred_box[i])
            match_ious, match_inds = iou_matrix.max(dim=0)  # (height*width*anchor)
            positive = (match_ious > 0.4).unsqueeze(-1)
            negative = (match_ious < 0.1).unsqueeze(-1)
            for key in matched:
                if key == "negative":
                    matched["negative"].append(negative)
                else:
                    matched[key].append(grtr[key][i, match_inds] * positive)
        for key in matched:
            matched[key] = torch.stack(matched[key], dim=0)
        return matched

    def split_feature(self, anchors, feature):
        slice_features = {key: [] for key in feature.keys()}
        for key in feature.keys():
            last_channel = 0
            for anchor in anchors:
                scales = anchor.shape[1] + last_channel
                slice_feature = feature[key][:, last_channel:scales]
                last_channel = scales
                slice_features[key].append(slice_feature)
        return slice_features

    def finalize(self):
        self.history_logger.make_summary()
        if self.exhuastive_logger:
            self.exhuastive_logger.make_summary()


    def get_history_summary(self):
        return self.history_logger.get_summary()

    def get_exhuastive_summary(self):
        return self.exhuastive_logger.get_summary()