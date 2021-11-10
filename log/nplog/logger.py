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
        pred = self.convert_tensor_to_numpy(pred)
        pred_slices_nms = self.convert_tensor_to_numpy(pred_slices_nms)
        loss_by_type = self.convert_tensor_to_numpy(loss_by_type)
        gt_feature = self.convert_tensor_to_numpy(gt_feature)
        gt_aligned = self.convert_tensor_to_numpy(gt_aligned)
        total_loss = total_loss.to('cpu').detach().numpy()

        self.history_logger(step, grtr, gt_aligned, gt_feature, pred, pred_slices_nms, total_loss, loss_by_type)
        if self.visual_logger:
            self.visual_logger(step, grtr, gt_feature, pred, pred_slices_nms)
        # if self.exhuastive_logger:
        #     self.exhuastive_logger(step, grtr, gt_aligned, pred_slices, loss_by_type, epoch, cfg.Logging.USE_ANCHOR)

    def select_best_ctgr_pred(self, pred):
        best_ctgr = torch.argmax(pred['ctgr_probs'], dim=-1).unsqueeze(-1)
        need_key = ['bbox3d', 'yaw_cls_probs','yaw_cls_logit', 'yaw_res', 'bbox3d_delta']
        select_pred = uf.select_category(pred, best_ctgr, need_key)
        best_yaw_cls_idx = torch.argmax(select_pred['yaw_cls_probs'], dim=-1).unsqueeze(-1)
        select_pred['yaw_res'] = torch.gather(select_pred['yaw_res'], dim=-1,
                                              index=best_yaw_cls_idx)
        select_pred['yaw_rads'] = (best_yaw_cls_idx * (math.pi / cfg.Model.Structure.VP_BINS) - (math.pi / 2) +
                                   select_pred['yaw_res'])
        return select_pred

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

    def matched_gt(self, grtr, pred_box, iou_threshold):
        batch_size = grtr['bbox2d'].shape[0]
        matched = {key: [] for key in
                   ['bbox3d', 'category', 'bbox2d', 'yaw_cls', 'yaw_rads', 'anchor_id', 'object', 'negative']}
        for i in range(batch_size):
            iou_matrix = uf.pairwise_iou(grtr['bbox2d'][i], pred_box[i])
            match_ious, match_inds = iou_matrix.max(dim=0)  # (height*width*anchor)
            positive = (match_ious >= iou_threshold[1]).unsqueeze(-1)
            negative = (match_ious < iou_threshold[0]).unsqueeze(-1)
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
        # if self.exhuastive_logger:
        #     self.exhuastive_logger.make_summary()

    def get_history_summary(self):
        return self.history_logger.get_summary()

    def get_exhuastive_summary(self):
        return self.exhuastive_logger.get_summary()
