import numpy as np
import torch
from torch import nn
from torchvision.ops import boxes as box_ops

from model.submodules.box_regression import Box2BoxTransform
import utils.util_function as uf
import config as cfg
import model.submodules.model_util as mu


class RPN(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        self.device = cfg.Hardware.DEVICE
        self.input_shapes = [cfg.get_img_shape("HW", dataset_name, scale) for scale in cfg.Model.Neck.OUT_SCALES]
        self.input_channels = cfg.Model.Neck.OUTPUT_CHANNELS
        self.num_anchor = len(cfg.Model.RPN.ANCHOR_RATIOS)
        self.num_proposals = cfg.Model.RPN.NUM_PROPOSALS
        self.layers = self.init_layers(len(cfg.Model.RPN.ANCHOR_RATIOS))
        self.match_thresh = {'high': 0.6, 'low': 0.5}
        self.indices = self.init_indices(cfg.Train.BATCH_SIZE)
        self.iou_threshold = cfg.Model.RPN.NMS_IOU_THRESH
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.labels = [0, 1]
        self.num_sample = cfg.Model.RPN.NUM_SAMPLE

    def init_layers(self, num_anchors):
        layers = {}
        for i, hw in enumerate(self.input_shapes):
            layers[hw] = torch.nn.Sequential(
                nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.input_channels, num_anchors * 5, kernel_size=3, stride=1, padding=1)
            )
            self.add_module(f'conv_s{i}', layers[hw])
            # output channel: 15 = 3*5 = num anchors per scale x (tlbr + objectness)
        return layers

    def init_indices(self, batch_size):
        indices = [torch.ones(self.num_proposals, dtype=torch.int64) * b for b in range(batch_size)]
        return indices

    def forward(self, features, grtr=None):
        """
        :param features: list of [batch, self.input_channels, height/scale, width/scale]
        :return:
        """

        del features['neck_s5']
        logit_features = list()
        for hw, feature in zip(self.input_shapes, features.values()):
            logits = self.forward_layers(feature, hw)
            logit_features.append(logits)

        # (batch,3*5,hi,wi) -> (batch,5,3,hi,wi) -> (batch,5,3*hi*wi) -> (batch,5,sum(3*hi*wi)) -> (batch,sum(3*hi*wi),5)
        logit_features = self.merge_features(logit_features)
        proposals_aux = self.decode_features(logit_features, grtr)
        proposal = self.sort_proposals(proposals_aux)
        proposals = self.select_proposals(proposal)
        if self.training:
            proposals = self.sample_proposals(proposals, grtr)
        return proposals, proposals_aux

    def forward_layers(self, features, hw):
        logits = self.layers[hw](features)
        return logits

    def merge_features(self, logit_features):
        """
        :param logit_features: (batch,anchor*channel(5),hi,wi)
                -> (batch,hi,wi,anchor,channel(5)+anchor_index(1))
        :return:
        """
        merged_feature_logit = list()
        for feature_i, logit in enumerate(logit_features):
            batch, _, height, width = logit.shape
            logit = logit.reshape(batch, self.num_anchor, 5, height, width)
            logit = logit.permute(0, 3, 4, 1, 2)
            # [anchor]
            anchor_id = torch.tensor(list(range(feature_i * self.num_anchor, (feature_i + 1) * self.num_anchor)),
                                     device=self.device)
            # [batch,height,width,anchor,1]
            anchor_id = anchor_id.repeat(batch, height, width, 1).unsqueeze(-1)
            # [batch,height,width,anchor,channel(5+1)]
            logit = torch.cat([logit, anchor_id], dim=-1).view(batch, -1, logit.shape[-1] + 1)
            merged_feature_logit.append(logit)

        # merged_feature_logit = torch.cat(merged_feature_logit, dim=-1)
        # merged_feature_logit = merged_feature_logit.permute(0, 2, 1)
        return merged_feature_logit

    def decode_features(self, logit_features, grtr):
        """
        :param logit_features: (batch,numbox,channel)
        :param grtr:
        :return:
        """
        proposals = {'bbox2d': [], 'objectness': [], 'anchor_id': []}
        for logit_feature, anchors in zip(logit_features, grtr["anchors"]):
            bbox2d_logit, object_logits, anchor_id = logit_feature[..., :4], logit_feature[..., 4:5], logit_feature[...,
                                                                                                      5:6]
            bbox2d = mu.apply_box_deltas(anchors, bbox2d_logit)
            objectness = torch.sigmoid(object_logits)
            proposals['bbox2d'].append(bbox2d)
            proposals['objectness'].append(objectness)
            proposals['anchor_id'].append(anchor_id)

        # for key in proposals:
        #     proposals[key] = torch.cat(proposals[key], dim=1)
        return proposals

    def sort_proposals(self, proposals):
        ######
        cat_proposal = dict()
        for key in proposals:
            cat_proposal[key] = torch.cat(proposals[key], dim=1)
        # sort by score -> select top 3000 indices -> slice boxes, score, index
        bbox2d = cat_proposal['bbox2d']
        score = cat_proposal['objectness']
        anchor_id = cat_proposal['anchor_id']
        score_sort, sort_idx = torch.sort(score, dim=1, descending=True)

        sort = {'bbox2d': [], 'anchor_id': []}
        for i in range(bbox2d.shape[0]):
            sort['bbox2d'].append(bbox2d[i, sort_idx[i].squeeze(1)])
            sort['anchor_id'].append(anchor_id[i, sort_idx[i].squeeze(1)])
        bbox2d_sort = torch.stack(sort['bbox2d'])
        achor_id_sort = torch.stack(sort['anchor_id'])
        sorted_proposals = {'bbox2d': bbox2d_sort[:, : self.num_proposals],
                            'objectness': score_sort[:, : self.num_proposals],
                            'anchor_id': achor_id_sort[:, : self.num_proposals]}

        return sorted_proposals

    def select_proposals(self, proposals):
        """

        :param proposals: bbox2d, objectness, anchor_id
        :return: bbox2d, objectness, anchor_id
        """
        selected_proposals = {'bbox2d': [], 'objectness': [], 'anchor_id': []}
        for i, (bbox2d, score, anchor_id) in enumerate(zip(proposals['bbox2d'], proposals['objectness'],  proposals['anchor_id'])):
            keep = box_ops.batched_nms(bbox2d, score.view(-1), self.indices[i], self.iou_threshold)
            bbox2d = bbox2d[keep]
            score = score[keep]
            anchor_id = anchor_id[keep]
            if keep.numel() < self.num_proposals:
                padding = torch.zeros(self.num_proposals - keep.numel(), device=self.device).view(-1, 1)
                box_padding = torch.cat([padding] * 4, dim=-1)
                score = torch.cat([score, padding])
                anchor_id = torch.cat([anchor_id, padding])
                bbox2d = torch.cat([bbox2d, box_padding])
            selected_proposals['bbox2d'].append(bbox2d)
            selected_proposals['objectness'].append(score)
            selected_proposals['anchor_id'].append(anchor_id)

        for key in selected_proposals:
            selected_proposals[key] = torch.stack(selected_proposals[key], dim=0)
        return selected_proposals

    def sample_proposals(self, proposals, grtr):
        """
        match proposals with grtr
        sample positive to fixed fraction (P)
        append gt boxes (G)
        fill negatives to rest (N)
        1000 = P + N + G
        """
        gt_bbox2d = grtr['bbox2d']  # (batch, fix_num,4)
        proposal_gt_box = torch.cat([proposals['bbox2d'], grtr['bbox2d']], dim=1)  # (batch, num_proposals + fix_num, 4)
        proposal_gt_object = torch.cat([proposals['objectness'], grtr['object']],
                                       dim=1)  # (batch, num_proposals + fix_num, 1)
        proposal_gt_anchor_id = torch.cat([proposals['anchor_id'], grtr['anchor_id']],
                                          dim=1)  # (batch, num_proposals + fix_num, 1)
        proposals = {'bbox2d': [], 'objectness': [], 'anchor_id': []}

        for batch_index in range(proposal_gt_box.shape[0]):
            # (fix_num, num_proposals + fix_num)
            iou_matrix = uf.pairwise_iou(gt_bbox2d[batch_index], proposal_gt_box[batch_index])
            match_ious, match_inds = iou_matrix.max(dim=0)  # (num_proposals + fix_num)
            positive = torch.nonzero(match_ious > self.match_thresh['high'])
            negative = torch.nonzero(match_ious < self.match_thresh['low'])

            num_pos = int(self.num_sample * 0.25)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.num_sample - num_pos
            num_neg = min(negative.shape[0], num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            positive = positive[perm1].squeeze(1)
            negative = negative[perm2].squeeze(1)
            sampling = torch.cat([positive, negative])
            # append to proposals
            for key, proposal_one in zip(proposals, [proposal_gt_box, proposal_gt_object, proposal_gt_anchor_id]):
                proposals[key].append(proposal_one[batch_index, sampling])
        # stack on batch dimension
        for key, value in proposals.items():
            proposals[key] = torch.stack(value, dim=0)
        return proposals
