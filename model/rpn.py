import torch
from torch import nn
from torchvision.ops import boxes as box_ops

from model.submodules.box_regression import Box2BoxTransform
import utils.util_function as uf
import config as cfg


class RPN(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        self.device = cfg.Hardware.DEVICE
        self.input_shapes = [cfg.get_img_shape("HW", dataset_name, scale) for scale in cfg.Model.Neck.OUT_SCALES]
        self.input_channels = cfg.Model.Neck.OUTPUT_CHANNELS
        self.num_anchor = len(cfg.Model.RPN.ANCHOR_RATIOS)
        self.layers = self.init_layers(len(cfg.Model.RPN.ANCHOR_RATIOS))
        self.thresholds = [-float("inf"), 0.5, float("inf")]
        self.indices = self.init_indices(cfg.Train.BATCH_SIZE, len(cfg.Model.RPN.ANCHOR_RATIOS))
        self.iou_threshold = cfg.Model.RPN.NMS_IOU_THRESH
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.labels = [0, 1]

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

    def init_indices(self, batch_size, num_anchors):
        num_cells = 0
        for hw in self.input_shapes:
            num_cells += hw[0] * hw[1] * num_anchors

        # [batch * anchor * sum(hi*wi)]
        indices = [torch.ones(3000, dtype=torch.int64) * b for b in range(batch_size)]
        # indices = torch.cat([torch.ones(num_cells, dtype=torch.int64) * b for b in range(batch_size)], dim=0)
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
        proposals = self.decode_features(logit_features, grtr)
        proposals = self.select_proposals(proposals)
        if self.training:
            proposals = self.sample_proposals(proposals, grtr)
        return proposals

    def forward_layers(self, features, hw):
        logits = self.layers[hw](features)
        return logits

    def merge_features(self, logit_features):
        """
        (batch,3*5,hi,wi) -> (batch,5,3,hi,wi) -> (batch,5,3*hi*wi) -> (batch,5,sum(3*hi*wi)) -> (batch,sum(3*hi*wi),5)
        :param logit_features:
        :return:
        """
        merged_feature_logit = list()
        for logit in logit_features:
            logit_shape = logit.shape
            batch = logit_shape[0]
            logit = logit.reshape(batch, 5, -1)
            merged_feature_logit.append(logit)

        merged_feature_logit = torch.cat(merged_feature_logit, dim=-1)
        merged_feature_logit = merged_feature_logit.permute(0, 2, 1)
        return merged_feature_logit

    def decode_features(self, logit_features, grtr):
        """

        :param logit_features:
        :param grtr:
        :return:
        """
        bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, object_logits = torch.chunk(logit_features, 5, -1)
        bbox2d_delta = torch.cat([bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4], dim=-1)
        bbox2d_logit = list()
        for anchor_per_image, bbox2d_per_image in zip(grtr['anchors'], bbox2d_delta):
            bbox2d = self.box2box_transform.apply_deltas(bbox2d_per_image, anchor_per_image)
            bbox2d = bbox2d.unsqueeze(0)
            bbox2d_logit.append(bbox2d)
        bbox2d_logit = torch.cat(bbox2d_logit, dim=0)

        proposals = {'bbox2d_logit': bbox2d_logit, 'object_logits': object_logits}
        return proposals

    def select_proposals(self, proposals):
        """
        :param bbox2d: (batch, sum(anchor*hi*wi), 4(tlbr))
        :param score: (batch, sum(anchor*hi*wi), 1(objectness))
        :return:
        """

        """
        example:
        bbox2d [2, 100, 4]
        keep = [11, 50, 77, 87(x), 141, 170]
        numout = 3
        keep -> [[11, 50, 77], [141, 70, -1]]
        mask = keep >= 0 (2, 5, 1)
        keep *= mask
        bbox2d: [2, 5, 4] * mask
        score: [2, 5, 1] * mask
        """
        # sort by score -> select top 3000 indices -> slice boxes, score, index
        bbox2d = proposals['bbox2d_logit']
        score = proposals['object_logits']
        score_sort, sort_idx = torch.sort(score, dim=1, descending=True)

        bbox2d_sort = list()
        for i in range(bbox2d.shape[0]):
            bbox2d_sort.append(bbox2d[i, sort_idx[i].squeeze(1)])
        bbox2d_sort = torch.cat(bbox2d_sort).view(bbox2d.shape[0], -1, 4)
        score_sort = score_sort[:, :3000]
        bbox2d_sort = bbox2d_sort[:, :3000]
        bbox2d_ = list()
        score_ = list()
        for i, (bbox2d, score) in enumerate(zip(bbox2d_sort, score_sort)):
            keep = box_ops.batched_nms(bbox2d, score.view(-1), self.indices[i], self.iou_threshold)
            bbox2d = bbox2d[keep]
            score = score[keep]
            if keep.numel() < 3000:
                padding = torch.ones(3000 - keep.numel(), device=self.device) * -1
                bbox2d = torch.cat([bbox2d, torch.cat([padding] * 4).view(-1, 4)])
                score = torch.cat([score, padding.view(-1, 1)])

            bbox2d_.append(bbox2d)
            score_.append(score)
        bbox2d = torch.cat(bbox2d_).view(bbox2d_sort.shape[0], bbox2d_sort.shape[1], bbox2d_sort.shape[2])
        score = torch.cat(score_).view(score_sort.shape[0], score_sort.shape[1], score_sort.shape[2])

        return {"bbox2d": bbox2d, "score": score}

    def sample_proposals(self, proposals, grtr):
        """
        match proposals with grtr
        sample positive to fixed fraction (P)
        append gt boxes (G)
        fill negatives to rest (N)
        1000 = P + N + G
        """
        bbox2d = grtr['bbox2d']
        category = grtr['category'].squeeze(-1)
        proposal_gt_box = torch.cat([proposals['bbox2d'], grtr['bbox2d']], dim=1)
        proposal_gt_object = torch.cat([proposals['score'], grtr['object']], dim=1)
        bbox = list()
        score = list()
        for i in range(proposal_gt_box.shape[0]):
            match_quality_matrix = pairwise_iou(bbox2d[i], proposal_gt_box[i])
            matched_vals, matches_inds = match_quality_matrix.max(dim=0)
            match_labels = matches_inds.new_full(matches_inds.size(), 1, dtype=torch.int8)
            for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
                low_high = (matched_vals >= low) & (matched_vals < high)
                match_labels[low_high] = l
            gt_category = category[i, match_labels.long()]
            gt_category[match_labels == 0] = 3
            gt_category[match_labels == -1] = -1

            positive = torch.nonzero((gt_category != -1) & (gt_category != 3))
            negative = torch.nonzero(gt_category == 3)
            num_pos = int(512 * 0.25)

            num_pos = min(positive.numel(), num_pos)
            num_neg = 512 - num_pos

            num_neg = min(negative.shape[0], num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            positive = positive[perm1].squeeze(1)
            negative = negative[perm2].squeeze(1)
            sampling = torch.cat([positive, negative])

            box_pos_per_image = proposal_gt_box[i, sampling]
            score_per_image = proposal_gt_object[i, sampling]
            score.append(score_per_image)
            bbox.append(box_pos_per_image)
        bbox = torch.cat(bbox).reshape(grtr['bbox2d'].shape[0], -1, 4)
        score = torch.cat(score).reshape(grtr['bbox2d'].shape[0], -1, 1)
        proposals = {'bbox2d': bbox, 'score': score}
        return proposals
        #
        #
        #
        #     ious.append(iou)
        # match_quality_matrix = torch.cat(ious).view(gt_bbbox2d.shape[0], gt_bbbox2d.shape[1], -1)
        #
        # matched_vals, matches_inds = match_quality_matrix.max(dim=1)
        #
        # match_labels = matches_inds.new_full(matches_inds.size(), 1, dtype=torch.int8)
        # for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
        #     low_high = (matched_vals >= low) & (matched_vals < high)
        #     match_labels[low_high] = l
        #
        # gt_category = gt_category[match_labels.long()]
        # gt_category[match_labels == 0] = 3
        # gt_category[match_labels == -1] = -1
        #
        # positive = torch.nonzero((gt_category != -1) & (gt_category != 3))
        # negative = torch.nonzero(gt_category == 3)
        # num_pos = int(1024 * 0.25)
        #
        # num_pos = min(positive.shape[0], num_pos)
        # num_neg = 1024 - num_pos
        #
        # num_neg = min(negative.shape[0], num_neg)
        # perm1 = torch.randperm(positive.shape[0], device=positive.device)[:num_pos]
        # perm2 = torch.randperm(negative.shape[0], device=negative.device)[:num_neg]
        # positive = positive[perm1, :]
        # negative = negative[perm2, :]


def pairwise_iou(boxes1, boxes2):
    # boxes1 = boxes1.to(DEVICE)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2],
                                                                             boxes2[:, :2])  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]

    iou = torch.where(intersection > 0, intersection / (area1[:, None] + area2 - intersection),
                      torch.zeros(1, dtype=intersection.dtype, device=intersection.device)
                      )
    return iou
