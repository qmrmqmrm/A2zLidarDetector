import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

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
        self.num_proposals = (cfg.Model.RPN.NUM_PROPOSALS[0] if self.training else cfg.Model.RPN.NUM_PROPOSALS[1])
        self.match_thresh = cfg.Model.RPN.MATCH_THRESHOLD
        self.indices = self.init_indices(cfg.Train.BATCH_SIZE)
        self.iou_threshold = cfg.Model.RPN.NMS_IOU_THRESH
        self.score_threshold = cfg.Model.RPN.NMS_SCORE_THRESH
        self.labels = [0, 1]
        self.num_sample = cfg.Model.RPN.NUM_SAMPLE

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.bn = nn.BatchNorm2d(self.input_channels)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(self.input_channels, self.num_anchor, kernel_size=(1, 1), stride=(1, 1))
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            self.input_channels, self.num_anchor * 4, kernel_size=(1, 1), stride=(1, 1)
        )
        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def init_indices(self, batch_size):
        indices = [torch.ones(self.num_proposals, dtype=torch.int64) * b for b in range(batch_size)]
        return indices

    def forward(self, features, anchors, grtr=None):
        """

        :param features: list of [batch, self.input_channels, height/scale, width/scale]
        :param grtr:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4(tlbr), 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
        :return:
        proposals: {
                'bbox2d' : list(torch.Size([4, num_sample, 4(tlbr)]))
                'objectness' :list(torch.Size([4, num_sample, 1]))
                'anchor_id' :list(torch.Size([4, num_sample, 1]))
                 }
        proposals_aux :
                {
                'bbox2d' : list(torch.Size([4, height/stride* width/stride* anchor, 4(tlbr)]))
                'objectness' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
                'anchor_id' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
                }
        """

        del features['neck_s5']

        pred_objectness_logits, pred_anchor_deltas = self.forward_layers(features)

        logit_features = self.merge_features(pred_objectness_logits, pred_anchor_deltas)
        proposals_aux = self.decode_features(logit_features, anchors)
        with torch.no_grad():
            proposal = self.sort_proposals(proposals_aux)
            proposals = self.select_proposals(proposal)
            if grtr:
                proposals = self.sample_proposals(proposals, grtr)
        return proposals, proposals_aux

    def forward_layers(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features.values():
            t = F.relu(self.conv(x))
            t = self.bn(t)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    def merge_features(self, pred_objectness_logits, pred_anchor_deltas):
        """
        :param logit_features: (batch,anchor*channel(5),hi,wi)
                -> (batch,hi,wi,anchor,channel(5)+anchor_index(1))
        :return:
        """
        merged_feature_logit = list()
        for feature_i, (pred_objectness_logit, pred_anchor_delta) in enumerate(
                zip(pred_objectness_logits, pred_anchor_deltas)):
            batch, _, height, width = pred_anchor_delta.shape
            pred_anchor_delta = pred_anchor_delta.reshape(batch, self.num_anchor, 4, height, width)
            pred_anchor_delta = pred_anchor_delta.permute(0, 3, 4, 1, 2)
            pred_objectness_logit = pred_objectness_logit.reshape(batch, self.num_anchor, 1, height, width)
            pred_objectness_logit = pred_objectness_logit.permute(0, 3, 4, 1, 2)
            # [anchor]
            anchor_id = torch.tensor(list(range(feature_i * self.num_anchor, (feature_i + 1) * self.num_anchor)),
                                     device=self.device)
            # [batch,height,width,anchor,1]
            anchor_id = anchor_id.repeat(batch, height, width, 1).unsqueeze(-1)
            # [batch,height,width,anchor,channel(5+1)]
            logit = torch.cat([pred_anchor_delta, pred_objectness_logit, anchor_id], dim=-1).view(batch, -1, 6)
            merged_feature_logit.append(logit)

        # merged_feature_logit = torch.cat(merged_feature_logit, dim=-1)
        # merged_feature_logit = merged_feature_logit.permute(0, 2, 1)
        return merged_feature_logit

    def decode_features(self, logit_features, anchors):
        """
        :param logit_features: (batch,numbox,channel)
        :param anchors: [batch, height/stride, width/stride, anchor, yxwh + id] * features
        :return: proposals :
        {
        'bbox2d' : list(torch.Size([4, height/stride* width/stride* anchor, 4(tlbr)]))
        'objectness' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        'anchor_id' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        }
        """
        proposals = {'bbox2d': [], 'objectness': [], 'anchor_id': [], 'bbox2d_yxhw': [], 'object_logits': [],
                     'anchors': []}
        for logit_feature, anchors in zip(logit_features, anchors):
            bbox2d_logit, object_logits, anchor_id = logit_feature[..., :4], logit_feature[..., 4:5], logit_feature[...,
                                                                                                      5:6]

            bbox2d_yxhw = mu.apply_box_deltas(anchors, bbox2d_logit)
            bbox2d_tlbr = mu.convert_box_format_yxhw_to_tlbr(bbox2d_yxhw)  # tlbr
            object_logits_numpy = object_logits.to('cpu').detach().numpy()
            object_quant = np.quantile(object_logits_numpy, np.arange(0, 1.1, 0.1))
            print("object logits quantile:", object_quant)
            b, h, w, a, c = anchors[..., :-1].shape
            anchors = anchors[..., :-1].view(b, h * w * a, c)
            objectness = torch.sigmoid(object_logits)
            proposals['bbox2d'].append(bbox2d_tlbr)
            proposals['anchors'].append(anchors)
            proposals['bbox2d_yxhw'].append(bbox2d_yxhw)
            proposals['objectness'].append(objectness)
            proposals['anchor_id'].append(anchor_id)
            proposals['object_logits'].append(object_logits)

        # for key in proposals:
        #     proposals[key] = torch.cat(proposals[key], dim=1)
        return proposals

    def sort_proposals(self, proposals):
        """

        :param proposals:
        {
        'bbox2d' : list(torch.Size([4, height/stride* width/stride* anchor, 4(tlbr)]))
        'objectness' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        'anchor_id' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        }
        :return: sorted_proposals:
        {
        'bbox2d' : list(torch.Size([4, height/stride* width/stride* anchor, 4(tlbr)]))
        'objectness' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        'anchor_id' :list(torch.Size([4, height/stride* width/stride* anchor, 1]))
        'anchor' :list(torch.Size([4, height/stride* width/stride* anchor, 4]))
        }
        """
        ######
        cat_proposal = dict()
        for key in proposals:
            cat_proposal[key] = torch.cat(proposals[key], dim=1)
        # sort by score -> select top 3000 indices -> slice boxes, score, index
        bbox2d = cat_proposal['bbox2d']
        score = cat_proposal['objectness']
        anchors = cat_proposal['anchors']
        anchor_id = cat_proposal['anchor_id']

        score_numpy = score.to('cpu').detach().numpy()
        score_quant = np.quantile(score_numpy, np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.995, 0.999, 1]))
        print("score quantile:", score_quant)

        score_sort, sort_idx = torch.sort(score, dim=1, descending=True)
        # score_mask = score_sort > 0.3
        sort = {'bbox2d': [], 'anchor_id': [], 'anchors': []}
        for i in range(bbox2d.shape[0]):
            sort['bbox2d'].append(bbox2d[i, sort_idx[i].squeeze(1), :])
            sort['anchor_id'].append(anchor_id[i, sort_idx[i].squeeze(1)])  # * score_mask[i, :]
            sort['anchors'].append(anchors[i, sort_idx[i].squeeze(1)])  # * score_mask[i, :]
        bbox2d_sort = torch.stack(sort['bbox2d'])
        achor_id_sort = torch.stack(sort['anchor_id'])
        anchors_sort = torch.stack(sort['anchors'])
        sorted_proposals = {'bbox2d': bbox2d_sort,
                            'object': score_sort,
                            'anchors': anchors_sort,
                            'anchor_id': achor_id_sort}

        return sorted_proposals

    def select_proposals(self, proposals):
        """

        :param proposals:
        {
        'bbox2d' : (torch.Size([4, sum(height/stride* width/stride* anchor), 4(tlbr)]))
        'object' :(torch.Size([4, sum(height/stride* width/stride* anchor), 1]))
        'anchor_id' :(torch.Size([4, sum(height/stride* width/stride* anchor), 1]))
        'anchor' :(torch.Size([4, sum(height/stride* width/stride* anchor), 4(tlbr)]))
        }
        :return: selected_proposals :
        {
        'bbox2d' : list(torch.Size([4, num_proposals, 4(tlbr)]))
        'object' :list(torch.Size([4, num_proposals, 1]))
        'anchor_id' :list(torch.Size([4, num_proposals, 1]))
        'anchor' :list(torch.Size([4, num_proposals,  4(tlbr)]))
        }
        """
        batch, hwa, channel = proposals['bbox2d'].shape
        selected_proposals = {'bbox2d': [], 'object': [], 'anchor_id': [], 'anchors': []}

        for batch_idx in range(batch):
            proposal_dict = dict()
            score_mask = (proposals['object'][batch_idx] >= self.score_threshold).squeeze(-1)
            for key in proposals.keys():
                proposal_dict[key] = proposals[key][batch_idx, score_mask]
            keep = box_ops.nms(proposal_dict['bbox2d'], proposal_dict['object'].view(-1), self.iou_threshold)
            keep = keep[:self.num_proposals]
            for key in proposal_dict.keys():
                proposal_dict[key] = proposal_dict[key][keep]
            obj_num = proposal_dict['object'].numel()
            if obj_num < self.num_proposals:

                for key in proposal_dict.keys():
                    padding = torch.zeros(self.num_proposals - obj_num, device=self.device).view(-1, 1)
                    if key == 'bbox2d' or key == 'anchors':
                        padding = torch.cat([padding] * 4, dim=-1)
                    proposal_dict[key] = torch.cat([proposal_dict[key], padding])
            for key in proposal_dict.keys():
                selected_proposals[key].append(proposal_dict[key])

        for key in selected_proposals:
            selected_proposals[key] = torch.stack(selected_proposals[key], dim=0)

        return selected_proposals

    def sample_proposals(self, proposals, grtr):
        """

        :param proposals:
        {
        'bbox2d' : list(torch.Size([4, num_proposals, 4(tlbr)]))
        'objectness' :list(torch.Size([4, num_proposals, 1]))
        'anchor_id' :list(torch.Size([4, num_proposals, 1]))
        }
        :param grtr:
        :return:
        {
        'bbox2d' : list(torch.Size([4, num_sample, 4(tlbr)]))
        'objectness' :list(torch.Size([4, num_sample, 1]))
        'anchor_id' :list(torch.Size([4, num_sample, 1]))
        }
        """
        proposal_gt = dict()
        for key in proposals.keys():
            proposal_gt[key] = torch.cat([proposals[key], grtr[key]], dim=1)

        batch, num, channel = proposal_gt['bbox2d'].shape
        proposals_sample = {'bbox2d': [], 'object': [], 'anchor_id': [], 'anchors': []}
        for batch_index in range(batch):
            # (fix_num, num_proposals + fix_num)
            iou_matrix = uf.pairwise_iou(grtr['bbox2d'][batch_index], proposal_gt['bbox2d'][batch_index])
            match_ious, match_inds = iou_matrix.max(dim=0)  # (num_proposals + fix_num)
            positive = torch.nonzero(match_ious > self.match_thresh[1])
            negative = torch.nonzero(match_ious < self.match_thresh[0])

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
            for key in proposal_gt.keys():
                proposals_sample[key].append(proposal_gt[key][batch_index, sampling])
        # stack on batch dimension
        for key, value in proposals_sample.items():
            proposals_sample[key] = torch.stack(value, dim=0)
        return proposals_sample
