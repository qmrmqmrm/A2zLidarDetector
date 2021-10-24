import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from typing import Dict
import numpy as np

from utils.util_class import ShapeSpec
import utils.util_function as uf
import config as cfg
from model.submodules.weight_init import c2_xavier_fill
import model.submodules.model_util as mu


class FastRCNNHead(nn.Module):
    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super(FastRCNNHead, self).__init__()
        self.device = cfg.Hardware.DEVICE
        self.in_features = cfg.Model.Neck.OUT_FEATURES
        self.pooler_resolution = cfg.Model.Head.POOLER_RESOLUTION
        self.pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        self.sampling_ratio = cfg.Model.Head.POOLER_SAMPLING_RATIO
        self.aligned = cfg.Model.Head.ALIGNED
        self.num_sample = cfg.Model.RPN.NUM_SAMPLE

        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_predictor = FastRCNNFCOutputHead(channels=in_channels, height=self.pooler_resolution,
                                                  width=self.pooler_resolution)

    def forward(self, features, proposals):
        """

        :param features:
        :param proposals:
        {
        'bbox2d' : list(torch.Size([4, num_sample, 4(tlbr)]))
        'objectness' :list(torch.Size([4, num_sample, 1]))
        'anchor_id' :list(torch.Size([4, num_sample, 1]))
        }
        :return:
        'head_output' : torch.Size([4, 512, 93])
        """
        features = [features[f] for f in self.in_features]
        # torch.Size([batch * 512, 256, 7, 7])
        box_features = self.box_pooler(features, proposals)
        # torch.Size([1024, 93]
        pred_features = self.box_predictor(box_features['xs'])

        batch, numsample, ch = proposals['bbox2d'].shape

        pred_features = pred_features.view(batch, numsample, -1)
        pred_features = self.decode(pred_features, proposals['anchors'], box_features['strides'])
        # decode()

        # torch.Size([batch, 512, 93]

        return pred_features

    def box_pooler(self, features, proposals):
        bbox2d = proposals['bbox2d']
        zeros = torch.zeros((bbox2d.shape[0], bbox2d.shape[1], 1), device=self.device)
        bbox2d = torch.cat([zeros, bbox2d], dim=-1)
        anchor_id = proposals['anchor_id']
        feature_aligned = {'xs': [], 'strides': []}
        for batch in range(features[0].shape[0]):
            xs = list()
            boxinds = list()
            box_list = list()
            strides = list()
            for i, (scale, feature) in enumerate(zip(self.pooler_scales, features)):
                # feature (batch, c, h, w), anchor_id (batch, numbox, 1)
                boxind, ch = torch.where(anchor_id[batch] // 3 == i)
                stride = torch.ones((1)) * i
                stride = stride.repeat(len(boxind))
                box = bbox2d[batch, boxind]
                x = roi_align(feature[batch:batch + 1], box, self.pooler_resolution, scale,
                              self.sampling_ratio, self.aligned)
                xs.append(x)
                boxinds.append(boxind)
                box_list.append(box)
                strides.append(stride)

            xs = torch.cat(xs, dim=0)
            strides = torch.cat(strides, dim=0)
            box_ = torch.cat(box_list, dim=0)
            boxinds = torch.cat(boxinds, dim=0)
            sort_box, sort_inds = torch.sort(boxinds)
            xs = xs[sort_inds]
            strides = strides[sort_inds]
            box_sort = box_[sort_inds]
            assert torch.all((box_sort - bbox2d[batch]) == 0)
            feature_aligned['xs'].append(xs)
            feature_aligned['strides'].append(strides)

        # [batch * numbox, channel, pooler_resolution, pooler_resolution]
        for key in feature_aligned.keys():
            feature_aligned[key] = torch.cat(feature_aligned[key], dim=0)
        return feature_aligned

    def decode(self, pred, anchors, strides):
        num_classes = cfg.Model.Structure.NUM_CLASSES
        loss_channel = cfg.Model.Structure.LOSS_CHANNEL
        sliced_features = {}
        last_channel = 0
        for loss, dims in loss_channel.items():
            slice_dim = last_channel + num_classes * dims
            if loss == 'category':
                slice_dim = slice_dim + 1
            sliced_features[loss] = pred[..., last_channel:slice_dim]
            last_channel = slice_dim
        # pred_slices = uf.merge_and_slice_features(pred)  # b, n, 18
        pred = uf.slice_class(sliced_features)  # b, n, 3, 6
        bbox3d_yxlw_delta = pred['bbox3d'][..., :-2]
        bbox3d_zh = pred['bbox3d'][..., -2:]
        bbox2d_yxlw = list()
        for i in range(3):
            bbox3d_split_cate = bbox3d_yxlw_delta[:, :, i, :].squeeze(-2)
            bbox2d_yxlw_per_cate = mu.apply_box_deltas(anchors, bbox3d_split_cate, strides)
            bbox2d_yxlw.append(bbox2d_yxlw_per_cate.unsqueeze(-2))
        bbox2d_yxlw = torch.cat(bbox2d_yxlw, dim=-2)
        bbox3d = torch.cat([bbox2d_yxlw, bbox3d_zh], dim=-1)
        pred['bbox3d'] = bbox3d
        pred['category'] = pred['category'].squeeze(-1)
        return pred


class FastRCNNFCOutputHead(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self._output_size = (channels, height, width)
        num_fc = cfg.Model.Head.NUM_FC
        fc_dim = cfg.Model.Head.FC_DIM
        self.fcs = []
        self.bns = []
        for k in range(num_fc):
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            bn = nn.BatchNorm1d(fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("bn{}".format(k + 1), bn)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self.bns.append(bn)
            self._output_size = fc_dim
        for layer in self.fcs:
            c2_xavier_fill(layer)
        input_size = cfg.Model.Head.FC_DIM
        num_classes = cfg.Model.Structure.NUM_CLASSES
        box_dim = cfg.Model.Structure.BOX_DIM
        bins = cfg.Model.Structure.VP_BINS
        out_channels = num_classes * (1 + box_dim + (2 * bins)) + 1
        self.pred_layer = nn.Linear(input_size, out_channels)
        nn.init.xavier_normal_(self.pred_layer.weight)
        nn.init.constant_(self.pred_layer.weight, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        for layer, bn in zip(self.fcs, self.bns):
            x = F.relu(bn(layer(x)))

        x = self.pred_layer(x)
        return x
