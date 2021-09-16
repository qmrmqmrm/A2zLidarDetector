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
from model.submodules.poolers import ROIPooler


class Head(nn.Module):
    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super(Head, self).__init__()

    def forward(self, features, proposals):
        raise NotImplementedError()


class FastRCNNHead(Head):
    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super(FastRCNNHead, self).__init__(input_shape)
        self.device = cfg.Hardware.DEVICE
        self.in_features = cfg.Model.Neck.OUT_FEATURES
        self.pooler_resolution = cfg.Model.Head.POOLER_RESOLUTION
        self.pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        self.sampling_ratio = cfg.Model.Head.POOLER_SAMPLING_RATIO
        self.aligned = cfg.Model.Head.ALIGNED

        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_predictor = FastRCNNFCOutputHead(
            ShapeSpec(channels=in_channels, height=self.pooler_resolution, width=self.pooler_resolution)
        )

    def forward(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, proposals)
        # torch.Size([batch * 512, 256, 7, 7])
        uf.print_structure('bboxfeature',box_features)
        pred_features = self.box_predictor(box_features)
        return pred_features

    def box_pooler(self, features, proposals):
        bbox2d = proposals['bbox2d']
        anchor_id =proposals['anchor_id']

        batch = bbox2d.shape[0]
        xs = list()
        batch_to_box = torch.tensor([[0], [1]], device=self.device)
        batch_to_box = torch.repeat_interleave(batch_to_box, bbox2d.shape[1], dim=1).unsqueeze(-1)
        bbox2d_with_batch = torch.cat([bbox2d, batch_to_box], dim=-1).view(-1, 5)
        for scale, feature in zip(self.pooler_scales, features):

            x = roi_align(feature, bbox2d_with_batch, self.pooler_resolution, scale,
                          self.sampling_ratio, self.aligned)
            xs.append(x)
        uf.print_structure('xs', xs)
        return xs

class FastRCNNFCOutputHead(nn.Module):
    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        num_fc = cfg.Model.Head.NUM_FC
        fc_dim = cfg.Model.Head.FC_DIM
        fc_shape = int(np.prod(self._output_size))
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(fc_shape, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim
        for layer in self.fcs:
            c2_xavier_fill(layer)

        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        num_classes = cfg.Model.Structure.NUM_CLASSES
        box_dim = 6
        bins = cfg.Model.Structure.VP_BINS
        out_channels = num_classes * (1 + box_dim + (2 * bins))
        self.pred_layer = nn.Linear(input_size, out_channels)
        nn.init.xavier_normal_(self.pred_layer.weight)
        nn.init.constant_(self.pred_layer.weight, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        x = self.pred_layer(x)
        return x
