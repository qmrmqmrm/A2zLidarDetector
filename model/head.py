import torch
from torch import nn
from torch.nn import functional as F
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
        self.in_features = cfg.Model.Neck.OUT_FEATURES
        pooler_resolution = cfg.Model.Head.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.Model.Head.POOLER_SAMPLING_RATIO

        pooler_type = "ROIAlignV2"

        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=int((cfg.Model.Neck.OUT_FEATURES[-1].split('neck_s'))[1])
        )
        self.box_predictor = FastRCNNFCOutputHead(
            ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

    def forward(self, features, proposals):
        features = [features[f] for f in self.in_features]
        uf.print_structure('features',features)
        uf.print_structure('proposals',proposals)
        box_features = self.box_pooler(features, proposals["bbox2d"])
        # torch.Size([batch * 512, 256, 7, 7])
        pred_features = self.box_predictor(box_features)
        return pred_features


class FastRCNNFCOutputHead(nn.Module):
    def __init__(self, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        num_fc = cfg.Model.Head.NUM_FC
        fc_dim = cfg.Model.Head.FC_DIM
        fc_shape =int(np.prod(self._output_size))
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
