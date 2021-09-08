import torch
from torch import nn
import numpy as np

import config as cfg
from utils.util_class import ShapeSpec, MyExceptionToCatch
from model.backbone import ResNet, BasicStem, BottleneckBlock, make_stage
from model.neck import FPN, LastLevelMaxPool
from model.rpn import RPN, RPN_
from model.head import StandardROIHeads
from model.architecture import GeneralizedRCNN


def build_model(model_name, backbone_name, neck_name, rpn_name, head_name):
    backbone = backbone_factory(backbone_name)
    neck = neck_factory(neck_name, backbone)
    print(neck.output_shape())
    rpn = rpn_factory(rpn_name, neck.output_shape())
    # head = head_factory(head_name, neck.output_shape())
    ModelClass = select_model(model_name)
    model = ModelClass(backbone=backbone,neck=neck, rpn=rpn, head=None)
    # print(model)
    return model


def backbone_factory(backbone_name):
    """
    try except, split functions, specify inputs
    :param backbone_name:
    :param neck_name:
    :return:
    """
    if backbone_name == "ResNet":
        # need registration of new blocks/stems?
        input_shape = ShapeSpec(channels=len(cfg.Model.Structure.PIXEL_MEAN))
        norm = cfg.Model.Backbone.NORM
        stem = BasicStem(
            in_channels=input_shape.channels,
            out_channels=cfg.Model.RESNET.STEM_OUT_CHANNELS,
            norm=norm,
        )
        freeze_at = 0

        # fmt: off
        out_features = cfg.Model.RESNET.OUT_FEATURES
        depth = cfg.Model.RESNET.DEPTH
        num_groups = cfg.Model.RESNET.NUM_GROUPS
        width_per_group = cfg.Model.RESNET.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        in_channels = cfg.Model.RESNET.STEM_OUT_CHANNELS
        out_channels = cfg.Model.RESNET.RES2_OUT_CHANNELS
        stride_in_1x1 = cfg.Model.RESNET.STRIDE_IN_1X1
        res5_dilation = cfg.Model.RESNET.RES5_DILATION

        # fmt: on
        assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)
        num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
        stages = []

        # Avoid creating variables without gradients
        # It consumes extra memory and may cause allreduce to fail
        out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
        max_stage_idx = max(out_stage_idx)
        for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
            dilation = res5_dilation if stage_idx == 5 else 1
            first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
            stage_kargs = {"num_blocks": num_blocks_per_stage[idx], "first_stride": first_stride,
                           "in_channels": in_channels, "bottleneck_channels": bottleneck_channels,
                           "out_channels": out_channels, "num_groups": num_groups, "norm": norm,
                           "stride_in_1x1": stride_in_1x1, "dilation": dilation, "block_class": BottleneckBlock}

            blocks = make_stage(**stage_kargs)
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2

            if freeze_at >= stage_idx:
                for block in blocks:
                    block.freeze()
            stages.append(blocks)
        bottom_up = ResNet(stem, stages, out_features=out_features)
        return bottom_up
    else:
        raise MyExceptionToCatch(f"[backbone] EMPTY")


def neck_factory(neck_name, bottom_up):
    if neck_name == "FPN":
        in_features = cfg.Model.RESNET.OUT_FEATURES
        out_channels = cfg.Model.Neck.OUTPUT_CHANNELS
        neck = FPN(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.Model.Neck.NORM,
            top_block=LastLevelMaxPool(cfg.Model.Neck.OUT_FEATURES[-1]),
            fuse_type='sum',
        )
        return neck
    else:
        raise MyExceptionToCatch(f"[neck] EMPTY")


def rpn_factory(rpn_name, output_shape):
    if rpn_name == 'RPN_':
        return RPN_(output_shape)
    if rpn_name == 'RPN':
        return RPN(output_shape)
    else:
        raise MyExceptionToCatch(f"[rpn] EMPTY")


def head_factory(head_name, output_shape):
    if head_name == 'ROI':
        return StandardROIHeads(output_shape)
    else:
        raise MyExceptionToCatch(f"[head] EMPTY")


def select_model(model_name):
    if model_name == "RCNN":
        return GeneralizedRCNN
    else:
        raise MyExceptionToCatch(f"[model] EMPTY")
