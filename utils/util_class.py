import torch

from collections import namedtuple
import config as cfg
from model.submodules.matcher import Matcher
from model.submodules.box_regression import Box2BoxTransform
from utils.util_function import subsample_labels, pairwise_iou


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class MyExceptionToCatch(Exception):
    def __init__(self, msg):
        super().__init__(msg)
