from abc import ABCMeta, abstractmethod
from torch import nn

from config import Config as cfg
from utils.util_class import ShapeSpec
# from utils.util_class import MyExceptionToCatch
# import model.model_util as mu

def backbone_factory(backbone, conv_kwargs):
    if backbone == "FPN":
        return FPN(conv_kwargs)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")


class BackboneBase:
    def __init__(self, conv_kwargs):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_kwargs)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_kwargs)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_kwargs)

    def residual(self, x, filters):
        short_cut = x
        conv = self.conv2d_k1(x, filters // 2)
        conv = self.conv2d(conv, filters)
        return short_cut + conv


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self,feature):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str: Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    # the properties below are not used any more

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}




