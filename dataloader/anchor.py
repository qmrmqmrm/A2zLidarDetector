import math
import torch

import config as cfg
import utils.util_function as uf


class Anchor:
    def __init__(self, dataset_name):
        self.aspect_ratio = cfg.Model.RPN.ANCHOR_RATIOS
        self.anc_sizes = cfg.Model.RPN.ANCHOR_SIZES
        self.strides = cfg.Model.RPN.OUT_SCALES
        self.feature_shapes = [cfg.get_img_shape("HW", dataset_name, s) for s in cfg.Model.RPN.OUT_SCALES]
        self.device = [cfg.Hardware.DEVICE, 'cpu'][1]

    def __call__(self):
        """
        :return: list of feature maps that contain anchor boxes [(height/scale, width/scale, anchor, 5(yxhw+anchor_id)) for in scales]
                boxes are in yxhw format
        """
        anchors = []
        for feature_i, (hw_shape, size, stride) in enumerate(zip(self.feature_shapes, self.anc_sizes, self.strides)):
            anchor_per_scale = len(self.aspect_ratio)
            zeros = torch.zeros((hw_shape[0], hw_shape[1], anchor_per_scale, 2), device=self.device)
            anc_area = size ** 2
            anchor_hws = [[math.sqrt(anc_area * aratio), math.sqrt(anc_area / aratio)] for aratio in self.aspect_ratio]
            anchor_hws = torch.tensor(anchor_hws, device=self.device)
            anchor_hws = anchor_hws.view(1, 1, 3, 2)
            anchor_hws = zeros + anchor_hws
            # range of top left pixel center is (0~1)
            # if stride=4, feature cell range is (0~4) and its center is 2
            pos_y = torch.arange(0, hw_shape[0] * stride, step=stride, dtype=torch.float32, device=self.device)
            pos_x = torch.arange(0, hw_shape[1] * stride, step=stride, dtype=torch.float32, device=self.device)
            mesh_y, mesh_x = torch.meshgrid(pos_y, pos_x)
            pos_yxs = torch.stack([mesh_y, mesh_x], -1).view((hw_shape[0], hw_shape[1], 1, 2))
            pos_yxs = torch.tile(pos_yxs, (anchor_per_scale, 1))
            # [anchor]
            anchor_id = torch.tensor(list(range(feature_i * anchor_per_scale, (feature_i + 1) * anchor_per_scale)),
                                     device=self.device)
            # [batch,height,width,anchor,1]
            anchor_id = anchor_id.repeat(hw_shape[0], hw_shape[1], 1).unsqueeze(-1)
            anchor_box2d = torch.cat((pos_yxs, anchor_hws, anchor_id), dim=-1)
            anchors.append(anchor_box2d)
        return anchors
