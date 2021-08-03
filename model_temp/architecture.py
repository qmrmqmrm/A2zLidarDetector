import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, backbone, rpn, head):
        pass

    def forward(self):
        # ...
        # return {"backbone_l": backbone_l, "backbone_m": backbone_m, "backbone_s": backbone_s,
        #         "boxreg": boxreg, "category": catetory, "validbox": valid_box}
        pass


class GeneralizedRCNN(ModelBase):
    pass


class YOLO(ModelBase):
    pass
