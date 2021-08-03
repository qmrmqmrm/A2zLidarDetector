

def build_model(model_name, backbone_name, neck_name, rpn_name, head_name):
    backbone = backbone_factory(backbone_name, neck_name)
    rpn = rpn_factory(rpn_name)
    head = head_factory(head_name)
    ModelClass = select_model(model_name)
    model = ModelClass(backbone=backbone, rpn=rpn, head=head)


def backbone_factory(backbone_name, neck_name):
    # if backbone_name == "FPN":
    #     return FPN()
    pass

    def build_resnet_fpn_backbone(input_shape=None):
        """
        Args:
            cfg: a detectron2 CfgNode

        Returns:
            backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
        """
        bottom_up = build_resnet_backbone(input_shape)
        in_features = cfg.Model.RESNET.OUT_FEATURES
        out_channels = cfg.Model.FPN.OUTPUT_CHANNELS
        backbone = FPN(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.Model.FPN.NORM,
            top_block=LastLevelMaxPool(cfg.Model.ROI_HEADS.INPUT_FEATURES[-1]),
            fuse_type=cfg.Model.FPN.FUSE_TYPE,
        )
        return backbone



def select_model(model_name):
    # if model_name == "RCNN":
    #     return RCNN
    pass


