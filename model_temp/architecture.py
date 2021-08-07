import torch
from torch import nn
import copy
import numpy as np
import cv2

from config import Config as cfg
from utils.image_list import ImageList


class ModelBase(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(ModelBase, self).__init__()
        self.device = torch.device("cuda")
        self.backbone = backbone
        self.proposal_generator = rpn
        self.roi_heads = head
        self.to(self.device)

        pass

    def forward(self, batched_inputs):
        # ...
        # return {"backbone_l": backbone_l, "backbone_m": backbone_m, "backbone_s": backbone_s,
        #         "boxreg": boxreg, "category": catetory, "validbox": valid_box}
        pass


class GeneralizedRCNN(ModelBase):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, backbone, rpn, head):
        super().__init__(backbone, rpn, head)



        assert len(cfg.Model.Structure.PIXEL_MEAN) == len(cfg.Model.Structure.PIXEL_STD)
        num_channels = len(cfg.Model.Structure.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.Model.Structure.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.Model.Structure.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)

        gt_instances = list()
        for batched_input in batched_inputs:
            gt_instance = dict()
            for key ,value in batched_input.items():
                if 'gt_' in key:
                    gt_instance.update({key: value})
            gt_instances.append(gt_instance)

        features = self.backbone(images.tensor)
        rpn_proposals, loss_instances = self.proposal_generator(images, features)
        pred, head_proposals = self.roi_heads(images, features, rpn_proposals, gt_instances)
        pred['rpn_proposals'] = rpn_proposals
        pred.update(loss_instances)
        pred['gt_instances'] = gt_instances
        pred['head_proposals'] = head_proposals
        # pred keys :  dict_keys(['scores', 'proposal_deltas', 'viewpoint_scores', 'viewpoint_residuals', 'height_scores', 'proposals', 'loss_instances'])
        # print("pred_anchor_deltas",loss_instances['pred_anchor_deltas'])


        return pred

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width,
                                         rotated_box_training=self.rotated_box_training)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [x.permute(2, 0, 1) for x in images]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class YOLO(ModelBase):
    pass
