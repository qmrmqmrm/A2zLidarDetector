import torch
import torch.nn.functional as F
import numpy as np
import math
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms

from config import Config as cfg
from model.submodules.box_regression import Box2BoxTransform


class Inference:
    def __init__(self, pred):
        """
        :param features:
        :param pred:
                {# header outputs
                'head_class_logits': torch.Size([batch * 512, 4]) pred_class_logits
                'bbox_3d_logits': torch.Size([batch * 512, 12]) pred_proposal_deltas
                'head_yaw_logits': torch.Size([batch * 512, 36]) viewpoint_logits
                'head_yaw_residuals': torch.Size([batch * 512, 36]) viewpoint_residuals
                'head_height_logits': torch.Size([batch * 512, 6]) height_logits

                'head_proposals': [{'proposal_boxes': torch.Size([512, 4])
                                    'objectness_logits': torch.Size([512])
                                    'category': torch.Size([512, 1])
                                    'bbox2d': torch.Size([512, 4])
                                    'bbox3d': torch.Size([512, 6])
                                    'object': torch.Size([512, 1])
                                    'yaw': torch.Size([512, 2])} * batch]

                # rpn outputs
                'rpn_proposals': [{'proposal_boxes': torch.Size([2000, 4]),
                                  'objectness_logits': torch.Size([2000])} * batch]

                'pred_objectness_logits' : [torch.Size([batch, 557568(176 * 352 * 9)]),
                                            torch.Size([batch, 139392(88 * 176 * 9)]),
                                            torch.Size([batch, 34848(44 * 88 * 9)])]

                'pred_anchor_deltas' : [torch.Size([batch, 557568(176 * 352 * 9), 4]),
                                        torch.Size([batch, 139392(88 * 176 * 9), 4]),
                                        torch.Size([batch, 34848(44 * 88 * 9), 4])]

                'anchors' : [torch.Size([557568(176 * 352 * 9), 4])
                             torch.Size([139392(88 * 176 * 9), 4])
                             torch.Size([34848(44 * 88 * 9), 4])]
                }
        """
        head_proposals = pred['head_proposals']
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.num_preds_per_image = 512
        self.pred_class_logits = pred['head_class_logits']
        self.pred_proposal_deltas = pred['bbox_3d_logits']
        self.smooth_l1_beta = 0.0
        self.viewpoint_logits = pred['head_yaw_logits']
        self.viewpoint = True if pred['head_yaw_logits'] is not None else False
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.viewpoint_res = True if pred['head_yaw_residuals'] is not None else False
        self.viewpoint_res_logits = pred['head_yaw_residuals']
        self.rotated_box_training = cfg.Model.Structure.ROTATED_BOX_TRAINING
        self.height_logits = pred['head_height_logits']
        self.height_training = True if pred['head_height_logits'] is not None else False
        self.weights_height = cfg.Model.Structure.WEIGHTS_HEIGHT

        box_type = type(head_proposals[0]['proposal_boxes'])
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = torch.cat([p['proposal_boxes'] for p in head_proposals])
        # assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [[700, 1400] for x in head_proposals]

        self.gt_boxes = torch.cat([p['bbox3d'] for p in head_proposals])
        self.gt_classes = torch.cat([p['gt_category'] for p in head_proposals], dim=0)
        self.gt_viewpoint = torch.cat([p['yaw'] for p in head_proposals], dim=0)
        self.gt_viewpoint_rads = torch.cat([p['yaw_rads'] for p in head_proposals], dim=0)
        self.gt_height = torch.cat([p['bbox3d'][:, - 2:] for p in head_proposals], dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        print(type(boxes))
        print(boxes[0].shape)
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        viewpoint = self.predict_viewpoint() if self.viewpoint else None
        viewpoint_residual = self.predict_viewpoint_residual() if self.viewpoint_res else None
        height = self.predict_height() if self.height_training else None

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, self.vp_bins, viewpoint,
            viewpoint_residual, self.rotated_box_training, height
        )

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
            self.rotated_box_training
        )
        print(boxes.view(num_pred, K * B).shape)
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def predict_viewpoint(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted orientation probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.viewpoint_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def predict_viewpoint_residual(self):
        num_pred = len(self.proposals)
        vp_residuals = self.apply_vp_deltas()
        return vp_residuals.split(self.num_preds_per_image, dim=0)

    def apply_vp_deltas(self):
        deltas = self.viewpoint_res_logits
        assert torch.isfinite(deltas).all().item()

        bin_dist = np.linspace(-math.pi, math.pi, self.vp_bins + 1)
        bin_res = (bin_dist[1] - bin_dist[0]) / 2.
        bin_dist = bin_dist - bin_res
        bin_dist = np.tile(bin_dist[:-1], self.pred_class_logits.shape[1] - 1)
        src_vp_res = torch.tensor(bin_dist, dtype=torch.float32).to(self.pred_proposal_deltas.device)

        wvp = np.trunc(1 / bin_res)

        dvp = deltas / wvp
        pred_vp_res = dvp + src_vp_res + bin_res
        pred_vp_res[pred_vp_res < -math.pi] += 2 * math.pi

        return pred_vp_res

    def predict_height(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic heights and elevations
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is 2
        """
        # pdb.set_trace()
        num_pred = len(self.proposals)
        B = 2
        K = self.height_logits.shape[1] // B
        heights = self.apply_h_deltas()
        return heights.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def apply_h_deltas(self):
        deltas = self.height_logits
        assert torch.isfinite(deltas).all().item()
        src_heights = torch.tensor([130.05, 149.6, 147.9]).to(deltas.device)  # Without background class?

        wh, wg, wz = self.weights_height
        dh = deltas[:, 0::2] / wh
        # dg = deltas[:, 1::2] / wg
        dz = deltas[:, 1::2] / wz

        pred_h = torch.exp(dh) * src_heights  # Every class multiplied by every mean height
        # pred_g = dg
        pred_z = dz * src_heights + src_heights / 2.

        pred_height = torch.zeros_like(deltas)
        pred_height[:, 0::2] = pred_h
        # pred_height[:, 1::2] = pred_g
        pred_height[:, 1::2] = pred_z
        return pred_height


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, vp_bins: int = None,
                        viewpoint: bool = None, viewpoint_residual: bool = None, rotated_box_training: bool = False,
                        height: bool = None):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
            :param viewpoint:
            :param viewpoint_residual:
    """

    viewpoint = [None] * boxes[0].shape[0] if viewpoint is None else viewpoint
    viewpoint_residual = [None] * boxes[0].shape[0] if viewpoint_residual is None else viewpoint_residual
    height = [None] * boxes[0].shape[0] if height is None else height
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vp_bins, vp,
            vp_res, rotated_box_training, h
        )
        for scores_per_image, boxes_per_image, image_shape, vp, vp_res, h in
        zip(scores, boxes, image_shapes, viewpoint, viewpoint_residual, height)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, vp_bins=None, vp=None, vp_res=None,
        rotated_box_training=False, h=None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = boxes.reshape(-1, 4)

    height, weight = image_shape
    x1 = boxes[:, 0].clamp(min=0, max=weight)
    y1 = boxes[:, 1].clamp(min=0, max=height)
    x2 = boxes[:, 2].clamp(min=0, max=weight)
    y2 = boxes[:, 3].clamp(min=0, max=height)
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    boxes = boxes.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    print('num_bbox_reg_classes')
    print(num_bbox_reg_classes)
    if num_bbox_reg_classes == 1:
        print('num_bbox_reg_classes == 1')
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        print('num_bbox_reg_classes != 1')
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    if not rotated_box_training or len(boxes) == 0:
        print('if not rotated_box_training or len(boxes) == 0')
        keep = box_ops.batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        print('if not rotated_box_training or len(boxes) == 0 else')
        # BBox with encoding ctr_x,ctr_y,w,l
        if vp is not None and vp_bins is not None:
            print('vp is not None and vp_bins is not None')
            _vp = vp.view(-1, num_bbox_reg_classes, vp_bins)  # R x C x bins
            _vp = _vp[filter_mask]
            if len(_vp) > 0:
                print('if len(_vp) > 0')
                _, vp_max = torch.max(_vp, 1)
                vp_filtered = vp_max
                if vp_res is not None:
                    print('if vp_res is not None')
                    _vp_res = vp_res.view(-1, num_bbox_reg_classes, vp_bins)
                    _vp_res = _vp_res[filter_mask]
                    vp_res_filtered = list()
                    for i, k in enumerate(vp_max):
                        vp_res_filtered.append(_vp_res[i, k])
                else:
                    print('if vp_res is not None else')
                    vp_filtered = _vp
            rboxes = []
            for i in range(boxes.shape[0]):
                box = boxes[i]
                angle = anglecorrection(vp_res_filtered[i] * 180 / math.pi).to(box.device)
                box = torch.cat((box, angle))
                rboxes.append(box)
            rboxes = torch.cat(rboxes).reshape(-1, 5).to(vp_filtered.device)
            # keep = nms_rotated(rboxes, scores, nms_thresh)

            # need check
            max_coordinate = (
                    torch.max(rboxes[:, 0], rboxes[:, 1]) + torch.max(rboxes[:, 2], rboxes[:, 3]) / 2
            ).max()
            min_coordinate = (
                    torch.min(rboxes[:, 0], rboxes[:, 1]) - torch.max(rboxes[:, 2], rboxes[:, 3]) / 2
            ).min()
            offsets = filter_inds[:, 1].to(rboxes) * (max_coordinate - min_coordinate + 1)
            boxes_for_nms = rboxes.clone()  # avoid modifying the original values in boxes
            boxes_for_nms[:, :2] += offsets[:, None]
            keep = torch.ops.detectron2.nms_rotated(boxes_for_nms, scores, nms_thresh)

            # keep = batched_nms_rotated(rboxes, scores, filter_inds[:, 1], nms_thresh)
        else:
            print('vp is not None and vp_bins is not None else')
            boxes[:, :, 2] = boxes[:, :, 2] + boxes[:, :, 0]  # x2
            boxes[:, :, 3] = boxes[:, :, 3] + boxes[:, :, 1]  # y2
            keep = box_ops.batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = dict()
    result['pred_boxes'] = boxes
    result['scores'] = scores
    result['pred_classes'] = filter_inds[:, 1]
    if vp is not None and vp_bins is not None:
        vp = vp.view(-1, num_bbox_reg_classes, vp_bins)  # R x C x bins
        vp = vp[filter_mask]
        vp = vp[keep]
        if vp_res is not None:
            vp_res = vp_res.view(-1, num_bbox_reg_classes, vp_bins)
            vp_res = vp_res[filter_mask]
            vp_res = vp_res[keep]
        if len(vp) > 0:
            _, vp_max = torch.max(vp, 1)
            result['viewpoint'] = vp_max
            if vp_res is not None:
                vp_res_filtered = list()
                for i, k in enumerate(vp_max):
                    vp_res_filtered.append(vp_res[i, k])
                # This result is directly the yaw orientation predicted
                result['viewpoint_residual'] = torch.tensor(vp_res_filtered).to(vp_max.device)
        else:
            result['viewpoint'] = vp
            result['viewpoint_residual'] = vp_res
    if h is not None:
        h = h.view(-1, num_bbox_reg_classes, 2)  # R x C x bins
        h = h[filter_mask]
        h = h[keep]
        result['height'] = h

    return result, filter_inds[:, 0]


def anglecorrection(angle):  # We need to transform from KITTI angle to normal one to perform nms rotated
    result = -angle - 90 if angle <= 90 else 270 - angle
    return torch.tensor([result])
