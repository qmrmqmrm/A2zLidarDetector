import torch
import torch.nn.functional as F
import numpy as np

from config import Config as cfg
from model.submodules.box_regression import Box2BoxTransform


class NonMaximumSuppresion:
    def __init__(self):
        self.box2box_transform = Box2BoxTransform(weights=cfg.Model.RPN.BBOX_REG_WEIGHTS)
        self.vp_bins = cfg.Model.Structure.VP_BINS
        self.device = cfg.Model.Structure.DEVICE
        self.score_thresh = cfg.Model.ROI_HEADS.NMS_SCORE_THRESH
        self.iou_thresh = cfg.Model.ROI_HEADS.NMS_IOU_THRESH

    def __call__(self, pred):
        batch_size = len(pred['head_proposals'])
        num_pred_per_img = pred['head_proposals'][0]['proposal_boxes'].size(0)
        bbox_3d_logits = pred['bbox_3d_logits'].view(batch_size, num_pred_per_img, -1)
        proposal_boxes = [proposal['proposal_boxes'] for proposal in pred['head_proposals']]
        pred_class_logits = pred['head_class_logits'].view(batch_size, num_pred_per_img, -1)
        pred_yaw_logits = pred['head_yaw_logits'].view(batch_size, num_pred_per_img, -1)
        pred_yaw_residuals = pred['head_yaw_residuals'].view(batch_size, num_pred_per_img, -1)
        pred_nms = list()
        for proposal_box, bbox_3d_logit, pred_class_logit, pred_yaw_logit, pred_yaw_residual in \
                zip(proposal_boxes, bbox_3d_logits, pred_class_logits, pred_yaw_logits, pred_yaw_residuals):
            keep, rboxes = self.nms_rotated(proposal_box, bbox_3d_logit, pred_class_logit, pred_yaw_logit,
                                            pred_yaw_residual)
            """
            pred_nms = [{"bbox2d": [num_box, 4(tlbr)], "height": [num_box, 2(z,h), "yaw": [num_box, 1(angle)], "category": [num_box, 1]}]
            """
            pred_filter = self.filter_pred(pred, keep, rboxes)
            pred_nms.append(pred_filter)
        return pred_nms

    def nms_rotated(self, proposal_boxes, bbox_3d_logits, pred_class_logits, pred_yaw_logits, pred_yaw_residuals):
        num_pred = len(proposal_boxes)
        B = proposal_boxes.shape[-1]  # 4
        K = bbox_3d_logits.shape[-1] // B  # 3
        boxes = self.box2box_transform.apply_deltas(
            bbox_3d_logits.view(num_pred * K, B),
            proposal_boxes.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
            True
        )
        height, weight = [700, 1400]
        x1 = boxes[:, 0].clamp(min=0, max=weight)
        y1 = boxes[:, 1].clamp(min=0, max=height)
        x2 = boxes[:, 2].clamp(min=0, max=weight)
        y2 = boxes[:, 3].clamp(min=0, max=height)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        boxes = boxes.view(-1, K, 4)
        scores = F.softmax(pred_class_logits, dim=-1)
        scores = scores[:, :-1]

        yaw_bin_prob = F.softmax(pred_yaw_logits, dim=-1)
        yaw_residual = self.apply_vp_deltas(pred_yaw_residuals, pred_class_logits)
        filter_mask = scores > self.score_thresh  # R x K
        filter_inds = filter_mask.nonzero()
        boxes = boxes[filter_mask]

        scores = scores[filter_mask]
        _vp = yaw_bin_prob.view(-1, K, self.vp_bins)  # R x C x bins
        _vp = _vp[filter_mask]
        if len(_vp) == 0:
            return None

        _, vp_max = torch.max(_vp, 1)
        vp_filtered = vp_max
        _vp_res = yaw_residual.view(-1, K, self.vp_bins)
        _vp_res = _vp_res[filter_mask]
        vp_res_filtered = list()
        for i, k in enumerate(vp_max):
            vp_res_filtered.append(_vp_res[i, k])

        rboxes = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            angle = self.anglecorrection(vp_res_filtered[i] * 180 / np.pi).to(box.device)
            box = torch.cat((box, angle))
            rboxes.append(box)
        rboxes = torch.cat(rboxes).reshape(-1, 5).to(vp_filtered.device)
        max_coordinate = (
                torch.max(rboxes[:, 0], rboxes[:, 1]) + torch.max(rboxes[:, 2], rboxes[:, 3]) / 2
        ).max()
        min_coordinate = (
                torch.min(rboxes[:, 0], rboxes[:, 1]) - torch.max(rboxes[:, 2], rboxes[:, 3]) / 2
        ).min()
        offsets = filter_inds[:, 1].to(rboxes) * (max_coordinate - min_coordinate + 1)
        boxes_for_nms = rboxes.clone()  # avoid modifying the original values in boxes
        boxes_for_nms[:, :2] += offsets[:, None]
        keep = torch.ops.detectron2.nms_rotated(boxes_for_nms, scores, self.iou_thresh)
        return keep, rboxes

    def apply_vp_deltas(self, pred_yaw_residuals, pred_class_logits):
        deltas = pred_yaw_residuals
        assert torch.isfinite(deltas).all().item()

        bin_dist = np.linspace(-np.pi, np.pi, self.vp_bins + 1)
        bin_res = (bin_dist[1] - bin_dist[0]) / 2.
        bin_dist = bin_dist - bin_res
        bin_dist = np.tile(bin_dist[:-1], pred_class_logits.shape[1] - 1)
        src_vp_res = torch.tensor(bin_dist, dtype=torch.float32).to(self.device)

        wvp = np.trunc(1 / bin_res)
        dvp = deltas / wvp

        pred_vp_res = dvp + src_vp_res + bin_res
        pred_vp_res[pred_vp_res < -np.pi] += 2 * np.pi
        return pred_vp_res

    def anglecorrection(self, angle):  # We need to transform from KITTI angle to normal one to perform nms rotated
        result = -angle - 90 if angle <= 90 else 270 - angle
        return torch.tensor([result])

    def filter_pred(self, pred, keep, rboxes):
        """

        :param pred:
        :param keep:
        :param rboxes:
        :return:
        pred_nms = [{"bbox2d": [num_box, 4(tlbr)], "height": [num_box, 2(z,h), "yaw": [num_box, 1(angle)], "category": [num_box, 1]}]
        """
        pred_height_logits = pred['head_height_logits'].view(-1, 2)
        bbox2d = rboxes[:, :-1]
        angle = rboxes[:, -1]
        yaw = angle * (np.pi / 180)
        height = pred_height_logits[keep]
        category = keep % 3
        return {"bbox2d": bbox2d, "height": height, "yaw": yaw, "category": category}
