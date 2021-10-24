import torch
import numpy as np
import cv2
import train.loss_pool as loss

import utils.util_function as uf
import model.submodules.model_util as mu
import config as cfg


class IntegratedLoss:
    def __init__(self, batch_size, loss_weights, valid_category):
        self.loss_weights = loss_weights
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = torch.Tensor(valid_category)
        self.loss_objects = self.create_loss_objects(loss_weights)
        self.batch_size = batch_size
        self.device = cfg.Hardware.DEVICE
        self.align_iou_threshold = cfg.Loss.ALIGN_IOU_THRESHOLD
        self.anchor_iou_threshold = cfg.Loss.ANCHOR_IOU_THRESHOLD

    def create_loss_objects(self, loss_weights):
        loss_objects = dict()
        if "bbox2d" in loss_weights:
            loss_objects["bbox2d"] = loss.Box2dRegression()

        if "object" in loss_weights:
            loss_objects["object"] = loss.ObjectClassification()

        if 'category' in loss_weights:
            loss_objects['category'] = loss.CategoryClassification()

        if "bbox3d" in loss_weights:
            loss_objects["bbox3d"] = loss.Box3dRegression()

        if 'yaw_cls' in loss_weights:
            loss_objects['yaw_cls'] = loss.YawClassification()

        if 'yaw_reg' in loss_weights:
            loss_objects['yaw_reg'] = loss.YawRegression()

        return loss_objects

    def __call__(self, features, predictions, split):
        """

        :param features:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4(tlbr)], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
        :param predictions:
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'head_output' : torch.Size([4, 512, 93])
            }
        :param split:
        :return:
        """
        # pred_slices = uf.merge_and_slice_features(predictions) # b, n, 18
        # pred = uf.slice_class(pred_slices) # b, n, 3, 6
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_objects}
        auxi = self.prepare_box_auxiliary_data(features, predictions)

        for loss_name, loss_object in self.loss_objects.items():
            scalar_loss = loss_object(features, predictions, auxi)
            weight = self.loss_weights[loss_name] if loss_name in self.loss_weights else self.loss_weights[loss_name]
            total_loss += scalar_loss * weight
            loss_by_type[loss_name] += scalar_loss * weight
        return total_loss, loss_by_type, auxi

    def prepare_box_auxiliary_data(self, grtr, pred):
        """

        :param grtr:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4(tlbr)], 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
        :param pred:
            {
            'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
            'objectness' : torch.Size([batch, 512, 1])
            'anchor_id' torch.Size([batch, 512, 1])
            'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
            'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
            'category' : torch.Size([batch, 512, class_num, 1])
            'bbox3d' : torch.Size([batch, 512, class_num, 6])
            'yaw' : torch.Size([batch, 512, class_num, 12])
            'yaw_rads' : torch.Size([batch, 512, class_num, 12])
            }
        :return: auxiliary:
        {
        'gt_aligned' : {
                        'bbox3d' : torch.Size([batch, 512, 6])
                        'category' : torch.Size([batch, 512, 1])
                        'bbox2d' : torch.Size([batch, 512, 4[tlbr])
                        'yaw' : torch.Size([batch, 512, 1])
                        'yaw_rads' : torch.Size([batch, 512, 1])
                        'anchor_id' : torch.Size([batch, 512, 1])
                        'object' : torch.Size([batch, 512, 1])
                        'negative' : torch.Size([batch, 512, 1])
                        }
        'gt_feature' : {
                        'bbox3d' : list(torch.Size([batch, height/stride* width/stride* anchor, 6]))
                        'category' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4]))
                        'yaw' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'yaw_rads' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'object' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        'negative' : list(torch.Size([batch, height/stride* width/stride* anchor, 1]))
                        }
        'pred_select' : {
                        'bbox3d' : torch.Size([batch, 512, 6])
                        'category' : torch.Size([batch, 512, 1])
                        'yaw' : torch.Size([batch, 512, 1])
                        'yaw_rads' : torch.Size([batch,512, 1])
                        }
        }
        """
        batch = grtr['bbox2d'].shape[0]
        anchors = list()
        for anchor in grtr['anc_feat']:  # batch, h, w, a, 4
            anchor = anchor.view(batch, -1, anchor.shape[-1])  # b hwa 4
            anchor = mu.convert_box_format_yxhw_to_tlbr(anchor)  # tlbr
            anchors.append(anchor)
        anchors_cat = torch.cat(anchors, dim=1)
        auxiliary = dict()
        auxiliary["gt_aligned"] = self.matched_gt(grtr, pred['bbox2d'], self.align_iou_threshold)  # tlbr tlbr
        auxiliary["gt_feature"] = self.matched_gt(grtr, anchors_cat[..., :-1], self.anchor_iou_threshold)  # tlbr tlbr
        auxiliary["gt_feature"] = self.split_feature(anchors, auxiliary["gt_feature"])
        auxiliary["pred_select"] = self.select_category(auxiliary['gt_aligned'], pred)

        return auxiliary

    def matched_gt(self, grtr, pred_box, iou_threshold):
        matched = {key: [] for key in
                   ['bbox3d', 'category', 'bbox2d', 'yaw', 'yaw_rads', 'anchor_id', 'object', 'negative']}
        for i in range(self.batch_size):
            iou_matrix = uf.pairwise_iou(grtr['bbox2d'][i], pred_box[i])

            match_ious, match_inds = iou_matrix.max(dim=0)  # (height*width*anchor)
            positive = (match_ious >= iou_threshold[1]).unsqueeze(-1)
            negative = (match_ious < iou_threshold[0]).unsqueeze(-1)
            for key in matched:
                if key == "negative":
                    matched["negative"].append(negative)
                else:
                    matched[key].append(grtr[key][i, match_inds] * positive)

        for key in matched:
            matched[key] = torch.stack(matched[key], dim=0)

        return matched

    def split_feature(self, anchors, feature):
        slice_features = {key: [] for key in feature.keys()}
        for key in feature.keys():
            last_channel = 0
            for anchor in anchors:
                scales = anchor.shape[1] + last_channel
                slice_feature = feature[key][:, last_channel:scales]
                last_channel = scales
                slice_features[key].append(slice_feature)
        return slice_features

    def select_category(self, aligned, pred):
        gt_cate = (aligned['category'].to(torch.int64)).unsqueeze(-1)
        select_pred = dict()
        for key in ['bbox3d', 'yaw', 'yaw_rads']:
            pred_key = pred[key]
            batch, num, cate, channel = pred_key.shape
            pred_padding = torch.zeros((batch, num, 1, channel), device=self.device)
            pred_key = torch.cat([pred_padding, pred_key], dim=-2)
            gather_gt = torch.gather(pred_key, dim=2, index=gt_cate.repeat(1, 1, 1, pred_key.shape[-1])).squeeze(-2)
            if key == 'yaw_rads':
                gt_yaw = aligned['yaw'].to(torch.int64)
                gather_gt = torch.gather(gather_gt, dim=-1, index=gt_yaw)
            select_pred[key] = gather_gt
        select_pred['category'] = pred['category'].squeeze(-1)
        return select_pred


def test_select_category():
    alined = dict()
    alined['category'] = torch.tensor([[[1], [3], [0]]], device='cuda')  # 1, 3, 1
    pred_box = dict()
    pred_box['bbox3d'] = torch.rand((1,3,3,2),device='cuda')
    loss = IntegratedLoss(1, {"haha": 1}, 1)
    sel_box = loss.select_category(alined, pred_box)
    print('pred_box', pred_box['bbox3d'])
    print('sel_box', sel_box)
    numind = 0
    cate1 = alined['category'][0, numind, 0] - 1
    pred1 = pred_box['bbox3d'][0, numind, cate1]
    sele1 = sel_box['bbox3d'][0, numind]
    print("compare:", cate1, pred1, sele1)



if __name__ == '__main__':
    test_select_category()