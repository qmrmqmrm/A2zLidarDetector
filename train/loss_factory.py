# from  utils.util_function import
import torch
import train.loss_pool as loss
import train.loss_util as lu
import utils.util_function as uf


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_weights = loss_weights
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = torch.Tensor(valid_category)
        self.loss_objects = self.create_loss_objects(loss_weights)

    def create_loss_objects(self, loss_weights):
        loss_objects = dict()
        if "bbox2d" in loss_weights:
            loss_objects["bbox2d"] = loss.Box2dRegression()
        if "object" in loss_weights:
            loss_objects["object"] = loss.ObjectClassification()
        if "bbox3d" in loss_weights:
            loss_objects["bbox3d"] = loss.Box3dRegression()
        if "height" in loss_weights:
            loss_objects["height"] = loss.HeightRegression()

        if 'yaw_reg' in loss_weights:
            loss_objects['yaw_reg'] = loss.YawRegression()
        if 'category' in loss_weights:
            loss_objects['category'] = loss.CategoryClassification()
        if 'yaw_cls' in loss_weights:
            loss_objects['yaw_cls'] = loss.YawClassification()

        return loss_objects

    def __call__(self, features, predictions, split):
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_objects}
        auxi = self.prepare_box_auxiliary_data(features, predictions)

        for loss_name, loss_object in self.loss_objects.items():
            scalar_loss = loss_object(features, predictions, auxi)
            weight = self.loss_weights[loss_name] if loss_name in self.loss_weights else self.loss_weights[loss_name]
            total_loss += scalar_loss * weight
            loss_by_type[loss_name] += scalar_loss
        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr, pred):
        auxiliary = dict()
        gt_aligned, match_result = uf.align_gt_with_pred(pred['head_proposals'], grtr)
        auxiliary['gt_matched'] = self.filter_foreground_objs(gt_aligned, match_result)
        pred_cat = self.cat_pred_batch(pred)
        auxiliary['pred_matched'] = self.filter_foreground_objs(pred_cat, match_result)
        return auxiliary

    def filter_foreground_objs(self, data, match_result):
        pos_mask = torch.nonzero(match_result == 1).squeeze(1)
        for key in data:
            data[key] = data[key][pos_mask]
        return data

    def cat_pred_batch(self, pred):
        head_proposals = pred['head_proposals']
        batch_size = len(head_proposals)
        pred_cat = {}
        for key in head_proposals[0]:
            batch_data = [head_proposals[i][key] for i in range(batch_size)]
            pred_cat[key] = torch.cat(batch_data)
        pred_cat['bbox2d'] = pred_cat['proposal_boxes']
        del pred_cat['proposal_boxes']
        pred_cat['bbox_3d_logits'] = pred['bbox_3d_logits']
        pred_cat['class_logits'] = pred['head_class_logits']
        pred_cat['yaw_logits'] = pred['head_yaw_logits']
        pred_cat['yaw_residuals'] = pred['head_yaw_residuals']
        pred_cat['height'] = pred['head_height_logits']
        return pred_cat

