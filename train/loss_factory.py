import torch
import train.loss_pool as loss
import train.loss_util as lu
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
        pred_slices = uf.merge_and_slice_features(predictions)
        pred = uf.slice_class(pred_slices)
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_objects}
        auxi = self.prepare_box_auxiliary_data(features, pred)

        for loss_name, loss_object in self.loss_objects.items():
            scalar_loss = loss_object(features, pred, auxi)
            weight = self.loss_weights[loss_name] if loss_name in self.loss_weights else self.loss_weights[loss_name]
            total_loss += scalar_loss * weight
            loss_by_type[loss_name] += scalar_loss
        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr, pred):
        batch = grtr['bbox2d'].shape[0]
        anchors = list()
        for anchor in grtr['anchors']:
            anchor = anchor.view(batch, -1, anchor.shape[-1])
            anchor = mu.convert_box_format_yxhw_to_tlbr(anchor)
            anchors.append(anchor)
        anchors_cat = torch.cat(anchors, dim=1)
        auxiliary = dict()
        auxiliary["gt_aligned"] = self.matched_gt(grtr, pred['rpn_bbox2d'])
        auxiliary["gt_feature"] = self.matched_gt(grtr, anchors_cat[..., :-1])
        auxiliary["gt_feature"] = self.split_feature(anchors, auxiliary["gt_feature"])

        # uf.print_structure('auxiliary["gt_feature"]', auxiliary["gt_feature"])
        auxiliary["pred_select"] = self.select_category(auxiliary['gt_aligned'], pred)
        return auxiliary

    def matched_gt(self, grtr, pred_box):
        matched = {key: [] for key in
                   ['bbox3d', 'category', 'bbox2d', 'yaw', 'yaw_rads', 'anchor_id', 'object', 'negative']}
        for i in range(self.batch_size):
            iou_matrix = uf.pairwise_iou(grtr['bbox2d'][i], pred_box[i])
            match_ious, match_inds = iou_matrix.max(dim=0)  # (height*width*anchor)
            positive = (match_ious > 0.4).unsqueeze(-1)
            negative = (match_ious < 0.1).unsqueeze(-1)
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
        gt_cate = aligned['category'].to(torch.int64).unsqueeze(-1)
        select_pred = dict()
        for key in ['bbox3d', 'yaw', 'yaw_rads']:
            pred_key = pred[key]
            gather_gt = torch.gather(pred_key, dim=2, index=gt_cate.repeat(1, 1, 1, pred_key.shape[-1])).squeeze(-2)
            if key == 'yaw_rads':
                gt_yaw = aligned['yaw'].to(torch.int64)
                gather_gt = torch.gather(gather_gt, dim=-1, index=gt_yaw)
            select_pred[key] = gather_gt
        select_pred['category'] = pred['category'].squeeze(-1)
        return select_pred
