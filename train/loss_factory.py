# from  utils.util_function import
import torch
import train.loss_pool as loss

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

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_objects}

        for loss_name, loss_object in self.loss_objects.items():
            scalar_loss = loss_object(features, predictions)
            weight = self.loss_weights[loss_name] if loss_name in self.loss_weights \
                                                     else self.loss_weights[loss_name]
            total_loss += scalar_loss * weight
            loss_by_type[loss_name] += scalar_loss
        return total_loss, loss_by_type


