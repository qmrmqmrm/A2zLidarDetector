from timeit import default_timer as timer
import os.path as op
import torch

import utils.util_function as uf
from log.nplog.logger import Logger
import config as cfg

class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.split = ""
        self.device = cfg.Hardware.DEVICE

    def run_epoch(self, logger, epoch, data_loader):

        self.mode_set()
        logger = Logger(logger,logger, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME), epoch)
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)

        for step in range(steps):
            # if step > 10:
            #     break
            features = next(train_loader_iter)
            features = self.to_device(features)
            start = timer()
            file_name = features['image_file']
            prediction, total_loss, loss_by_type = self.run_step(features)
            logger.log_batch_result(step, features, prediction, total_loss, loss_by_type, epoch)

            features['image_file'] = file_name
            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"({self.split}) {step}/{steps} steps in {epoch} epoch, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")

        logger.finalize()
        return logger

    def to_device(self, features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(device=self.device)
            if isinstance(features[key], list):
                data = list()
                for feature in features[key]:
                    if isinstance(feature, torch.Tensor):
                        feature = feature.to(device=self.device)
                    data.append(feature)
                features[key] = data
        return features

    def run_step(self, features):
        """

        :param features:
            {'image': [batch, height, width, channel],
             'anchors': [batch, height/stride, width/stride, anchor, yxwh + id] * features
            'category': [batch, fixbox, 1],
            'bbox2d': [batch, fixbox, 4](tlbr), 'bbox3d': [batch, fixbox, 6], 'object': [batch, fixbox, 1],
            'yaw': [batch, fixbox, 1], 'yaw_rads': [batch, fixbox, 1]}, 'anchor_id': [batch, fixbox, 1]
            'image_file': image file name per batch
            }
        :return:
           prediction: {
                        'bbox2d' : torch.Size([batch, 512, 4(tlbr)])
                        'objectness' : torch.Size([batch, 512, 1])
                        'anchor_id' torch.Size([batch, 512, 1])
                        'rpn_feat_bbox2d' : list(torch.Size([batch, height/stride* width/stride* anchor, 4(tlbr)])
                        'rpn_feat_objectness' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
                        'rpn_feat_anchor_id' : list(torch.Size([batch, height/stride* width/stride* anchor, 1])
                        'head_output' : torch.Size([4, 512, 93])
                        }
        """
        raise NotImplementedError()

    def mode_set(self):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
        self.split = "train"

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction, True)
        for key, val in loss_by_type.items():
            print(key, val.to('cpu').detach().numpy())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print('lr',self.optimizer.param_groups[0]['lr'])
        return prediction, total_loss, loss_by_type

    def mode_set(self):
        self.model.train()


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)
        self.split = "val"

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction, False)
        return prediction, total_loss, loss_by_type

    def mode_set(self):
        self.model.eval()


def get_train_val(model, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, loss_object, epoch_steps)
    return trainer, validator
