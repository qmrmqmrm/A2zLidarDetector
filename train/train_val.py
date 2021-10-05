from timeit import default_timer as timer
import os
import cv2
import torch

import utils.util_function as uf
from log.nplog.logger import Logger
from log.nplog.logfile import LogFile
# from train.inference import Inference
import config as cfg
import model.submodules.model_util as mu


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
        logger = Logger(logger,logger, cfg.Paths.CHECK_POINT, epoch)
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)

        for step in range(steps):
            # if step > 10:
            #     break
            features = next(train_loader_iter)
            features = self.to_device(features)
            start = timer()
            file_name = features['image_file']
            prediction, total_loss, loss_by_type, features = self.run_step(features)
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
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.train()


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)
        self.split = "val"

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction, False)
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.eval()


def get_train_val(model, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, loss_object, epoch_steps)
    return trainer, validator
