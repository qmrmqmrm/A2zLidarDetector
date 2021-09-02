from timeit import default_timer as timer
import os
import cv2
import torch

import utils.util_function as uf
from log.logger import LogData
from train.inference import Inference
from config import Config as cfg
import model.submodules.model_util as mu


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.split = ""
        self.device = cfg.Model.Structure.DEVICE

    def run_epoch(self, visual_log, epoch, data_loader):
        logger = LogData(visual_log, cfg.Paths.CHECK_POINT, epoch)
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)
        self.mode_set()
        for step in range(steps):
            if step > 10:
                break
            features = next(train_loader_iter)
            features = self.to_device(features)

            start = timer()
            prediction, total_loss, loss_by_type = self.run_step(features)
            # uf.print_structure("feat", features)
            # uf.print_structure("pred", prediction)
            logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"({self.split}) {step}/{steps} steps in {epoch} epoch, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")
            print("")

        logger.finalize()
        return logger

    def to_device(self, features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(self.device)
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
        features = mu.remove_padding(features)

        total_loss, loss_by_type = self.loss_object(features, prediction, True)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return prediction, total_loss, loss_by_type

    def mode_set(self):
        self.model.train()


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)
        self.split = "val"

    def run_step(self, features):
        prediction = self.model(features)

        # uf.print_structure('prediction', prediction)
        features = mu.remove_padding(features)
        output = Inference(prediction)
        inference = output.inference(0.05, 0.5, 100)
        uf.print_structure('features', inference)

        total_loss, loss_by_type = self.loss_object(features, prediction, False)
        return prediction, total_loss, loss_by_type

    def mode_set(self):
        self.model.eval()


def get_train_val(model, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, loss_object, epoch_steps)
    return trainer, validator
