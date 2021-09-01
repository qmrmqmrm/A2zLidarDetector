from timeit import default_timer as timer
import os
import cv2

import utils.util_function as uf
from log.logger import LogData
from train.inference import Inference
from config import Config as cfg


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps

    def run_epoch(self, visual_log, epoch, data_loader):
        logger = LogData(visual_log, cfg.Paths.CHECK_POINT, epoch)
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)
        for step in range(steps):
            # if step > 10:
            #     break
            features = next(train_loader_iter)
            print()
            start = timer()
            prediction, total_loss, loss_by_type, split = self.run_step(features)
            logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"({split}) {step}/{steps} steps in {epoch} epoch, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")

        logger.finalize()
        return logger

    def run_step(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)

    def run_step(self, features):
        self.model.train()
        prediction = self.model(features)

        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()

        total_loss.backward()
        self.optimizer.step()

        return prediction, total_loss, loss_by_type, 'train'


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    def run_step(self, features):
        self.model.eval()
        prediction = self.model(features)

        print(type(prediction))
        for key, features in prediction.items():
            if isinstance(features, list):
                for k, feat in enumerate(features):
                    if isinstance(feat, dict):
                        for f_k, f in feat.items():
                            print("prediction", key, k, f_k, f.shape)
                    else:
                        print("prediction", key, k, feat.shape)
            else:

                if isinstance(features, dict):
                    for f_k, feat in features.items():
                        if isinstance(feat, list):
                            for j, f in enumerate(feat):
                                print("prediction", key, j, f_k, f.shape)
                else:
                    print("prediction", key, features.shape)

        outputs = Inference(prediction)
        pred_instances, _ = outputs.inference(0.05, 0.5, 100)

        for key, features in pred_instances.items():
            if isinstance(features, list):
                for k, feat in enumerate(features):
                    if isinstance(feat, dict):
                        for f_k, f in feat.items():
                            print("pred_instances", key, k, f_k, f.shape)
                    else:
                        print("pred_instances", key, k, feat.shape)
            else:

                if isinstance(features, dict):
                    for f_k, feat in features.items():
                        if isinstance(feat, list):
                            for j, f in enumerate(feat):
                                print("pred_instances", key, j, f_k, f.shape)
                else:
                    print("pred_instances", key, features.shape)

        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type, 'valid'


def get_train_val(model, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, loss_object, epoch_steps)
    return trainer, validator
