import numpy as np
from timeit import default_timer as timer

import utils.util_function as uf
from train.logger import LogData


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model


        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps

    def run_epoch(self,epoch, data_loader):
        logger = LogData()
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)
        for step in range(steps):
            # if step > 10:
            #     break
            features, image_file = next(train_loader_iter)
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
        # self.model.train()
        prediction = self.model(features)

        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()

        total_loss.backward()
        self.optimizer.step()

        # grads = gradient(total_loss, self.model.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type, 'train'


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    def run_step(self, features):
        # self.model.eval()
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type, 'valid'


def get_train_val(model, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, loss_object, epoch_steps)
    return trainer, validator
