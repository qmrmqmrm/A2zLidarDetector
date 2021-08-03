
import numpy as np
from timeit import default_timer as timer

import utils.util_function as uf


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps

    def run_epoch(self, dataset):
        # logger = LogData()
        for step, features in enumerate(dataset):
            start = timer()
            prediction, total_loss, loss_by_type = self.run_step(features)
            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")
            # if step > 20:
            #     break

        # logger.finalize()
        # return logger

    def run_step(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    def run_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["image"])
            total_loss, loss_by_type = self.loss_object(features, prediction)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    def run_step(self, features):
        prediction = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type

