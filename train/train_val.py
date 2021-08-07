import numpy as np
from timeit import default_timer as timer

import utils.util_function as uf


class TrainValBase:
    def __init__(self, model, data_loader, loss_object, optimizer, epoch_steps):
        self.model = model
        self.data_loader = data_loader
        self.train_loader_iter = iter(self.data_loader)
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps

    def run_epoch(self):
        # logger = LogData()
        # for step, features in enumerate(dataset):
        steps = len(self.train_loader_iter)
        for step in range(steps):

            features = next(self.train_loader_iter)
            # print("step[0].get('image').shape")
            # print(step[0].get('image').shape)
            # print(step[0].get('image'))
            start = timer()
            prediction, total_loss, loss_by_type = self.run_step(features)
            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"{self.epoch_steps} epoch, training {step}/{steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")
            # print(prediction, total_loss, loss_by_type)
            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            # uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
            #                   f"time={timer() - start:.3f}, "
            #                   f"loss={total_loss:.3f}, ")
                # if step > 20:
                #     break

        # logger.finalize()
        # return logger

    def run_step(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, data_loader, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, data_loader, loss_object, optimizer, epoch_steps)

    def run_step(self, features):
        prediction = self.model(features)
        # print('pre : ', prediction)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()
        # print("total_loss : ",total_loss)
        total_loss.backward()
        self.optimizer.step()

        # grads = gradient(total_loss, self.model.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type


class ModelValidater(TrainValBase):
    def __init__(self, model, data_loader, loss_object, epoch_steps=0):
        super().__init__(model, data_loader, loss_object, None, epoch_steps)

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type


def get_train_val(model, data_loader, loss_object, optimizer, epoch_steps=0):
    trainer = ModelTrainer(model, data_loader, loss_object, optimizer, epoch_steps)
    validator = ModelValidater(model, data_loader, loss_object, epoch_steps)
    return trainer, validator
