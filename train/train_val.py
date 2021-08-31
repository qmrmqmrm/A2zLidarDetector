from timeit import default_timer as timer
import  os
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

    def run_epoch(self, epoch, data_loader):
        logger = LogData(True,cfg.Paths.CHECK_POINT,epoch)
        train_loader_iter = iter(data_loader)
        steps = len(train_loader_iter)
        for step in range(steps):
            # if step > 10:
            #     break
            features = next(train_loader_iter)
            print(features['image_file'])
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
        # self.model.train()
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()

        total_loss.backward()
        self.optimizer.step()
        outputs = Inference(prediction)
        pred_instances, _ = outputs.inference(0.05, 0.5, 100)

        # pred_boxes = list()
        print(type(pred_instances))
        print(len(pred_instances))

        for pred_instance in pred_instances:
            pred_boxes = pred_instance['pred_boxes']
            print(pred_boxes.shape)
            # pred_boxes.append(pred_boxes)
            # print(pred_boxes)
        for i, (image_file,pred_instance) in enumerate(zip(features['image_file'], pred_instances)):
            # inx = fg_inds[i:]
            pred_boxes = pred_instance['pred_boxes']
            li = image_file.split('/')[-3:]
            a = '/'.join(li)
        # #
            pred_path = os.path.join(cfg.Paths.RESULT_ROOT, 'image', cfg.Train.CKPT_NAME, 'ppp', a)
        #
            pred_folder = '/'.join(pred_path.split('/')[:-1])
            if not os.path.exists(pred_folder):
                os.makedirs(pred_folder)
            pred_img = cv2.imread(image_file)
            for box in pred_boxes:
                cv2.rectangle(pred_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              (255, 255, 255), 2)
            cv2.imwrite(pred_path, pred_img)
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
