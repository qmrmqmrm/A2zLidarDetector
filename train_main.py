import os
import os.path as op
from typing import Any, Dict, List
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.a2z_loader import A2D2_Loader
from dataloader.sampler import TrainingSampler
import utils.util_function as uf
from config import Config as cfg
from model.model_factory import GeneralizedRCNN

import train.train_val as tv
from train.train_loop import SimpleTrainer, DefaultTrainer


def train_main():
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, model_save in cfg.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save)


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    path, ckpt_path = cfg.Datasets.A2D2.PATH, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    # valid_category = cfg.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)
    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    data_loader = get_dataset(path, batch_size)
    # dataset_val, val_steps, _, _ = get_dataset(tfrd_path, dataset_name, False, batch_size, "val")
    # print(dataset_train)
    model = GeneralizedRCNN(3)

    optimizer = build_optimizer(model, learning_rate)
    trainer = DefaultTrainer(model, data_loader, optimizer)
    # trainer = tv.trainer_factory(train_mode, model, loss_object, optimizer, train_steps)

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        train_result = trainer.train()
        print(train_result)

    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0

        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print(f"[read_previous_epoch] NO history in {filename}")
        return 0


def get_dataset(path, batch_size, shuffle=True):
    loader = A2D2_Loader(path)
    sampler = TrainingSampler(len(loader))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    data_loader = DataLoader(dataset=loader,
                             batch_sampler=batch_sampler,
                             collate_fn=trivial_batch_collator,
                             num_workers=2)

    print("\n")
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model.load_weights(ckpt_file)
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def build_optimizer(model: torch.nn.Module, learning_rate) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = learning_rate
        weight_decay = cfg.Model.Output.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.Model.Output.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            lr = learning_rate
            weight_decay = cfg.Model.Output.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.Model.Output.MOMENTUM)
    return optimizer


def drow_box(img, bboxes_2d):
    # print(bboxes_2d)
    img = img.permute(1, 2, 0)
    drow_img = img.numpy()
    # bboxes_2d = bboxes_2d.numpy()
    bboxes_2d = bboxes_2d[np.where(bboxes_2d > 0)].reshape(-1, 4)
    shape = bboxes_2d.shape
    for i in range(shape[0]):
        bbox = bboxes_2d[i, :]
        x0 = int(bbox[0] - bbox[2] / 2)
        x1 = int(bbox[0] + bbox[2] / 2)
        y0 = int(bbox[1] - bbox[3] / 2)
        y1 = int(bbox[1] + bbox[3] / 2)

        drow_img = cv2.rectangle(drow_img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imshow("drow_img", drow_img)
    cv2.waitKey()


def test_():
    path = "/media/dolphin/intHDD/birdnet_data/my_a2d2"
    train_loader = get_dataset(path, 2)
    train_loader_iter = iter(train_loader)
    for i in range(3):
        batch = next(train_loader_iter)
        img = batch[0].get("image")
        drow_box(img, batch[0].get('bbox2D'))
        # cv2.imshow("img",img)
        # cv2.waitKey()



if __name__ == '__main__':
    train_main()
