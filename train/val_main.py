import os
import os.path as op
from typing import Any, Dict, List
import torch
import pandas as pd

import settings
import config as cfg
from model.model_factory import ModelFactory
from dataloader.loader_factory import get_dataset
from train.train_val import get_train_val, ModelValidater
from train.loss_factory import IntegratedLoss
from log.nplog.logfile import LogFile
import utils.util_function as uf


# TODO: rearrange-code-21-11
def validation_main(dataset_name="a2d2", loss_weights=LossComb.BIRDNET):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    ckpt_path = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    valid_category = cfg.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)

    test_data_loader = get_dataset(dataset_name, 'test', batch_size, False)
    model_factory = ModelFactory(dataset_name)
    model = model_factory.make_model()
    model = try_load_weights(ckpt_path, model)
    loss_object = IntegratedLoss(batch_size, loss_weights, valid_category)

    validator = ModelValidater(model, loss_object, start_epoch)
    val_result = validator.run_epoch(True, start_epoch+1, test_data_loader)

    log_file = LogFile(ckpt_path)
    log_file.save_val_log(val_result)


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


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model.load_state_dict(torch.load(ckpt_file))
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


if __name__ == '__main__':
    validation_main()
