import torch

import config as cfg
import settings
from model.model_factory import ModelFactory
from dataloader.loader_factory import get_dataset
import utils.util_function as uf


def test_model(dataset_name="a2d2"):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    data_loader = get_dataset(dataset_name, 'train', batch_size)
    model = ModelFactory(dataset_name)
    model = model.make_model()
    train_loader_iter = iter(data_loader)
    steps = len(train_loader_iter)

    for step in range(steps):
        print("----- index:", step)
        features = next(train_loader_iter)
        features = to_device(features)
        rpn_proposals= model(features)
        uf.print_structure('rpn_proposals',rpn_proposals)



def to_device(features):
    device = cfg.Hardware.DEVICE
    for key in features:
        if isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(device=device)
    return features


if __name__ == "__main__":
    test_model()
