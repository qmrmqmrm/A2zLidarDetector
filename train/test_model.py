import torch

import config as cfg
import settings
from model.model_factory import ModelFactory
from dataloader.loader_factory import get_dataset
import utils.util_function as uf


def test_model(dataset_name="a2d2"):
    batch_size = cfg.Train.BATCH_SIZE
    data_loader = get_dataset(dataset_name, 'train', batch_size)
    model = ModelFactory(dataset_name)
    model = model.make_model()
    print(model)
    train_loader_iter = iter(data_loader)
    steps = len(train_loader_iter)

    for step in range(steps):
        print("----- index:", step)
        features = next(train_loader_iter)
        features = to_device(features)
        model_output = model(features)
        uf.print_structure('model_output', model_output)


def to_device(features):
    device = cfg.Hardware.DEVICE
    for key in features:
        if isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(device=device)
        if isinstance(features[key], list):
            data = list()
            for feature in features[key]:
                if isinstance(feature, torch.Tensor):
                    feature = feature.to(device=device)
                data.append(feature)
            features[key] = data

    return features


if __name__ == "__main__":
    test_model()
