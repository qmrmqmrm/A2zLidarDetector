import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.a2d2_dataset import A2D2Dataset
from dataloader.kitti_loader import KittiDataset
import utils.util_function as uf
from config import Config as cfg


def loader_factory(dataset_name):
    if dataset_name == "a2d2":
        path = cfg.Datasets.A2D2.PATH
        dataset = A2D2Dataset(path)
        return dataset
    if dataset_name == "kitti":
        path = cfg.Datasets.Kitti.PATH
        dataset = KittiDataset(path)
        return dataset


def get_dataset(dataset_name, batch_size):
    dataset = loader_factory(dataset_name)
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
    return data_loader


def test_data_loader():
    train_loader = get_dataset("a2d2", 4)
    for i, features in enumerate(train_loader):
        print("---- frame", i)
        print('features', type(features))
        # for key, val in features.items():
        image = features["image"].detach().numpy().astype(np.uint8)[0]
        boxes = features["bbox2d"].detach().numpy()[0]
        box_image = uf.draw_box(image, boxes)
        cv2.imshow('img',box_image)
        cv2.waitKey()


if __name__ == '__main__':
    test_data_loader()
