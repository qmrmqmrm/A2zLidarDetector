import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.a2d2_dataset import A2D2Dataset
from dataloader.kitti_loader import KittiDataset
import model.submodules.model_util as mu
import utils.util_function as uf
import config as cfg


def loader_factory(dataset_name, split):
    if dataset_name == "a2d2":
        path = cfg.Datasets.A2D2.PATH
        dataset = A2D2Dataset(path, split)
        return dataset
    if dataset_name == "kitti":
        path = cfg.Datasets.Kitti.PATH
        dataset = KittiDataset(path, split)
        return dataset


def get_dataset(dataset_name, split, batch_size, shuffle=True):
    dataset = loader_factory(dataset_name, split)
    data_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True, num_workers=1)
    return data_loader


def test_data_loader():
    train_loader = get_dataset("a2d2", 'train', 1)
    print(len(train_loader))
    for i, features in enumerate(train_loader):
        print("---- frame", i, features['image_file'])
        print('features', type(features))
        # for key, val in features.items():
        image = features["image"].detach().numpy().astype(np.uint8)[0]
        boxes = features["bbox2d"].detach().numpy()[0]
        boxes_3d = features["bbox3d"].detach().numpy()[0]
        print(boxes_3d.shape)
        boxes_3d = mu.convert_box_format_yxhw_to_tlbr(boxes_3d[:, :4])
        box_image = uf.draw_box(image, boxes)
        image_i = image.copy()
        box_image_3d = uf.draw_box(image_i, boxes_3d)
        cv2.imshow('img', box_image)
        cv2.imshow('img_2', box_image_3d)
        cv2.waitKey()


if __name__ == '__main__':
    test_data_loader()
