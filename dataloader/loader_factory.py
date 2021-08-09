import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.a2d2_dataset import A2D2Dataset
from dataloader.kitti_loader import KittiDataset
from dataloader.sampler import TrainingSampler
from config import Config as cfg


def loader_factory(dataset_name):
    if dataset_name == "a2d2":
        print('a2d2')
        path = cfg.Datasets.A2D2.PATH
        dataset = A2D2Dataset(path)
        return dataset
    if dataset_name == "kitti":
        path = cfg.Datasets.Kitti.PATH
        dataset = KittiDataset(path)
        return dataset


def get_dataset(dataset_name, batch_size):
    dataset = loader_factory(dataset_name)
    sampler = TrainingSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=trivial_batch_collator,
                             num_workers=2)

    return data_loader


def trivial_batch_collator(batch):
    return batch


def drow_box(img, bboxes_2d):
    # print(bboxes_2d)
    img = img.permute(1, 2, 0)
    drow_img = img.numpy()

    for bbox in bboxes_2d:
        x0 = int(bbox[0])
        x1 = int(bbox[2])
        y0 = int(bbox[1])
        y1 = int(bbox[3])

        drow_img = cv2.rectangle(drow_img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imshow("drow_img", drow_img)
    cv2.waitKey()


def test_():
    train_loader = get_dataset("a2d2", 1)
    train_loader_iter = iter(train_loader)

    for i in range(len(train_loader_iter)):
        batches = next(train_loader_iter)
        print("batches type: ", type(batches))
        print("batches type: ", (batches))
        # for batch in batches:
        #     img = batch.get("image")
        #     drow_box(img, batch.get('gt_bbox2D'))


if __name__ == '__main__':
    test_()
