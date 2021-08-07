import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


from dataloader.a2z_loader import A2D2Loader
from dataloader.sampler import TrainingSampler
from config import Config as cfg


def loader_factory(dataset_name):
    if dataset_name == "a2d2":
        print('a2d2 data')
        path = cfg.Datasets.A2D2.PATH
        loader = A2D2Loader(path)
        return loader


def get_dataset(dataset_name, batch_size):
    loader = loader_factory(dataset_name)
    sampler = TrainingSampler(len(loader))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    data_loader = DataLoader(dataset=loader,
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
    # bboxes_2d = bboxes_2d.numpy()
    bboxes_2d = bboxes_2d[np.where(bboxes_2d > 0)].reshape(-1, 4)
    shape = bboxes_2d.shape
    for i in range(shape[0]):
        bbox = bboxes_2d[i, :]
        x0 = int(bbox[0])
        x1 = int(bbox[2])
        y0 = int(bbox[1])
        y1 = int(bbox[3])

        drow_img = cv2.rectangle(drow_img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imshow("drow_img", drow_img)
    cv2.waitKey()


def test_():
    train_loader = get_dataset("a2d2", 2)
    train_loader_iter = iter(train_loader)

    for i in range(len(train_loader_iter)):
        batch = next(train_loader_iter)
        img = batch[0].get("image")
        drow_box(img, batch[0].get('bbox2D'))
        # cv2.imshow("img",img)
        # cv2.waitKey()


if __name__ == '__main__':
    test_()
