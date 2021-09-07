import torch
import numpy as np
import cv2
from torchvision.ops import boxes as box_ops
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple, batched_nms_rotated
from torchvision.ops import nms  # BC-compat


def test_nms():
    img = np.zeros([700, 1400, 3])
    org_img = img.copy()
    boxes = np.array([[800.000, 100.000, 850.000, 150.000],
                      [810.000, 110.000, 840.000, 150.000],
                      [805.000, 110.000, 860.000, 160.000],
                      [790.000, 90.000, 840.000, 140.000],
                      [770.000, 50.000, 790.000, 140.000],

                      [790.000, 120.000, 835.000, 180.000],
                      [760.000, 110.000, 798.000, 140.000],
                      [810.000, 95.000, 840.000, 140.000],
                      [820.000, 100.000, 870.000, 145.000],
                      [790.000, 90.000, 870.000, 150.000],
                      ])
    for i, box in enumerate(boxes):
        org_img = cv2.rectangle(org_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                (50 * (i + 2), 50 * (i + 3), 50 * (i + 1)), (i + 1))
    cv2.imshow('org_img', org_img)
    cv2.waitKey()
    scores = np.array([0.7452, 2.2258, 2.7207, 0.4410, 0.7378,
                       0.50, 1.7378, 2.7378, 0.40, 0.7573])
    idx = np.array([1, 1, 1, 0, 2,
                    2, 2, 0, 0, 0])
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    idx = torch.tensor(idx)
    print(boxes.shape)
    print(scores.shape)
    img = np.zeros([700, 1400, 3])
    iou = box_ops.batched_nms(boxes, scores, idx, 0.2)
    boxes = boxes[iou]
    for i, box in enumerate(boxes):
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (0, 20 * (i + 3), 20 * (i + 1)), (i + 1))
    cv2.imshow('img', img)
    cv2.waitKey()
    print(iou)


def nms_rotated_test():
    boxes1 = [[[0, 0, 20, 20, 0], [14, 14, 20, 20, 0], [50, 50, 20, 20, 0]],
              [[0, 0, 20, 20, 0], [14, 14, 20, 20, 0], [50, 50, 20, 20, 0]]]
    boxes2 = [[0, 0, 20, 20, 45], [14, 14, 20, 20, 45], [50, 50, 20, 20, 0]]
    boxes1 = torch.tensor(boxes1).type(torch.float32)
    boxes2 = torch.tensor(boxes2).type(torch.float32)
    score1 = torch.tensor([[0.9, 0.7, 0.99],
                           [0.9, 0.7, 0.5]])
    score2 = torch.tensor([0.7, 0.9, 0.99])
    inds_1 = torch.ops.detectron2.nms_rotated(boxes1, score1, 0.01)
    inds_2 = torch.ops.detectron2.nms_rotated(boxes2, score2, 0.01)
    print(inds_1, inds_2)


if __name__ == '__main__':
    nms_rotated_test()
