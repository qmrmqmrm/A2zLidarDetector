import numpy as np
import torch
import cv2
from timeit import Timer

import utils.util_function as uf


def comput_rotated_iou(b_imgs, box_b):
    # denorm_bbox = torch.cat([box_a[:, :4] * 768, box_a[:, 4].unsqueeze(1) * 90], dim=1)
    # print('denorm_bbox', denorm_bbox.shape)
    # b_boxes = list(map(lambda x: np.ceil(cv2.boxPoints(((x[1], x[0]), (x[2], x[3]), x[4]))), denorm_bbox.numpy()))
    # print('b_boxes', len(b_boxes))
    # b_imgs = torch.from_numpy(
    #     np.array([cv2.fillConvexPoly(np.zeros((768, 768), dtype=np.uint8), np.int0(b), 1) for b in b_boxes]).astype(
    #         float)).float()
    # print('b_imgs', b_imgs.shape)
    intersection = torch.FloatTensor()
    summation = torch.FloatTensor()
    for b_img in b_imgs:
        intersection = torch.cat([intersection, (b_img * box_b).sum((1, 2)).unsqueeze(0)])
        print('intersection', intersection.shape)
        summation = torch.cat([summation, (b_img + box_b).sum((1, 2)).unsqueeze(0)])
        print('summation',summation.shape)
    return intersection / (summation - intersection + 1.0)


def main():
    anc_grids = [3, 6, 12]
    anc_zooms = [0.7]
    anc_ratios = [(1., 1)]
    anc_angles = np.array(range(-90, 90, 45)) / 90
    anchor_scales = [(anz * i, anz * j) for anz in anc_zooms for (i, j) in anc_ratios]
    k = len(anchor_scales) * len(anc_angles)  # number of anchor boxes per anchor point
    anc_offsets = [1 / (o * 2) for o in anc_grids]

    anc_x = np.concatenate([np.repeat(np.linspace(ao, 1 - ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])
    anc_y = np.concatenate([np.tile(np.linspace(ao, 1 - ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])
    anc_ctrs = np.repeat(np.stack([anc_x, anc_y], axis=1), k, axis=0)
    anc_sizes = np.tile(np.concatenate([np.array([[o / ag, p / ag] for i in range(ag * ag) for o, p in anchor_scales])
                                        for ag in anc_grids]), (len(anc_angles), 1))
    grid_sizes = torch.from_numpy(np.concatenate([np.array([1 / ag for i in range(ag * ag) for o, p in anchor_scales])
                                                  for ag in anc_grids for aa in anc_angles])).unsqueeze(1)
    anc_rots = np.tile(np.repeat(anc_angles, len(anchor_scales)), sum(i * i for i in anc_grids))[:, np.newaxis]
    anchors = torch.from_numpy(np.concatenate([anc_ctrs, anc_sizes, anc_rots], axis=1)).float()
    print('anchors', anchors)
    denorm_anchors = torch.cat([anchors[:, :4] * 768, anchors[:, 4].unsqueeze(1) * 90], dim=1)
    print('denorm_anchors', denorm_anchors)
    np_anchors = denorm_anchors.numpy()
    iou_anchors = list(map(lambda x: np.ceil(cv2.boxPoints(((x[1], x[0]), (x[2], x[3]), x[4]))), np_anchors))
    print('iou_anchors', len(iou_anchors), iou_anchors[0].shape)
    print('iou_anchors', len(iou_anchors), iou_anchors[0])
    anchor_imgs = torch.from_numpy(
        np.array([cv2.fillConvexPoly(np.zeros((768, 768), dtype=np.uint8), np.int0(a), 1) for a in iou_anchors]).astype(
            float)).float()
    test_tensor = torch.Tensor([[0.0807, 0.2844, 0.0174, 0.0117, -0.8440],
                                [0.3276, 0.0358, 0.0169, 0.0212, -0.1257],
                                [0.3040, 0.2904, 0.0101, 0.0157, -0.5000],
                                [0.0065, 0.2109, 0.0130, 0.0078, -1.0000],
                                [0.1895, 0.1556, 0.0143, 0.0091, -1.0000]])
    b_imgs = uf.fillconvex_rotated_box(test_tensor,(768,768))
    # t = Timer(lambda: comput_rotated_iou(test_tensor, anchor_imgs))
    # print(
    #     f'Consuming {(len(anchors) * np.dtype(float).itemsize * 768 * 768) / 1000000000} Gb on {"GPU" if anchors.is_cuda else "RAM"}')
    # print(f'Averaging {t.timeit(number=100) / 100} seconds per function call')
    print('b_imgs', b_imgs.shape)


    print(comput_rotated_iou(b_imgs, b_imgs))


if __name__ == '__main__':
    main()
