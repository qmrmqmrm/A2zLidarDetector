import torch
from torchvision.ops import boxes as box_ops


def test_nms():
    bbox2d = [
        [[10, 40, 20, 60],
         [11, 45, 21, 60],
         [15, 45, 20, 60],
         [17, 43, 25, 63],
         [9, 39, 28, 65]],

        [[10, 40, 20, 60],
         [11, 45, 21, 60],
         [15, 45, 20, 60],
         [17, 43, 25, 63],
         [9, 39, 28, 65]]

    ]
    scores = [
        [0.5, 0.4, 0.8, 0.3, 0.1],
        [0.8, 0.9, 0.4, 0.5, 0.2]
    ]
    idxs = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]
    bbox2d = torch.tensor(bbox2d, dtype=torch.int64).view(-1, 4)
    scores = torch.tensor(scores, dtype=torch.float32).view(-1)
    idxs = torch.tensor(idxs, dtype=torch.float32).view(-1)

    keep = box_ops.batched_nms(bbox2d.float(), scores, idxs, 0.1)
    print(keep)
    print(keep % 5)
    print(keep // 5)


def sort_test():
    a = torch.tensor([[[0.2162],
                       [0.5793],
                       [0.5071]], [[-0.2162],
                                   [0.5793],
                                   [-0.5071]]])

    a_, inds = torch.sort(a, dim=1, descending=True)
    print(a.shape)
    print(a)
    print(a_)
    print(inds)


def test_min_max():
    a = torch.tensor(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
          [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]]]
            , [[[101, 102, 103, 104], [105, 106, 107, 108], [109, 110, 111, 112]],
               [[113, 114, 115, 116], [117, 118, 119, 120], [121, 122, 123, 124]],
               [[125, 126, 127, 128], [129, 130, 131, 132], [133, 134, 135, 136]]]])
    print(a)
    print(a.shape)


if __name__ == '__main__':
    test_min_max()
