"""
for feature in dataloader:
    feature = {"bev": [batch, channel, height, width],
                "box2d_map_l": [batch, channel, height, width],    (channel: y1,x1,y2,x2)
                "box2d_map_m": [batch, channel, height, width],    (channel: y1,x1,y2,x2)
                "box2d_map_s": [batch, channel, height, width],    (channel: y1,x1,y2,x2)
                "object_map_l": [batch, channel, height, width],    (channel: obj)
                "object_map_m": [batch, channel, height, width],    (channel: obj)
                "object_map_s": [batch, channel, height, width],    (channel: obj)
                "box2d": [batch, numbox, channel],     (channel: y1,x1,y2,x2)
                "object": [batch, numbox, channel],
                "box3d": [batch, numbox, channel],  (channel: X,Y,Z,length,width,height) 단위??
                "yaw": [batch, numbox, channel],
                "class": [batch, numbox, channel]
                }
    box2d = torch.zeros(B, C, H, W)
    for box in box_list:
        row, col = box_x/64, box_y/64
        box2d[0, :, row, col] = [y1, x1, y2, x2]
"""
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, anns_dict):
        self.anns_dict = anns_dict
        self.anns_list = self.anns_dict.get("annotations")
        self.len = len(self.anns_list)

    def get_anno_data(self, img_file):
        raise NotImplementedError

    def __getitem__(self, index):
        # self.annotation = torch.from_numpy(self.anno_dict[index].values())
        anns = self.anns_list[index]
        ann_feature = self.get_anno_data(anns)
        return ann_feature

    #
    def __len__(self):
        return self.len
