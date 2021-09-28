import os.path as op
import parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/media/dolphin/intHDD/birdnet_data/bv_a2d2/result"
    TFRECORD = op.join(RESULT_ROOT, "pyloader")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")


class Datasets:
    # specific dataset configs MUST have the same items
    class Standard:
        CATEGORY_NAMES = ["Pedestrian", "Car", "Cyclist"]

    class A2D2:
        NAME = "a2d2"
        PATH = "/media/dolphin/intHDD/birdnet_data/bv_a2d2"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist"]
        CATEGORY_REMAP = {}
        MAX_NUM = 15
        INPUT_RESOLUTION = (640, 640)

    DATASET_CONFIGS = {"a2d2": A2D2}
    TARGET_DATASET = "a2d2"

    @classmethod
    def get_dataset_config(cls, dataset):
        return cls.DATASET_CONFIGS[dataset]


class Hardware:
    DEVICE = ['cuda', 'cpu'][0]


class Scales:
    DEFAULT_FEATURE_SCALES = [4, 8, 16]


class Model:
    MODEL_NAME = 'RCNN'

    class Backbone:
        ARCHITECTURE = "ResNet"
        DEPTH = 50
        NUM_GROUPS = 1
        WIDTH_PER_GROUP = 64
        STEM_OUT_CHANNELS = 64
        RES_OUT_CHANNELS = 256
        STRIDE_IN_1X1 = True
        RES5_DILATION = 1
        NORM = "FrozenBN"
        OUT_SCALES = Scales.DEFAULT_FEATURE_SCALES
        OUT_FEATURES = ["backbone_s2", "backbone_s3", "backbone_s4"]

    class Neck:
        ARCHITECTURE = "FPN"
        OUT_SCALES = Scales.DEFAULT_FEATURE_SCALES
        NORM = ''
        OUT_FEATURES = ["neck_s2", "neck_s3", "neck_s4"]
        OUTPUT_CHANNELS = 256

    class RPN:
        ARCHITECTURE = "RPN"
        OUT_SCALES = Scales.DEFAULT_FEATURE_SCALES
        ANCHOR_SIZES = [16, 64, 80]
        ANCHOR_RATIOS = [0.5, 1., 2.]
        NMS_IOU_THRESH = 0.5
        BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        NUM_PROPOSALS = 2000
        NUM_SAMPLE = 512

    class Head:
        ARCHITECTURE = "FRCNN"
        NUM_FC = 2
        FC_DIM = 1024
        POOLER_RESOLUTION = 7
        POOLER_SAMPLING_RATIO = 0
        ALIGNED = True

    class Output:
        FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
        NUM_ANCHORS_PER_SCALE = 3
        OUT_COMPOSITION = ()  # assigned by set_out_channel()
        WEIGHT_DECAY = 0.0001
        WEIGHT_DECAY_BIAS = 0.0001
        WEIGHT_DECAY_NORM = 0.0
        MOMENTUM = 0.9
        MAX_ITER = 30000

    class Structure:
        VP_BINS = 12
        PIXEL_MEAN = [0.0, 0.0, 0.0]
        PIXEL_STD = [1.0, 1.0, 1.0]
        NUM_CLASSES = 3
        BOX_DIM = 6
        IMAGE_SHAPE = [640, 640]
        STRIDE_SHAPE = Scales.DEFAULT_FEATURE_SCALES
        LOSS_CHANNEL = {'category': 1, 'bbox3d': BOX_DIM, 'yaw': VP_BINS, 'yaw_rads': VP_BINS}


class Train:
    CKPT_NAME = "new_build"
    MODE = ["eager", "graph"][1]
    BATCH_SIZE = 4
    TRAINING_PLAN = params.TrainingPlan.A2D2_SIMPLE


def summary(cls):
    # return dict of important parameters
    pass


def get_img_shape(code="HW", dataset="kitti", scale_div=1):
    dataset_cfg = Datasets.get_dataset_config(dataset)
    imsize = dataset_cfg.INPUT_RESOLUTION
    code = code.upper()
    if code == "H":
        return imsize[0] // scale_div
    elif code == "W":
        return imsize[1] // scale_div
    elif code == "HW":
        return imsize[0] // scale_div, imsize[1] // scale_div
    elif code == "WH":
        return imsize[1] // scale_div, imsize[0] // scale_div
    elif code == "HWC":
        return imsize[0] // scale_div, imsize[1] // scale_div, 3
    elif code == "BHWC":
        return Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
    else:
        assert 0, f"Invalid code: {code}"


def get_valid_category_mask(dataset="kitti"):
    """
    :param dataset: dataset name
    :return: binary mask e.g. when
        Tfrdata.CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"] and
        Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
        Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
        this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
        but "Bicycle" is not
    """
    dataset_cfg = Datasets.get_dataset_config(dataset)
    renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                          for categ in dataset_cfg.CATEGORIES_TO_USE]

    mask = np.zeros((len(Datasets.Standard.CATEGORY_NAMES),), dtype=np.int32)
    for categ in renamed_categories:
        if categ in Datasets.Standard.CATEGORY_NAMES:
            index = Datasets.Standard.CATEGORY_NAMES.index(categ)
            mask[index] = 1
    return mask

# Config.Tfrdata.set_anchors()
# Config.Model.Output.set_out_channel()
