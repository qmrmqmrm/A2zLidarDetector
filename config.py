import os.path as op
import parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/oem/workspace/datasets/bv/result"
    TFRECORD = op.join(RESULT_ROOT, "pyloader")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")


class Datasets:
    # specific dataset configs MUST have the same itemsS
    class Standard:
        CATEGORY_NAMES = ["Pedestrian", "Car", "Cyclist"]

    class A2D2:
        NAME = "a2d2"
        PATH = "/home/oem/workspace/datasets/bv"
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
        NORM = "BN"
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
        NMS_IOU_THRESH = 0.4
        MATCH_THRESHOLD = [0.1, 0.4]
        NMS_SCORE_THRESH = 0.3
        NUM_PROPOSALS = [3000, 1000]
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
        MINOR_CTGR = False

    class Structure:
        VP_BINS = 6
        PIXEL_MEAN = [0.0, 0.0, 0.0]
        PIXEL_STD = [1.0, 1.0, 1.0]
        NUM_CLASSES = 3
        BOX_DIM = 6
        IMAGE_SHAPE = [640, 640]
        STRIDE_SHAPE = Scales.DEFAULT_FEATURE_SCALES
        LOSS_CHANNEL = {'category': 1, 'bbox3d_delta': BOX_DIM, 'yaw_cls': VP_BINS, 'yaw_rads': VP_BINS}


class Train:
    CKPT_NAME = "check_bg_mask"
    MODE = ["eager", "graph"][0]
    BATCH_SIZE = 2
    TRAINING_PLAN = params.TrainingPlan.A2D2_SIMPLE


class Loss:
    ALIGN_IOU_THRESHOLD = [0.1, 0.4]
    ANCHOR_IOU_THRESHOLD = [0.1, 0.6]


class NMS:
    MAX_OUT = [100, 100, 100]
    IOU_THRESH = [0.2, 0.2, 0.2]
    SCORE_THRESH = [0.3, 0.3, 0.3]


class Validation:
    TP_IOU_THRESH = [0.3, 0.3, 0.3, 0.3]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"


class Logging:
    COLNUMS = ["ciou_loss", "object_loss", "maj_cat_loss",
               "dist_loss", "pos_obj", "neg_obj", "iou", "box_hw", "box_yx", "true_class", "false_class"]
    USE_ANCHOR = [True, False][0]
    COLUMNS_TO_MEAN = ["anchor", "category", "ciou_loss", "object_loss", "maj_cat_loss", "dist_loss", "pos_obj",
                       "neg_obj", "iou", "box_hw", "box_yx", "true_class", "false_class"]
    COLUMNS_TO_SUM = ["anchor", "category", "trpo", "grtr", "pred"]


class Debug:
    DEBUG = False


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
