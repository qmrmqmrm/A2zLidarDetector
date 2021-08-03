import os.path as op
import parameter_pool as params
import numpy as np


class Config:
    class Paths:
        RESULT_ROOT = "/media/dolphin/intHDD/birdnet_data/bv_a2d2/result"
        TFRECORD = op.join(RESULT_ROOT, "pyloader")
        CHECK_POINT = op.join(RESULT_ROOT, "ckpt")

    class Datasets:
        # specific dataset configs MUST have the same items
        class Kitti:
            NAME = "kitti"
            PATH = "/media/ian/Ian4T/dataset/kitti_detection"
            CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist"]
            CATEGORY_REMAP = {"Pedestrian": "Person", "Cyclist": "Bicycle"}
            INPUT_RESOLUTION = (256, 832)  # (4,13) * 64
            CROP_TLBR = [0, 0, 0, 0]  # crop [top, left, bottom, right] or [y1 x1 y2 x2]

        class A2D2:
            NAME = "a2d2"
            PATH = "/media/dolphin/intHDD/birdnet_data/my_a2d2"
            CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist"]
            CATEGORY_REMAP = {}
            INPUT_RESOLUTION = (256, 832)  # (4,13) * 64
            CROP_TLBR = [0, 0, 0, 0]  # crop [top, left, bottom, right] or [y1 x1 y2 x2]

        DATASET_CONFIGS = {"kitti": Kitti}
        TARGET_DATASET = "kitti"

        @classmethod
        def get_dataset_config(cls, dataset):
            return cls.DATASET_CONFIGS[dataset]

    class Tfrdata:
        DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
        MAX_BBOX_PER_IMAGE = 20
        CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"]
        SHARD_SIZE = 2000
        ANCHORS_PIXEL = None  # assigned by set_anchors()

        @classmethod
        def set_anchors(cls):
            basic_anchor = params.Anchor.COCO_YOLOv3
            target_dataset = Config.Datasets.TARGET_DATASET
            dataset_cfg = Config.Datasets.get_dataset_config(target_dataset)
            input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)
            anchor_resolution = np.array(params.Anchor.COCO_RESOLUTION, dtype=np.float32)
            scale = np.min(input_resolution / anchor_resolution)
            Config.Tfrdata.ANCHORS_PIXEL = np.around(basic_anchor * scale, 1)
            print("[set_anchors] anchors in pixel:\n", Config.Tfrdata.ANCHORS_PIXEL)

    class Model:
        class Output:
            FEATURE_SCALES = {"feature_s": 4, "feature_m": 8, "feature_l": 16}
            FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
            NUM_ANCHORS_PER_SCALE = 3
            OUT_CHANNELS = 0  # assigned by set_out_channel()
            OUT_COMPOSITION = ()  # assigned by set_out_channel()
            WEIGHT_DECAY = 0.0001
            WEIGHT_DECAY_BIAS = 0.0001
            WEIGHT_DECAY_NORM = 0.0
            MOMENTUM = 0.9
            MAX_ITER = 30000

            @classmethod
            def set_out_channel(cls):
                num_cats = len(Config.Tfrdata.CATEGORY_NAMES)
                Config.Model.Output.OUT_COMPOSITION = [('bbox', 4), ('object', 1), ('category', num_cats)]
                Config.Model.Output.OUT_CHANNELS = sum([val for key, val in Config.Model.Output.OUT_COMPOSITION])

        class Structure:
            BACKBONE = "Darknet53"
            HEAD = "FPN"
            VP_BINS = 12
            VIEWPOINT = 'true'
            VIEWPOINT_RESIDUAL = 'true'
            HEIGHT_TRAINING = 'true'
            VP_WEIGHT_LOSS = 1.0
            WEIGHTS_HEIGHT = [5.0, 0.5, 10.0]
            PIXEL_MEAN = [0.0, 0.0, 0.0]
            PIXEL_STD = [1.0, 1.0, 1.0]
            ROTATED_BOX_TRAINING = 'true'

            BACKBONE_CONV_ARGS = {"activation": "leaky_relu", "scope": "back"}
            HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}

        class FPN:
            OUTPUT_CHANNELS = 256
            NORM = ''
            FUSE_TYPE = 'sum'

        class RESNET:
            OUT_FEATURES = ['res2', 'res3', 'res4']
            NORM = "FrozenBN"
            STEM_OUT_CHANNELS = 64
            DEPTH = 50
            NUM_GROUPS = 1
            WIDTH_PER_GROUP = 64
            RES2_OUT_CHANNELS = 256
            STRIDE_IN_1X1 = "true"
            RES5_DILATION = 1

        class RPN:
            INPUT_FEATURES = ['p2', 'p3', 'p4']
            MIN_SIZE = 0
            NMS_THRESH = 0.7
            IOU_THRESHOLDS = [0.3, 0.7]
            IOU_LABELS = [0, -1, 1]
            BATCH_SIZE_PER_IMAGE = 256
            POSITIVE_FRACTION = 0.5
            SMOOTH_L1_BETA = 0.0
            LOSS_WEIGHT = 1.0
            PRE_NMS_TOPK_TEST = 6000
            PRE_NMS_TOPK_TRAIN = 12000
            POST_NMS_TOPK_TEST = 1000
            POST_NMS_TOPK_TRAIN = 2000
            BOUNDARY_THRESH = -1
            BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
            ANCHOR_SIZES = [[16, 64, 80]]
            ANCHOR_RATIOS = [[0.5, 1., 2.]]

        class ROI_HEADS:
            INPUT_FEATURES = ['p2', 'p3', 'p4']
            IOU_LABELS = [0, 1]
            IOU_THRESHOLDS = [0.5]
            BATCH_SIZE_PER_IMAGE = 512
            POSITIVE_FRACTION = 0.25
            SCORE_THRESH_TEST = 0.05
            NMS_THRESH_TEST = 0.5
            NUM_CLASSES = 3
            PROPOSAL_APPEND_GT = "true"

        class ROI_BOX_HEAD:
            CLS_AGNOSTIC_BBOX_REG = "false"
            SMOOTH_L1_BETA = 0.0
            ROTATED_BOX_TRAINING = 'true'
            BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
            POOLER_RESOLUTION = 7
            POOLER_SAMPLING_RATIO = 0
            NUM_CONV = 0
            NUM_FC = 2
            CONV_DIM = 256
            FC_DIM = 1024
            NORM = ''

    class Train:
        CKPT_NAME = "birdnet"
        MODE = ["eager", "graph"][1]
        BATCH_SIZE = 4
        TRAINING_PLAN = params.TrainingPlan.A2D2_SIMPLE

    @classmethod
    def summary(cls):
        # return dict of important parameters
        pass

    @classmethod
    def get_img_shape(cls, code="HW", dataset="kitti", scale_div=1):
        dataset_cfg = cls.Datasets.get_dataset_config(dataset)
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
            return cls.Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
        else:
            assert 0, f"Invalid code: {code}"

    @classmethod
    def get_valid_category_mask(cls, dataset="kitti"):
        """
        :param dataset: dataset name
        :return: binary mask e.g. when
            Tfrdata.CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"] and
            Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
            Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
            this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
            but "Bicycle" is not
        """
        dataset_cfg = cls.Datasets.get_dataset_config(dataset)
        renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                              for categ in dataset_cfg.CATEGORIES_TO_USE]

        mask = np.zeros((len(cls.Tfrdata.CATEGORY_NAMES),), dtype=np.int32)
        for categ in renamed_categories:
            if categ in cls.Tfrdata.CATEGORY_NAMES:
                index = cls.Tfrdata.CATEGORY_NAMES.index(categ)
                mask[index] = 1
        return mask


# Config.Tfrdata.set_anchors()
# Config.Model.Output.set_out_channel()
