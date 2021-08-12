from config import Config as cfg
import settings
from model.model_factory import build_model
from dataloader.loader_factory import get_dataset


def test_model(dataset_name="a2d2"):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    data_loader = get_dataset(dataset_name, batch_size)
    model = build_model(*cfg.Model.Structure.NAMES)
    train_loader_iter = iter(data_loader)
    steps = len(train_loader_iter)
    for step in range(steps):
        print("----- index:", step)
        features, image_file = next(train_loader_iter)
        pred = model(features)
        for key, features in  pred.items():
            if isinstance(features, list):
                for k, feat in enumerate(features):
                    if isinstance(feat, dict):
                        for f_k, f in feat.items():
                            print("prediction", key, k, f_k, f.shape)
                    else:
                        print("prediction", key, k, feat.shape)
            else:
                print("prediction", key, features.shape)
            # for key, feat in features.items():
            #     print("features", key, feat.shape)
        # for i, proposal in enumerate(rpn_proposals):
        #     for key, feat in proposal.items():
        #         print("proposal", i, key, feat.shape)
        # for key, feats in auxiliary.items():
        #     if isinstance(feats, list):
        #         for k, feat in enumerate(feats):
        #             print("auxil", key, k, feat.shape)
        #     else:
        #         print("auxil", key, feats.shape)
        # for key, feats in pred.items():
        #     if isinstance(feats, list):
        #         for k, feat in enumerate(feats):
        #             for f_k, f in feat.items():
        #                 print("prediction", key, k, f_k, f.shape)
        #
        #


if __name__ == "__main__":
    test_model()
