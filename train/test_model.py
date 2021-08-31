from config import Config as cfg
import settings
from model.model_factory import build_model
from dataloader.loader_factory import get_dataset


def test_model(dataset_name="a2d2"):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    data_loader = get_dataset(dataset_name, 'train', batch_size)
    model = build_model(*cfg.Model.Structure.NAMES)
    train_loader_iter = iter(data_loader)
    steps = len(train_loader_iter)
    '''
    pred :{'head_class_logits': torch.Size([batch * 512, 4])
          'bbox_3d_logits': torch.Size([batch * 512, 12])
          'head_yaw_logits': torch.Size([batch * 512, 36])
          'head_yaw_residuals': torch.Size([batch * 512, 36])
          'head_height_logits': torch.Size([batch * 512, 6])

          'head_proposals': [{'proposal_boxes': torch.Size([512, 4])
                              'objectness_logits': torch.Size([512])
                              'gt_category': torch.Size([512, 1])
                              'bbox2d': torch.Size([512, 4])
                              'bbox3d': torch.Size([512, 6])
                              'object': torch.Size([512, 1])
                              'yaw': torch.Size([512])
                              'yaw_rads': torch.Size([512])} * batch]

          'rpn_proposals': [{'proposal_boxes': torch.Size([2000, 4]),
                            'objectness_logits': torch.Size([2000])} * batch]

          'pred_objectness_logits' : [torch.Size([batch, 557568(176 * 352 * 9)]),
                                      torch.Size([batch, 139392(88 * 176 * 9)]),
                                      torch.Size([batch, 34848(44 * 88 * 9)])]

          'pred_anchor_deltas' : [torch.Size([batch, 557568(176 * 352 * 9), 4]),
                                  torch.Size([batch, 139392(88 * 176 * 9), 4]),
                                  torch.Size([batch, 34848(44 * 88 * 9), 4])]

          'anchors' : [torch.Size([557568(176 * 352 * 9), 4])
                       torch.Size([139392(88 * 176 * 9), 4])
                       torch.Size([34848(44 * 88 * 9), 4])]
                  }
    '''
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

                if isinstance(features, dict):
                    for f_k, feat in features.items():
                        if isinstance(feat, list):
                            for j, f in enumerate(feat):
                                print("prediction", key, j, f_k, f.shape)
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
