
import torch
from torch.utils.data import DataLoader
from dataloader.a2z_loader import A2D2Loader
from dataloader.sampler import TrainingSampler


def loader_factory(dataset_name):
    if dataset_name == "A2D2":
        return None


def get_dataset(path, batch_size):
    loader = A2D2Loader(path)
    sampler = TrainingSampler(len(loader))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    data_loader = DataLoader(dataset=loader,
                             batch_sampler=batch_sampler,
                             collate_fn=trivial_batch_collator,
                             num_workers=2)

    print("\n")
    return data_loader

def trivial_batch_collator(batch):
    return batch