from ast import Call
from typing import Callable, List

import torch
from torch.utils.data import (
    Dataset, DataLoader, random_split,
)
from pytorch_lightning import LightningDataModule


class ConcatDataLoader():
    """
    Class to concatenate multiple dataloaders
    
    source: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/35
    """
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)


class PINNDataModule(LightningDataModule):
    """
    Basic PINN data module.
    """
    def __init__(
        self, 
        datasets: List[Dataset],
        collate_fns: List[Callable],
        valid_splits: List[float],
        batch_sizes: List[int]
    ):
        assert len(datasets) == len(collate_fns) and len(collate_fns) == len(valid_splits) and len(valid_splits) == len(batch_sizes)
        self.datasets = datasets
        self.collate_fns = collate_fns
        self.valid_splits = valid_splits
        self.batch_sizes = batch_sizes

    def prepare_data(self) -> None:
        pass

    @staticmethod
    def train_valid_split(dataset, valid_split):
        len_valid = int(valid_split*len(dataset))
        train_dataset, valid_dataset = random_split(
            dataset, 
            [len(dataset)-len_valid, len_valid],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, valid_dataset

    def setup(self, stage=None):
        datasets_split = [
            self.train_valid_split(dataset, valid_split)
            for (dataset, valid_split) in zip(
                self.datasets, self.valid_splits
            )
        ]
        self.train_datasets = [pair[0] for pair in datasets_split]
        self.valid_datasets = [pair[1] for pair in datasets_split]

    def train_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=collate_fn,
                shuffle=True,
            )
            for (dataset, batch_size, collate_fn) in zip(
                self.train_datasets, self.batch_sizes, self.collate_fns
            )
        ]
        return ConcatDataLoader(*dataloaders)

    def val_dataloader(self):
        dataloaders = [
            DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=collate_fn,
                shuffle=True,
            )
            for (dataset, batch_size, collate_fn) in zip(
                self.valid_datasets, self.batch_sizes, self.collate_fns
            )
        ]
        return ConcatDataLoader(*dataloaders)


