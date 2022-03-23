from typing import Callable, List, Union, Optional
from itertools import cycle

import os, sys
from xmlrpc.client import Boolean
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./"))))

from pinn_lightning.utils.utils import convert_to_list

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


class CycleDataLoader():
    """
    Class to cycle one dataloader and avoid StopIteration
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return cycle(iter(self.dataloader))


class PINNDataModule(LightningDataModule):
    """
    Basic PINN data module.
    """
    def __init__(
        self, 
        datasets: List[Dataset],
        collate_fns: List[Callable],
        valid_splits: List[float],
        batch_sizes: List[int],
        shuffle: List[bool],
        do_not_split_dataset_index: Optional[Union[int, List[int]]] = None,
        cycle_dataloader_index: Optional[Union[int, List[int]]] = None,
        pin_memory: bool = True
    ):
        assert len(datasets) == len(collate_fns) \
            and len(collate_fns) == len(valid_splits) \
            and len(valid_splits) == len(batch_sizes)
        self.datasets = datasets
        self.collate_fns = collate_fns
        self.valid_splits = valid_splits
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.do_not_split_dataset_index = convert_to_list(do_not_split_dataset_index)
        self.cycle_dataloader_index = convert_to_list(cycle_dataloader_index)
        self.pin_memory = pin_memory

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
        datasets_split = []
        for (index, (dataset, valid_split)) in enumerate(
            zip(self.datasets, self.valid_splits)
        ):
            if index in self.do_not_split_dataset_index:
                datasets_split.append([dataset, dataset])
            else:
                datasets_split.append(
                    self.train_valid_split(dataset, valid_split)
                )
        self.train_datasets = [pair[0] for pair in datasets_split]
        self.valid_datasets = [pair[1] for pair in datasets_split]

    def train_dataloader(self):
        dataloaders = []
        for index, (dataset, batch_size, collate_fn, shuffle) in enumerate(
            zip(self.train_datasets, self.batch_sizes, self.collate_fns, self.shuffle)
        ):
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=collate_fn,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
            )
            if index in self.cycle_dataloader_index:
                dataloader = CycleDataLoader(dataloader)
            dataloaders.append(dataloader)
        return ConcatDataLoader(*dataloaders)

    def val_dataloader(self):
        dataloaders = []
        for index, (dataset, batch_size, collate_fn, shuffle) in enumerate(
            zip(self.valid_datasets, self.batch_sizes, self.collate_fns, self.shuffle)
        ):
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=collate_fn,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
            )
            if index in self.cycle_dataloader_index:
                dataloader = CycleDataLoader(dataloader)
            dataloaders.append(dataloader)
        return ConcatDataLoader(*dataloaders)


class PINNDataModuleWithoutValidation(LightningDataModule):
    """
        PINN data module without validation.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        collate_fns: List[Callable],
        batch_sizes: List[int],
        shuffle: List[bool],
        cycle_dataloader_index: Optional[Union[int, List[int]]] = None,
        pin_memory: bool = True,
    ):
        assert len(datasets) == len(collate_fns) \
            and len(collate_fns) == len(batch_sizes)
        self.datasets = datasets
        self.collate_fns = collate_fns
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.cycle_dataloader_index = convert_to_list(cycle_dataloader_index)
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.train_datasets = self.datasets

    def train_dataloader(self):
        dataloaders = []
        for index, (dataset, batch_size, collate_fn, shuffle) in enumerate(
            zip(self.train_datasets, self.batch_sizes,
                self.collate_fns, self.shuffle)
        ):
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
            )
            if index in self.cycle_dataloader_index:
                dataloader = CycleDataLoader(dataloader)
            dataloaders.append(dataloader)
        return ConcatDataLoader(*dataloaders)