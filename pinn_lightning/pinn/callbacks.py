from typing import List, Any

import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class NTKLossWeightCallback(Callback):
    def __init__(
        self, 
        data_samples: List[Any], 
        update_freq: int,
        alpha: float = 0.0,
        beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.data_samples = data_samples
        self.update_freq = update_freq
        self.alpha = alpha
        self.beta = beta

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass