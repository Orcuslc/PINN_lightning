from typing import List, Any

import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath("./"))))
from pinn_lightning.pinn import PINN


class NTKLossWeightCallback(Callback):
    """
        update_freq: update every n epochs.
    """
    def __init__(
        self, 
        update_freq: int,
        alpha: float = 0.0,
        beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.update_freq = update_freq
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def collect_params(pl_module):
        params = []
        for param in pl_module.parameters():
            params.append()

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: PINN) -> None:
        if pl_module.current_epoch % self.update_freq != 0:
            return

        