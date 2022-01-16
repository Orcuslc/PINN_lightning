import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback


class NTKLossWeightCallback(Callback):
