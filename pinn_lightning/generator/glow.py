import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

import normflow as nf
import numpy as np

from .base import GeneratorBase, TransformBase


class Glow(GeneratorBase):
    def __init__(
        self,
        input_shape,
        hidden_channels,
        output_channels, # L in the example code
        
    ) -> None:
        super().__init__()