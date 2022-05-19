import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

import normflow as nf
import numpy as np

from .base import GeneratorBase, TransformBase


class CombinedGenerator(GeneratorBase):
    def __init__(self, generators) -> None:
        super().__init__()
        self.generators = nn.ModuleList(generators)

    def sample(self):
        return self.forward()

    def forward(self):
        outputs = []
        for generator in self.generators:
            outputs.append(generator())
        return outputs

    def log_prob(self, *args, **kwargs):
        res = 0.0
        for generator in self.generators:
            res += generator.log_prob(*args, **kwargs)
        return res

    def kl_divergence(self, *args, **kwargs):
        raise NotImplementedError

    def compute_kl_divergence(self):
        res = 0.0
        for generator in self.generators:
            res += generator.compute_kl_divergence()
        return res


class DistributionWrapper(nf.distributions.BaseDistribution):
    def __init__(self, distribution) -> None:
        super().__init__()
        self.distribution = distribution

    def forward(self, num_samples=1):
        out = self.distribution.sample()
        log_p = self.distribution.log_prob(out)
        return out, log_p

    def log_prob(self, x):
        return self.distribution.log_prob(x)

