from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

import normflow as nf
import numpy as np

from .base import GeneratorBase, TransformBase


class RealNVP(GeneratorBase):
    """
        dim: input/output dimension
        depth: depth of RealNVP blocks
        prior_x: prior on output x
        q_z: base distribution of input z (latent variable)
    """
    def __init__(
        self,
        dim,
        depth,
        prior_x,
        q_z,
    ) -> None:
        super().__init__()
        self._build_realnvp(dim, depth, prior_x, q_z)

    def _build_realnvp(self, dim, depth, prior_x, q_z):
        b = torch.Tensor(
            [
                1. if i % 2 == 0 else 0.
                for i in range(dim)
            ]
        )
        flows = []
        for i in range(depth):
            s = nf.nets.MLP(
                [dim, 2*dim, dim],
                init_zeros=True,
            )
            t = nf.nets.MLP(
                [dim, 2*dim, dim],
                init_zeros=True,
            )
            if i % 2 == 0:
                flows.append(nf.flows.MaskedAffineFlow(b, t, s))
            else:
                flows.append(nf.flows.MaskedAffineFlow(1-b, t, s))
            # flows.append(nf.flows.ActNorm(dim))

        self._flow = nf.NormalizingFlow(
            q0=q_z, flows=flows, p=prior_x,
        )

    def forward(self):
        return self.sample()
        
    def sample(self, num_samples=1):
        z, log_q = self._flow.sample(num_samples)
        self._cache = log_q.squeeze()
        return z.squeeze()

    def log_prob(self, x):
        return self._flow.log_prob(x)

    def kl_divergence(self, x):
        out = self._cache
        self._cache = 0
        return out

    def compute_kl_divergence(self):
        return self.kl_divergence(None)
            