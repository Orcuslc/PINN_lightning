from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class GeneratorBase(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define forward method for nn.Module.
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, *args, **kwargs):
        """
        Define log probability `log q_theta(z)`.
        """
        raise NotImplementedError

    @abstractmethod
    def kl_divergence(self, *args, **kwargs):
        """
        Define KL divergence against prior `log q_theta(z) - log p(z)`.
        """
        raise NotImplementedError


class TransformBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError

    def log_det(self, x):
        """
        log(det(df^-1(x)/dx))
        """
        raise NotImplementedError
