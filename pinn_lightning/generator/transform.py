import torch
import torch.nn as nn

from .base import TransformBase


class ScaledSigmoid(TransformBase):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, z):
        return torch.sigmoid(z)*self.scale

    def inverse(self, x):
        return -torch.log(self.scale/x-1)

    def log_det(self, x):
        return -torch.log(x*(1-x/self.scale))


class DoubleScaledSigmoid(TransformBase):
    def __init__(self, s_inner, s_outer) -> None:
        super().__init__()
        self.s_inner = s_inner
        self.s_outer = s_outer

    def forward(self, z):
        return torch.sigmoid(z/self.s_inner)*self.s_outer

    def inverse(self, x):
        return -self.s_inner*torch.log(self.s_outer/x-1)

    def log_det(self, x):
        return -torch.log(x/self.s_inner*(1-x/self.s_outer))


class SoftPlus(TransformBase):
    def __init__(self, beta, threshold=None) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, z):
        return 1/self.beta*torch.log1p(torch.exp(self.beta*z))
        # return nn.functional.softplus(z, self.beta, self.threshold)

    def inverse(self, x):
        return 1/self.beta*torch.log(torch.exp(self.beta*x)-1)

    def log_det(self, x):
        return torch.log1p(1/(torch.exp(self.beta*x)-1))
