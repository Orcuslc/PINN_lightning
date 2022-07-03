import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from .base import GeneratorBase, TransformBase, CombinedGenerator


class GaussianGenerator(GeneratorBase):
    def __init__(self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        prior: Distribution,
        output_transform: TransformBase = None,
        threshold: float = None,
        prob_without_prior: bool = False,
        kl_div_weight: float = 1.0,
        mask: torch.Tensor = None,
        mask_val: float = None,
    ) -> None:
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.gamma = nn.Parameter(
            torch.log(torch.exp(sigma)-1)
        )
        self.prior = prior
        self.output_transform = output_transform
        if threshold is not None:
            self.register_buffer("threshold", torch.tensor([threshold]))
        else:
            self.threshold = None

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.mask_val = mask_val

        self.prob_without_prior = prob_without_prior
        self.kl_div_weight = kl_div_weight

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.gamma))

    def reparameterize(self):
        epsilon = Normal(
            torch.zeros_like(self.mu), torch.ones_like(self.mu)
        ).sample()
        return self.mu + self.sigma*epsilon

    def log_prob(self, x):
        return Normal(self.mu, self.sigma).log_prob(x)

    def kl_divergence(self, x):
        if self.prior is not None:
            log_prob_prior = self.prior.log_prob(x)
        else:
            if not self.prob_without_prior:
                return 0.0
            else:
                log_prob_prior = 0.0
        if self.output_transform is None:
            log_prob_qtheta = self.log_prob(x)
        else:
            log_det = self.output_transform.log_det(x)
            x = self.output_transform.inverse(x)
            log_prob_qtheta = self.log_prob(x) + log_det
        return torch.mean(log_prob_qtheta - log_prob_prior)*self.kl_div_weight

    def forward(self):

        x = self.reparameterize()
        if self.output_transform is not None:
            x = self.output_transform(x)

        if self.threshold is not None:
            x[x < self.threshold] = self.threshold

        if self.mask is not None:
            x[self.mask] = self.mask_val

        self._cache = x
        return x

    def compute_kl_divergence(self):
        out = self.kl_divergence(self._cache)
        self._cache = None
        return out
