import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from .base import GeneratorBase, TransformBase


class CombinedGenerator(GeneratorBase):
    def __init__(self, generators) -> None:
        super().__init__()
        self.generators = nn.ModuleList(generators)

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


class GaussianGenerator(GeneratorBase):
    def __init__(self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        prior: Distribution,
        output_transform: TransformBase = None,
        threshold: float = 1e-16,
        prob_without_prior: bool = False,
        kl_div_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.gamma = nn.Parameter(
            torch.log(torch.exp(sigma)-1)
        )
        self.prior = prior
        self.output_transform = output_transform
        self.register_buffer("threshold", torch.tensor([threshold]))
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

        x[x < self.threshold] = self.threshold
        self._cache = x
        return x

    def compute_kl_divergence(self):
        out = self.kl_divergence(self._cache)
        self._cache = None
        return out
