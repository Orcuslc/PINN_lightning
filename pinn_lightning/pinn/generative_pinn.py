from typing import Any, Union, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./"))))

from pinn_lightning.pinn.base.generator_base import GeneratorBase
from pinn_lightning.pinn.pinn import PINN


class GenerativePINN(PINN):
    """
        forward_module: nn.Module instance. Must have a `generator` property.

        sample_size: the number of samples used to compute the expectation (of both likelihood and KL divergence).
    """
    def __init__(self, forward_module, sample_size):
        super().__init__(forward_module)
        self.sample_size = sample_size

    def _step_one_sample(self, batch):
        losses, weighted_loss = super()._step(batch)
        kl_divergence = self.forward_module.generator.compute_kl_divergence() # neg log likelihood
        losses = torch.hstack([losses, kl_divergence])
        weighted_loss += kl_divergence
        return losses, weighted_loss

    def _step(self, batch):
        losses = []
        weighted_losses = []
        for _ in range(self.sample_size):
            loss, w_loss = self._step_one_sample(batch)
            losses.append(loss)
            weighted_losses.append(w_loss)
        losses = torch.stack(losses).mean(0)
        weighted_losses = torch.stack(weighted_losses).mean(0)
        return losses, weighted_losses

    def configure_tasks(self, tasks):
        super().configure_tasks(tasks)
        self.task_names.append("kl_div")