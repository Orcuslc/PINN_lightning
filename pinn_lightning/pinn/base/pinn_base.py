from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl


class PINNBase(ABC, pl.LightningModule):
    """
    Base class for PINN-based models.
    """
    @abstractmethod
    def get_outputs(self, *args, **kwargs):
        """
        Define outputs for PINN, including both value (Dirichlet) outputs and residue outputs (lhs - rhs).
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define `forward` method for LightningModule.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Define `training_step` method for LightningModule.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """
        Define `configure_optimizers` method for LightningModule.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_tasks(self):
        """
        Define tasks for PINN modules.
        """
        raise NotImplementedError