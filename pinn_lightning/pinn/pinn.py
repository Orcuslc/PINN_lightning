from typing import Any, Union, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./"))))

from pinn_lightning.pinn.base.pinn_base import PINNBase
from pinn_lightning.pinn.task import Task
from pinn_lightning.utils.utils import flatten_nested_list


class PINN(PINNBase):
    """
    Basic PINN implementation.
    """
    def __init__(self, forward_module):
        super().__init__()
        self.forward_module = forward_module
        self._loss_fn = None
        self._loss_weights = None

    def get_outputs(self, batch, dataset_idx):
        """
        Define outputs including value and residue outputs. Must be overloaded by subclasses.
        """
        raise NotImplementedError

    def forward(self, *args):
        return self.forward_module(*args)

    def configure_tasks(
        self,
        tasks: List[Task],
    ):
        """
        Configure tasks.

        """
        self.tasks = tasks
        self.loss_fns = flatten_nested_list([task.loss_fns for task in tasks])
        self.loss_weights = flatten_nested_list([task.loss_weights for task in tasks])

    def _split_io(self, batch):
        assert hasattr(self, "tasks")
        inputs = [b[:task.n_input] for (b, task) in zip(batch, self.tasks)]
        targets = [b[task.n_input:] for (b, task) in zip(batch, self.tasks)]
        targets = flatten_nested_list(targets)
        return inputs, targets

    def _step(self, batch):
        batch_input, batch_target = self._split_io(batch)
        batch_output = self.get_output(batch_input)
        assert len(batch_output) == len(batch_target)

        losses = torch.hstack([
            loss_fn(target, output) 
            for (loss_fn, target, output) in zip(
                self.loss_fns, batch_target, batch_output
            )
        ])
        weighted_loss = torch.stack(
            [weight*loss for (weight, loss) in zip(
                self.loss_weights, losses
            )]
        ).sum()
        return losses, weighted_loss

    def training_step(self, batch, batch_idx):
        losses, weighted_loss = self._step(batch)
        self.log(f"train_loss", weighted_loss, on_step=False, on_epoch=True)
        for i in range(len(self.tasks)):
            self.log(f"train_loss_{i}", losses[i], on_step=False, on_epoch=True)
        return weighted_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        losses, weighted_loss = self._step(batch)
        self.log(f"valid_loss", weighted_loss, on_step=False, on_epoch=True)
        for i in range(len(self.tasks)):
            self.log(f"valid_loss_{i}", losses[i], on_step=False, on_epoch=True)
        return weighted_loss

    @property
    def param_groups(self):
        return [self.forward_module.parameters()]

    def configure_optimizers(self):
        assert hasattr(self, "params")
        return torch.optim.Adam(self.params)

    def configure_lrs(
        self, 
        lr: Union[float, List[float]],
        weight_decay: Union[float, List[float]],
    ):
        if isinstance(lr, list):
            assert isinstance(weight_decay, list) and len(weight_decay) == len(lr)
            assert len(lr) == len(self.param_groups)
        else:
            assert not isinstance(weight_decay, list)
            lr = [lr for _ in range(len(self.param_groups))]
            weight_decay = [weight_decay for _ in range(len(self.param_groups))]
        
        params = [
            {
                "params": param,
                "lr": _lr,
                "weight_decay": _weight_decay
            }
            for (param, _lr, _weight_decay) in zip(
                self.param_groups, lr, weight_decay
            )
        ]
        self.params = params


class InversePINN(PINN):
    """
    Basic PINN implementation for inverse problems.
    """
    def __init__(self, forward_module, inverse_module):
        super().__init__(forward_module)
        self.inverse_module = inverse_module

    @property
    def param_groups(self):
        return [self.forward_module.parameters(), self.inverse_module.parameters()]