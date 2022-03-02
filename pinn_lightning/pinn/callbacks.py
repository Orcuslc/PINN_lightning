from typing import List, Any

import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath("./"))))
from pinn_lightning.pinn import PINN
from pinn_lightning.utils.operators import grad


class NTKLossWeightCallback(Callback):
    """
        max_samples: maximum samples used to compute the NTK.
        update_freq: update every n epochs.
    """
    def __init__(
        self, 
        max_samples: List[int],
        update_freq: int,
        alpha: float = 0.0,
        beta: float = 0.9,
        min_val: float = 0.0,
        max_val: float = 1.0,
        normalize_initialization: bool = True,
        skip_first: bool = True,
    ) -> None:
        super().__init__()
        self.max_samples = max_samples
        self.update_freq = update_freq
        self.alpha = alpha
        self.beta = beta
        self.min_val = min_val
        self.max_val = max_val
        self.normalize_intialization = normalize_initialization
        self.skip_first = skip_first
        self.base_losses = None

    def prepare_batch(self, batch, device):
        prepared = []
        for b, max_sample in zip(batch, self.max_samples):
            prepared.append(
                [
                    bi[:max_sample].to(device)
                    for bi in b
                ]
            )
        return prepared

    def normalize_weights(
        self,
        weights: torch.Tensor,
     ):
        return weights/weights.sum()*len(weights)

    def compute_loss_and_ntk_diag(
        self, 
        pl_module: PINN, 
        data_module: pl.LightningDataModule
    ):
        for batch in data_module.train_dataloader():
            batch = self.prepare_batch(batch, pl_module.device)
            batch_input, batch_target = pl_module._split_io(batch)
            outputs = pl_module.get_outputs(batch_input)
            break

        losses = [
            loss_fn(target, output)*weight
            for (loss_fn, target, output, weight) in zip(
                pl_module.loss_fns, batch_target, outputs, pl_module.loss_weights
            )
        ]

        ntk_diag = []
        for output in outputs:
            grad_norm_for_samples = [] # grad for each input sample
            for i in range(output.shape[0]):
                grad_output = torch.zeros_like(output).type_as(output)
                grad_output[i] = 1.0
                grads = torch.autograd.grad(
                    output,
                    pl_module.parameters(),
                    grad_outputs=grad_output,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_norm_for_samples.append(
                    torch.stack(
                        [
                            torch.sum(x*x) 
                            if x is not None
                            else torch.zeros((1,)).sum().to(pl_module.device)
                            for x in grads
                        ]
                    ).sum()
                )
            ntk_diag.append(
                torch.stack(grad_norm_for_samples).sum()
            )
        return losses, ntk_diag

    def compute_weights(
        self, 
        losses: List[torch.Tensor], 
        ntk_diag: List[torch.Tensor],
        cur_weights: List[torch.Tensor],
    ):
        ntk_diag = torch.hstack(ntk_diag).detach()
        weights = ntk_diag.sum() / ntk_diag
        
        losses = torch.hstack(losses).detach()
        if self.base_losses is None:
            self.base_losses = losses
        scores = losses / self.base_losses
        mean_score = scores.mean()
        scores = (scores / mean_score)**self.alpha

        ntk_weights = weights*scores
        ntk_weights = torch.minimum(
            ntk_weights, 
            torch.ones_like(ntk_weights).type_as(ntk_weights)*self.max_val
        )
        ntk_weights = torch.maximum(
            ntk_weights,
            torch.ones_like(ntk_weights).type_as(ntk_weights)*self.min_val
        )
        ntk_weights = self.normalize_weights(ntk_weights)
        
        cur_weights = torch.hstack(cur_weights)
        new_weights = (1-self.beta)*cur_weights + self.beta*ntk_weights
        return new_weights

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: PINN) -> None:
        if pl_module.current_epoch == 0 and self.normalize_intialization:
            weights = self.normalize_weights(
                torch.hstack(pl_module.loss_weights)
            )
            pl_module.loss_weights = [w for w in weights]

        if pl_module.current_epoch == 0 and self.skip_first:
            return

        if pl_module.current_epoch % self.update_freq != 0:
            return

        losses, ntk_diag = self.compute_loss_and_ntk_diag(pl_module, trainer.datamodule)
        new_weights = self.compute_weights(losses, ntk_diag, pl_module.loss_weights)
        pl_module.loss_weights = [w for w in new_weights]
        for (weight, task_name) in zip(
            pl_module.loss_weights, pl_module.task_names
        ):
            pl_module.log(f"weight_{task_name}", weight, on_epoch=True)
        

class AdaptiveSampleWeight(Callback):
    def __init__(
        self, 
    ) -> None:
        super().__init__()