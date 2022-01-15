from typing import List, Union
import torch
from torch.nn.modules.loss import _Loss


class Task:
    def __init__(
        self,
        n_input: int,
        n_output: int,
        loss_fns: Union[_Loss, List[_Loss]],
        loss_weights: Union[float, List[float]],
    ):
        self.n_input = n_input
        self.n_output = n_output

        if isinstance(loss_fns, list):
            assert len(loss_fns) == n_output
            self.loss_fns = loss_fns
        else:
            self.loss_fns = [loss_fns for _ in range(n_output)]
        
        if isinstance(loss_weights, list):
            assert len(loss_weights) == n_output
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [loss_weights for _ in range(n_output)]