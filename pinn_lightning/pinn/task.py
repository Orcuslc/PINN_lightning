from typing import List, Union, Callable

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("./"))))

from pinn_lightning.utils.utils import convert_to_list

import torch
from torch.nn.modules.loss import _Loss

class Task:
    """
        If `adaptive_weights` is set to be True, `loss_weights` is ignored.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        loss_fns: Union[_Loss, List[_Loss]],
        loss_weights: Union[float, List[float]],
        names: Union[str, List[str]] = "",
    ):
        self.n_input = n_input
        self.n_output = n_output
        self.loss_fns = convert_to_list(loss_fns, n_output, assertion=True)
        self.loss_weights = convert_to_list(loss_weights, n_output, assertion=True)
        self.loss_weights = [torch.Tensor([w]) for w in self.loss_weights]
        self.names = convert_to_list(names, n_output, assertion=True)
        if len(self.names) > 1 and self.names[0] == self.names[1]:
            for i in range(len(self.names)):
                self.names[i] = self.names[i] + "_" + str(i)