import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_nonlinear_net(structure, activation_fn):
    net = []
    for (f_in, f_out) in zip(structure[:-1], structure[1:]):
        net.append(nn.Linear(f_in, f_out))
        net.append(activation_fn())
    return nn.Sequential(*net[:-1])


class Swish(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(x)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class DenseNet(nn.Module):
    def __init__(
        self, 
        structure, 
        activation_fn = nn.Tanh, 
        input_hook = None,
        output_hook = None,
    ):
        super().__init__()
        self.net = build_nonlinear_net(structure, activation_fn)
        self.input_hook = input_hook or self._input_hook
        self.output_hook = output_hook or self._output_hook

    def forward(self, *args):
        x = torch.hstack(args)
        x = self.input_hook(x)
        output = self.net(x)
        output = self.output_hook(x, output)
        return output

    def _input_hook(self, inputs):
        return inputs

    def _output_hook(self, inputs, outputs):
        return outputs


class DeepONet(nn.Module):
    """
        y = F(x; z) = \sum_i b_i(x)c_i(x; z)   ? or c_i(z)

        basis_net and coefficient_net can have higher dimentional outputs:
            (x_sample, freq_j, basis_i)
    """
    def __init__(
        self,
        basis_net: nn.Module,
        coefficient_net: nn.Module,
    ):
        super().__init__()
        self.basis_net = basis_net
        self.coefficient_net = coefficient_net

    def forward(self, x, z):
        """Args:
            - x: tensor (N, K). spatial points
            - z: tensor (N, P). parameter values
        """
        basis = self.basis_net(x)
        # coefficient = self.coefficient_net(torch.hstack([x, z]))
        coefficient = self.coefficient_net(z)
        return torch.sum(basis*coefficient, dim=-1, keepdim=True)
        