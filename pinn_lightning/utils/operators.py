import torch
import numpy as np

def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True, only_inputs = True)[0]

def _grad(u, x):
    """
        u has the same dimension with x or is one-dimension higher than x;
        
        e.g.,
            - u has shape [N, 1],
            - x has shape [N, 1],

        or,
            - u has shape [N, D, 1],
            - x has shape [N, 1]

        The second scenario happens when u is the output of a system equation u = u(x).
        In this scenario, the output 
            grad = [du[:, 0:1, :]/dx, du[:, 1:2, :]/dx, ...]
        has shape [N, D, 1].
        
    """
    if len(u.shape) == len(x.shape):
        return _grad(u, x)
    grads = []
    for i in range(u.shape[1]):
        grads.append(_grad(u[:, i, :], x))
    return torch.stack(grads, dim=1)

def directional_grad(u, *args, direction=None):
    grads = [
        direction[:, i:i+1]*grad(u, x) 
        for (i, x) in enumerate(args)
    ]
    return torch.stack(grads).sum(0)

def laplacian_2d(u, x, y):
    du_dx = grad(u, x)
    du_dy = grad(u, y)
    du_dxx = grad(du_dx, x)
    du_dyy = grad(du_dy, y)
    return du_dxx + du_dyy, du_dx, du_dy, du_dxx, du_dyy