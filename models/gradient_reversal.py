"""
Gradient Reversal Layer (GRL) for Domain-Adversarial Training (DANN).
Paper: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse the gradient sign and scale by lambda
        return grad_output.neg() * ctx.lambd, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def set_lambda(self, lambd: float):
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambd)


def calc_lambda(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    Gradual lambda scheduling strategy:
    lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] is the relative training progress.
    """
    if total_steps <= 0:
        return 1.0
    p = float(current_step) / float(total_steps)
    p = min(max(p, 0.0), 1.0)
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
