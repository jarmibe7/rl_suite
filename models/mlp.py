"""
A multilayer perceptron (MLP) used for both actor and critic networks in RL algorithms.
"""
import torch
from torch import nn

class MLP(nn.Module):
    """Small MLP used by both actor and critics."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int = 2):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be >= 0")

        layers = []
        prev_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)