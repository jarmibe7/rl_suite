"""
A continuous and discrete policy module for reinforcement learning algorithms.

"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from models.mlp import MLP

class ContinuousPolicy(nn.Module):
    """Gaussian policy for continuous action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int = 2):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim, hidden_layers=hidden_layers)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_layer(h)
        log_std = torch.tanh(self.log_std_layer(h))
        log_std = -0.5 + 0.5 * log_std
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        if deterministic:               # TODO: Is this the right way to handle deterministic actions for continuous policies?
            action = torch.tanh(mean)
            return action, mean, log_std

        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        return action, mean, log_std


class DiscretePolicy(nn.Module):
    """Categorical policy for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int = 2):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim, hidden_layers=hidden_layers)
        self.logits_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        return self.logits_layer(h)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self(obs)
        if deterministic:
            action = logits.argmax(dim=-1)
            return action, logits

        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, logits