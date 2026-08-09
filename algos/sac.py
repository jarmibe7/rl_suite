"""Soft Actor-Critic (SAC) implementation for the RL suite.

This is a compact SAC implementation that supports:
- discrete action spaces (categorical policy over actions)
- continuous action spaces (Gaussian policy with tanh squashing)

https://spinningup.openai.com/en/latest/algorithms/sac.html

It follows the core SAC ideas from OpenAI SpinningUp:
- twin Q-networks and target Q-networks
- entropy reg policy learning
- automatic temp tuning (when enabled)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from gymnasium import spaces

from algos.base import Algorithm
from models.mlp import MLP
from models.policy import ContinuousPolicy, DiscretePolicy

class SAC(Algorithm):
    """Simple SAC implementation with support for discrete and continuous actions."""

    requires_sequences = False

    def __init__(
        self,
        action_space,
        observation_space,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        device: Optional[str] = None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.obs_dim = self._observation_dim()
        self._critic_input_dim = None
        if self.is_discrete:
            self.action_dim = int(action_space.n)
            self.policy = DiscretePolicy(self.obs_dim, self.action_dim, hidden_dim, hidden_layers=self.hidden_layers).to(self.device)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            self.low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
            self.high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
            self.policy = ContinuousPolicy(self.obs_dim, self.action_dim, hidden_dim, hidden_layers=self.hidden_layers).to(self.device)

        self._init_critics(obs_dim=self.obs_dim)

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        if self.automatic_entropy_tuning and not self.is_discrete:
            self.target_entropy = -float(self.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=False, device=self.device)
            self.automatic_entropy_tuning = False

        self._has_seen_obs = False

    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = self._to_tensor(obs)
        if self.is_discrete:
            action, _ = self.policy.sample(obs_tensor, deterministic=deterministic)
            if deterministic:
                return int(action.item())
            return int(action.item())

        action, _, _ = self.policy.sample(obs_tensor, deterministic=deterministic)
        action = action.squeeze(0).detach().cpu().numpy()
        return action

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = self._to_tensor(batch['obs'])
        action = self._to_tensor(batch['action'])
        reward = self._to_tensor(batch['reward']).unsqueeze(-1)
        next_obs = self._to_tensor(batch['next_obs'])
        done = self._to_tensor(batch['done']).float().unsqueeze(-1)

        if self.is_discrete:
            policy_action, policy_logits = self.policy.sample(obs)
            policy_action = policy_action.long()
            log_prob = F.log_softmax(policy_logits, dim=-1)
            selected_log_prob = log_prob.gather(1, policy_action.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_policy_action, next_policy_logits = self.policy.sample(next_obs)
                next_policy_action = next_policy_action.long()
                next_log_prob = F.log_softmax(next_policy_logits, dim=-1)
                next_selected_log_prob = next_log_prob.gather(1, next_policy_action.unsqueeze(-1)).squeeze(-1)
                next_q1 = self.q1_target(torch.cat([next_obs, next_policy_action.float().unsqueeze(-1)], dim=-1))
                next_q2 = self.q2_target(torch.cat([next_obs, next_policy_action.float().unsqueeze(-1)], dim=-1))
                next_q = torch.minimum(next_q1, next_q2)
                q_target = reward + self.gamma * (1 - done) * (next_q - self.alpha * next_selected_log_prob.unsqueeze(-1))

            q1_loss = F.mse_loss(self.q1(torch.cat([obs, action.float().unsqueeze(-1)], dim=-1)), q_target.detach())
            q2_loss = F.mse_loss(self.q2(torch.cat([obs, action.float().unsqueeze(-1)], dim=-1)), q_target.detach())
        else:
            policy_action, mean, log_std = self.policy.sample(obs)
            log_prob = self._log_prob_from_action(policy_action, mean, log_std)

            with torch.no_grad():
                next_policy_action, next_mean, next_log_std = self.policy.sample(next_obs)
                next_log_prob = self._log_prob_from_action(next_policy_action, next_mean, next_log_std)
                next_q1 = self.q1_target(torch.cat([next_obs, next_policy_action], dim=-1))
                next_q2 = self.q2_target(torch.cat([next_obs, next_policy_action], dim=-1))
                next_q = torch.minimum(next_q1, next_q2)
                q_target = reward + self.gamma * (1 - done) * (next_q - self.alpha * next_log_prob.unsqueeze(-1))

            q1_loss = F.mse_loss(self.q1(torch.cat([obs, action], dim=-1)), q_target.detach())
            q2_loss = F.mse_loss(self.q2(torch.cat([obs, action], dim=-1)), q_target.detach())

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        total_q_loss = q1_loss + q2_loss
        total_q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Don't want to backprop policy loss through critic weights
        if self.is_discrete:
            policy_action, policy_logits = self.policy.sample(obs)
            policy_action = policy_action.long()
            log_prob = F.log_softmax(policy_logits, dim=-1)
            selected_log_prob = log_prob.gather(1, policy_action.unsqueeze(-1)).squeeze(-1)
            q1_val = self.q1(torch.cat([obs, policy_action.float().unsqueeze(-1)], dim=-1))
            q2_val = self.q2(torch.cat([obs, policy_action.float().unsqueeze(-1)], dim=-1))
            min_q = torch.minimum(q1_val, q2_val)
            policy_loss = (self.alpha * (-selected_log_prob) - min_q).mean()
            policy_entropy = -torch.mean(selected_log_prob)
        else:
            policy_action, mean, log_std = self.policy.sample(obs)
            log_prob = self._log_prob_from_action(policy_action, mean, log_std)
            q1_val = self.q1(torch.cat([obs, policy_action], dim=-1))
            q2_val = self.q2(torch.cat([obs, policy_action], dim=-1))
            min_q = torch.minimum(q1_val, q2_val)
            policy_loss = (self.alpha * (-log_prob) - min_q).mean()
            policy_entropy = -torch.mean(log_prob)

        policy_loss.backward()
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        self._soft_update_targets()

        return {
            'q1_loss': float(q1_loss.detach().cpu()),
            'q2_loss': float(q2_loss.detach().cpu()),
            'policy_loss': float(policy_loss.detach().cpu()),
            'alpha': float(self.alpha),
            'policy_entropy': float(policy_entropy.detach().cpu()),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'policy': self.policy.state_dict(),
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'q1_target': self.q1_target.state_dict(),
                'q2_target': self.q2_target.state_dict(),
                'log_alpha': self.log_alpha.detach().cpu(),
                'alpha': self.alpha,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.alpha = checkpoint['alpha']

    def reset(self) -> None:
        pass

    def _soft_update_targets(self) -> None:
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _init_critics(self, obs_dim: int) -> None:
        critic_input_dim = obs_dim + self._action_input_dim()
        self.q1 = MLP(critic_input_dim, self.hidden_dim, 1, hidden_layers=self.hidden_layers).to(self.device)
        self.q2 = MLP(critic_input_dim, self.hidden_dim, 1, hidden_layers=self.hidden_layers).to(self.device)
        self.q1_target = MLP(critic_input_dim, self.hidden_dim, 1, hidden_layers=self.hidden_layers).to(self.device)
        self.q2_target = MLP(critic_input_dim, self.hidden_dim, 1, hidden_layers=self.hidden_layers).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.learning_rate)
        self._critic_input_dim = critic_input_dim

    def _action_input_dim(self) -> int:
        return 1 if self.is_discrete else self.action_dim

    def _observation_dim(self) -> int:
        if self.observation_space is None:
            return 1
        shape = getattr(self.observation_space, 'shape', None)
        if shape is None:
            return 1
        if len(shape) == 0:
            return 1
        return int(np.prod(shape))

    def _to_tensor(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _log_prob_from_action(self, action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6).sum(-1, keepdim=True)
        return log_prob
