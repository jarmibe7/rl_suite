"""DreamerV1 (Hafner et al. 2020, "Dream to Control: Learning Behaviors by
Latent Imagination"): https://arxiv.org/abs/1912.01603

Learns a latent world model (RSSM) from real experience via a reconstruction
+ reward + KL ELBO, then learns an actor and a critic purely from rollouts
imagined inside the model, backpropagating value gradients through the
(reparameterized) latent dynamics and into the actor.

Reuses the Gaussian RSSM already implemented in models/rssm.py as the world
model; this file only adds the imagination-based actor-critic on top of it.

Simplifications vs. the paper: no learned discount/continue predictor (a
fixed gamma is used for the whole imagined horizon instead).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from algos.base import Algorithm
from models.rssm import RSSM, RSSMLoss
from models.policy import ContinuousPolicy
from models.mlp import MLP


class Dreamer(Algorithm):
    """DreamerV1: RSSM world model + actor-critic trained via latent imagination."""

    requires_sequences = True

    def __init__(
        self,
        action_space,
        observation_space,
        conv_params: Dict,
        in_channels: int = 3,
        stochastic_size: int = 30,
        deterministic_size: int = 200,
        enc_latent_size: int = 200,
        hidden_dim: int = 200,
        hidden_layers: int = 2,
        imagination_horizon: int = 15,
        gamma: float = 0.99,
        lam: float = 0.95,
        model_lr: float = 6e-4,
        actor_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        recon_mult: float = 1.0,
        reward_mult: float = 1.0,
        beta: float = 1.0,
        free_nats: float = 3.0,
        kld_anneal_mode: str = 'const',
        anneal_steps: int = 100000,
        device: Optional[str] = None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.action_dim = int(np.prod(action_space.shape))
        self.act_low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)

        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lam = lam

        self.rssm = RSSM(
            enc_latent_size=enc_latent_size,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            control_size=self.action_dim,
            pred_length=imagination_horizon,
            conv_params=conv_params,
            device=self.device,
            img_channel_count=in_channels,
            reward_size=1,
        ).to(self.device)
        self.rssm_loss = RSSMLoss(
            num_epochs=anneal_steps,
            loss_params={
                'recon_mult': recon_mult,
                'reward_mult': reward_mult,
                'beta': beta,
                'free_nats': free_nats,
                'kld_anneal_mode': kld_anneal_mode,
            },
        )

        feat_dim = deterministic_size + stochastic_size
        self.actor = ContinuousPolicy(feat_dim, self.action_dim, hidden_dim, hidden_layers=hidden_layers).to(self.device)
        self.value_net = MLP(feat_dim, hidden_dim, 1, hidden_layers=hidden_layers).to(self.device)

        self.rssm_optimizer = torch.optim.Adam(self.rssm.parameters(), lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)

        self._update_count = 0
        self._h = None
        self._z = None
        self._prev_action = None

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_image_tensor(obs)
        if self._h is None:
            h = torch.zeros(self.rssm.num_gru_layers, 1, self.rssm.deterministic_size, device=self.device)
        else:
            h, _, _, _ = self.rssm.rssm_step(self._h, self._z.unsqueeze(1), self._prev_action.unsqueeze(1))
        _, _, z = self.rssm.encode_posterior(obs_t, h)
        feat = torch.cat([h[-1], z], dim=-1)

        _, action, _, _ = self.actor.sample(feat, deterministic=deterministic)
        self._h, self._z, self._prev_action = h, z, action.detach()

        env_action = self._denormalize_action(action).detach().cpu().numpy().reshape(self.action_space.shape)
        return env_action.astype(np.float32)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = self._to_image_sequence(batch['obs'])  # (B, T, C, H, W)
        action = self._normalize_action(batch['action'].to(self.device)).permute(1, 0, 2)  # (B, T, A)
        reward = batch['reward'].to(self.device).permute(1, 0)  # (B, T)
        B, T = obs.shape[0], obs.shape[1]

        # Encode the real sequence with the RSSM, collecting every visited
        # (h, z) model state along the way as a seed for imagination below.
        h = torch.zeros(self.rssm.num_gru_layers, B, self.rssm.deterministic_size, device=self.device)
        _, _, z = self.rssm.encode_posterior(obs[:, 0], h)
        x_recon = self.rssm.decoder(torch.cat([h[-1], z], dim=-1))

        states = [(h, z)]
        mu_priors, log_var_priors, mu_posts, log_var_posts = [], [], [], []
        x_preds, reward_preds = [], []
        for t in range(T - 1):
            h, z_prior, mu_p, log_var_p = self.rssm.rssm_step(h, z.unsqueeze(1), action[:, t].unsqueeze(1))
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)
            x_preds.append(self.rssm.decoder(torch.cat([h[-1], z_prior], dim=-1)))
            reward_preds.append(self.rssm.reward_decoder(torch.cat([h[-1], z_prior], dim=-1)))
            mu_q, log_var_q, z = self.rssm.encode_posterior(obs[:, t + 1], h)
            mu_posts.append(mu_q)
            log_var_posts.append(log_var_q)
            states.append((h, z))

        tr = {
            'x': obs[:, 0],
            'x_next': obs[:, 1:],
            'x_recon': x_recon,
            'x_pred': torch.stack(x_preds, dim=1),
            'reward_next': reward[:, :T - 1].unsqueeze(-1),
            'reward_pred': torch.stack(reward_preds, dim=1),
            'mu_posts': torch.stack(mu_posts, dim=1),
            'log_var_posts': torch.stack(log_var_posts, dim=1),
            'mu_priors': torch.stack(mu_priors, dim=1),
            'log_var_priors': torch.stack(log_var_priors, dim=1),
        }
        world_loss, world_metrics = self.rssm_loss(tr, self._update_count)
        self._update_count += 1

        self.rssm_optimizer.zero_grad(set_to_none=True)
        world_loss.backward()
        self.rssm_optimizer.step()

        actor_loss, critic_loss, imagined_value_mean = self._imagine_and_update_actor_critic(states, T, B)

        metrics = {f'world_model/{k}': v for k, v in world_metrics.items()}
        metrics.update({
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'imagined_value_mean': imagined_value_mean,
        })
        return metrics

    def _imagine_and_update_actor_critic(self, states, T: int, B: int):
        """Roll the actor forward purely in imagination and update actor/critic
        from the resulting lambda-returns (Dreamer eqs. 4-11)."""
        start_h = torch.stack([s[0] for s in states], dim=1).reshape(
            self.rssm.num_gru_layers, T * B, self.rssm.deterministic_size
        ).detach()
        start_z = torch.stack([s[1] for s in states], dim=0).reshape(
            T * B, self.rssm.stochastic_size
        ).detach()

        h, z = start_h, start_z
        feat = torch.cat([h[-1], z], dim=-1)
        feats = [feat]
        for _ in range(self.imagination_horizon):
            _, imag_action, _, _ = self.actor.sample(feat)
            h, z, _, _ = self.rssm.rssm_step(h, z.unsqueeze(1), imag_action.unsqueeze(1))
            feat = torch.cat([h[-1], z], dim=-1)
            feats.append(feat)
        imag_feat = torch.stack(feats, dim=1)  # (N, H+1, feat_dim)

        rewards = self.rssm.reward_decoder(imag_feat)[:, 1:, 0]
        values = self.value_net(imag_feat)[:, 1:, 0]
        bootstrap = values[:, -1]
        lambda_ret = self._lambda_return(rewards, values, bootstrap, self.gamma, self.lam)

        weights = (self.gamma ** torch.arange(
            self.imagination_horizon, dtype=torch.float32, device=self.device
        )).unsqueeze(0)
        actor_loss = -(weights * lambda_ret).sum(dim=1).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Fresh detached forward pass so the critic regression graph is
        # independent of the actor's (already consumed) imagination graph.
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_pred = self.value_net(imag_feat.detach())[:, 1:, 0]
        critic_loss = (weights * (value_pred - lambda_ret.detach()) ** 2).sum(dim=1).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu()), float(values.mean().detach().cpu())

    def save(self) -> Dict:
        return {
            'rssm': self.rssm.state_dict(),
            'actor': self.actor.state_dict(),
            'value_net': self.value_net.state_dict(),
            'update_count': self._update_count,
        }

    def load(self, state: Dict) -> None:
        self.rssm.load_state_dict(state['rssm'])
        self.actor.load_state_dict(state['actor'])
        self.value_net.load_state_dict(state['value_net'])
        self._update_count = state.get('update_count', 0)

    def reset(self) -> None:
        self._h = None
        self._z = None
        self._prev_action = None

    def predict_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reconstruct obs through the world model, as an (H, W, C) uint8 frame."""
        with torch.no_grad():
            obs_t = self._to_image_tensor(obs)
            if self._h is None:
                h = torch.zeros(self.rssm.num_gru_layers, 1, self.rssm.deterministic_size, device=self.device)
            else:
                h, _, _, _ = self.rssm.rssm_step(self._h, self._z.unsqueeze(1), self._prev_action.unsqueeze(1))
            _, _, z = self.rssm.encode_posterior(obs_t, h)
            recon = self.rssm.decoder(torch.cat([h[-1], z], dim=-1))
        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (np.clip(recon, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def _lambda_return(reward: torch.Tensor, value: torch.Tensor, bootstrap: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
        """TD(lambda) returns V_1^lambda .. V_H^lambda (Dreamer eq. 6)."""
        horizon = reward.shape[1]
        next_values = torch.cat([value[:, 1:], bootstrap.unsqueeze(1)], dim=1)
        inputs = reward + gamma * (1.0 - lam) * next_values
        outputs = []
        last = bootstrap
        for t in reversed(range(horizon)):
            last = inputs[:, t] + gamma * lam * last
            outputs.append(last)
        outputs.reverse()
        return torch.stack(outputs, dim=1)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.act_low) / (self.act_high - self.act_low) * 2 - 1

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2

    def _to_tensor(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _to_image_tensor(self, obs) -> torch.Tensor:
        """Convert a single channel-last (H, W, C) obs to batched (1, C, H, W)."""
        obs = self._to_tensor(obs)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return obs.permute(0, 3, 1, 2).contiguous()

    def _to_image_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert a (T, B, H, W, C) sequence batch to (B, T, C, H, W)."""
        return obs.to(self.device).permute(1, 0, 4, 2, 3).contiguous()
