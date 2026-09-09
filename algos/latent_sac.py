"""Soft Actor-Critic (SAC) implementation for the RL suite.

This is a compact SAC implementation that supports:
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
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from algos.base import Algorithm
from algos.sac import SAC
from models.mlp import MLP
from models.encoder import ConvEncoder
from models.decoder import ConvDecoder

class LatentSAC(Algorithm):
    """TODO"""

    requires_sequences = False

    def __init__(
        self,
        action_space,
        observation_space,
        sac_params: Dict,
        latent_state_size: int,
        latent_action_size: int,
        in_channels: int, 
        conv_params: Dict,
        beta: float = 1e-4,
        obs_recon_mult: float = 1.0,
        device: Optional[str] = None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        # Reassign act/obs space as same type but with latent sizes
        sac_params["observation_space"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(latent_state_size,), dtype=np.float32  # TODO: Should clip?
        )
        sac_params["action_space"] = spaces.Box(
            low=-1.0, high=1.0, shape=(latent_action_size,), dtype=np.float32
        )

        sac_params["device"] = device
        self.sac = SAC(**sac_params)

        self.device = device
        self.latent_state_size = latent_state_size
        self.latent_action_size = latent_action_size
        self.beta = beta
        self.obs_recon_mult = obs_recon_mult
        self.learning_rate = sac_params["learning_rate"]

        # VAE
        self.obs_encoder = ConvEncoder(self.latent_state_size, in_channels, conv_params).to(self.device)
        self.act_encoder = nn.Linear(int(np.prod(self.action_space.shape)), self.latent_action_size).to(self.device)   # TODO: Need more layers?

        self.mu_obs = nn.Linear(self.latent_state_size, self.latent_state_size).to(self.device)
        self.log_var_obs = nn.Linear(self.latent_state_size, self.latent_state_size).to(self.device)

        self.mu_act = nn.Linear(self.latent_action_size, self.latent_action_size).to(self.device)
        self.log_var_act = nn.Linear(self.latent_action_size, self.latent_action_size).to(self.device)

        self.obs_decoder = ConvDecoder(
            self.latent_state_size, 
            conv_params, 
            self.obs_encoder.out_dim_flat, 
            self.obs_encoder.out_shape
        ).to(self.device)
        self.out_image_shape = self.obs_decoder.out_image_shape

        self.act_decoder = nn.Linear(self.latent_action_size, int(np.prod(self.action_space.shape))).to(self.device)

        # Optimizers collect all VAE params
        # TODO: Different learning rates for obs vs act? If not, can we use same optimizer?
        self.obs_optimizer = torch.optim.Adam(
            list(self.obs_encoder.parameters())
            + list(self.mu_obs.parameters())
            + list(self.log_var_obs.parameters())
            + list(self.obs_decoder.parameters()),
            lr=self.learning_rate,
        )

        self.act_optimizer = torch.optim.Adam(
            list(self.act_encoder.parameters())
            + list(self.mu_act.parameters())
            + list(self.log_var_act.parameters())
            + list(self.act_decoder.parameters()),
            lr=self.learning_rate,
        )

    def encode_obs(self, obs: np.ndarray):
        obs = self._to_image_tensor(obs)
        encoded = self.obs_encoder(obs)
        flattened = encoded.view(encoded.size(0), -1)

        mu_obs = self.mu_obs(flattened)
        log_var_obs = self.log_var_obs(flattened)
        z = self._reparameterize(mu_obs, log_var_obs)
        return z, mu_obs, log_var_obs

    def encode_act(self, act: np.ndarray):
        act = self._to_tensor(act)      # TODO: _to_tensor going to cause problems?
        encoded = self.act_encoder(act)

        mu_act = self.mu_act(encoded)
        log_var_act = self.log_var_act(encoded)
        u = self._reparameterize(mu_act, log_var_act)
        return u, mu_act, log_var_act

    def act(self, obs: np.ndarray, deterministic: bool = False):
        z, _, _ = self.encode_obs(obs)
        latent_action = self.sac.act(z.detach().cpu().numpy(), deterministic=deterministic)
        action = self.act_decoder(self._to_tensor(latent_action))
        action = action.detach().cpu().numpy().reshape(self.action_space.shape)
        return action.astype(np.float32)

    def predict_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reconstruct obs through the VAE, returned as an (H, W, C) uint8 frame."""
        with torch.no_grad():
            z, _, _ = self.encode_obs(obs)
            recon = self.obs_decoder(z)
        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (np.clip(recon, 0.0, 1.0) * 255.0).astype(np.uint8)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        # TODO
        # Batch encode observations (current and next) and actions to latent space
        z, mu_obs, log_var_obs = self.encode_obs(batch['obs'])
        with torch.no_grad():
            next_z, _, _ = self.encode_obs(batch['next_obs'])  # TODO: Should consider doing with grad for double perception recon update?
        u, mu_act, log_var_act = self.encode_act(batch['action'])

        # Perception gradient updates
        obs_recon = self.obs_decoder(z)
        obs_target = self._to_image_tensor(batch['obs'])
        act_recon = self.act_decoder(u)
        act_target = self._to_tensor(batch['action'])

        obs_loss, obs_recon_loss, obs_kl = self._vae_loss(obs_recon, obs_target, mu_obs, log_var_obs)
        act_loss, act_recon_loss, act_kl = self._vae_loss(act_recon, act_target, mu_act, log_var_act)

        self.obs_optimizer.zero_grad(set_to_none=True)
        obs_loss.backward()
        self.obs_optimizer.step()

        self.act_optimizer.zero_grad(set_to_none=True)
        act_loss.backward()
        self.act_optimizer.step()

        # SAC gradient updates
        latent_batch = {
            'obs': z.detach().cpu().numpy(),    # TODO: Should detach here or should RL grads affect perception?
            'action': u.detach().cpu().numpy(),
            'reward': batch['reward'],
            'next_obs': next_z.detach().cpu().numpy(),
            'done': batch['done'],
        }

        metrics = self.sac.update(latent_batch)
        metrics.update({
            'obs_vae_loss': obs_loss.item(),
            'obs_recon_loss': obs_recon_loss.item(),
            'obs_kl_loss': obs_kl.item(),
            'act_vae_loss': act_loss.item(),
            'act_recon_loss': act_recon_loss.item(),
            'act_kl_loss': act_kl.item(),
        })
        return metrics

    def save(self) -> Dict:
        return {
            'sac': self.sac.save(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'act_encoder': self.act_encoder.state_dict(),
            'mu_obs': self.mu_obs.state_dict(),
            'log_var_obs': self.log_var_obs.state_dict(),
            'mu_act': self.mu_act.state_dict(),
            'log_var_act': self.log_var_act.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'act_decoder': self.act_decoder.state_dict(),
        }

    def load(self, state: Dict) -> None:
        self.sac.load(state['sac'])
        self.obs_encoder.load_state_dict(state['obs_encoder'])
        self.act_encoder.load_state_dict(state['act_encoder'])
        self.mu_obs.load_state_dict(state['mu_obs'])
        self.log_var_obs.load_state_dict(state['log_var_obs'])
        self.mu_act.load_state_dict(state['mu_act'])
        self.log_var_act.load_state_dict(state['log_var_act'])
        self.obs_decoder.load_state_dict(state['obs_decoder'])
        self.act_decoder.load_state_dict(state['act_decoder'])

    def reset(self) -> None:
        pass

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _to_image_tensor(self, obs) -> torch.Tensor:
        """Convert channel-last (H, W, C) pixel obs to batched channel-first (N, C, H, W)."""
        obs = self._to_tensor(obs)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return obs.permute(0, 3, 1, 2).contiguous()

    def _vae_loss(self, reconstruction, target, mu, log_var):
        # TODO: Need reconstruction mult? Different mults for obs vs action (recon and kld)?
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / target.size(0)
        return self.obs_recon_mult * reconstruction_loss + self.beta * kl_loss, reconstruction_loss, kl_loss

    def _to_tensor(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
