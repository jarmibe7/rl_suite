"""
rssm.py

RSSM model architecture made with PyTorch.

Author: Jared Berry
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.encoder import ConvEncoder
from models.decoder import ConvDecoder


class RSSM(nn.Module):
    """
    An RSSM with convolutional encoder-decoder and transition model.

    Args:
        enc_latent_size: Latent dimension of encoder
        stochastic_size: Stochastic state latent dimension
        deterministic_size: Deterministic state latent dimension
        control_size: Dimension of control vector
        pred_length: Prediction horizon length
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
        uncertainty_output: Whether to use an uncertainty decoder
        reward_size: Dimension of reward prediction output, if present
    """

    def __init__(self, enc_latent_size, stochastic_size, deterministic_size,
                 control_size, pred_length, conv_params, device,
                 img_channel_count=3, reward_size=None):
        super().__init__()
        self.device = device

        # Network sizes
        self.enc_latent_size = enc_latent_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.control_size = control_size
        self.reward_size = reward_size
        
        self.pred_length = pred_length

        # Encoder (posterior/representation model)
        embedded_size = self.enc_latent_size
        self.encoder = ConvEncoder(enc_latent_size, img_channel_count, conv_params)
        self.post = nn.Sequential(                      
            nn.Linear(embedded_size + self.deterministic_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.stochastic_size)
        )
        
        # Decoder (observation model)
        self.decoder = ConvDecoder(
            self.deterministic_size + self.stochastic_size, 
            conv_params, 
            self.encoder.out_dim_flat, 
            self.encoder.out_shape
        )
        self.out_image_shape = self.decoder.out_image_shape

        # Reward model
        if self.reward_size is not None:
            self.reward_decoder = nn.Sequential(
                nn.Linear(self.stochastic_size + self.deterministic_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.reward_size),
            )
        else:
            self.reward_decoder = None

        # Deterministic and stochastic state models
        self.num_gru_layers = 2
        self.rnn = nn.GRU(
            self.stochastic_size + self.control_size,
            self.deterministic_size,
            num_layers=self.num_gru_layers,
            batch_first=True
        )
        self.prior = nn.Sequential(
            nn.Linear(self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size)
        )
        
    def _reward_features(self, h, z):
        return torch.cat([h[-1], z], dim=-1)

    # def _match_image_shape(self, image, reference):
    #     if image.shape[-2:] != reference.shape[-2:]:
    #         image = F.interpolate(image, size=reference.shape[-2:], mode='bilinear', align_corners=False)
    #     return image

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_posterior(self, obs, h):
        enc = self.encoder(obs)
        stats = self.post(torch.cat([enc, h[-1]], dim=-1))
        mu, log_var = stats.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=1e-5, max=1e5)
        z = self._reparameterize(mu, log_var)

        return mu, log_var, z
    
    def rssm_step(self, h, z, u):
        # Get next deterministic and stochastic states
        rnn_input = torch.cat([z, u], dim=-1)
        _, h_next = self.rnn(rnn_input, h)

        stats = self.prior(h_next[-1])
        mu, log_var = stats.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=1e-5, max=1e5)
        z_next = self._reparameterize(mu, log_var)

        return h_next, z_next, mu, log_var
    
    def forward(self, x, x_next, u):
        h = torch.zeros(self.num_gru_layers, x.size(0), self.deterministic_size, device=self.device)

        # Reconstruct current observation
        _, _, z = self.encode_posterior(x, h)
        x_recon = self.decoder(torch.cat([h[-1], z], dim=-1))
        # x_recon = self._match_image_shape(x_recon, x)

        if self.reward_decoder is not None:
            reward_recon = self.reward_decoder(self._reward_features(h, z))

        # Iterate over pred_length
        mu_priors, log_var_priors = [], []
        mu_posts, log_var_posts = [], []
        x_preds = []
        if self.reward_decoder is not None:
            reward_preds = []

        for t in range(x_next.size(1)):
            # Prior
            h, z_prior, mu_p, log_var_p = self.rssm_step(h, z.unsqueeze(1), u[:, t].unsqueeze(1))
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)

            # Decode prior in open loop
            x_pred = self.decoder(torch.cat([h[-1], z_prior], dim=-1))
            # x_pred = self._match_image_shape(x_pred, x_next[:, t])
            x_preds.append(x_pred)
            
            if self.reward_decoder is not None:
                reward_pred = self.reward_decoder(self._reward_features(h, z_prior))
                reward_preds.append(reward_pred)
            
            # Posterior update using real next frame
            mu_q, log_var_q, z = self.encode_posterior(x_next[:, t], h)

            mu_posts.append(mu_q)
            log_var_posts.append(log_var_q)

        # Stack accumulated priors + posteriors
        outputs = {
            "x_recon": x_recon,
            "x_pred": torch.stack(x_preds, dim=1),
            "mu_posts": torch.stack(mu_posts, dim=1),
            "log_var_posts": torch.stack(log_var_posts, dim=1),
            "mu_priors": torch.stack(mu_priors, dim=1),
            "log_var_priors": torch.stack(log_var_priors, dim=1),
        }

        if self.reward_decoder is not None:
            outputs["reward_recon"] = reward_recon
            outputs["reward_pred"] = torch.stack(reward_preds, dim=1)

        return outputs

class RSSMLoss(nn.Module):
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = float(loss_params['recon_mult'])
        self.reward_mult = float(loss_params.get('reward_mult', 1.0))
        self.beta = float(loss_params['beta'])
        self.free_nats = float(loss_params.get('free_nats', 0.0))
        self.anneal_mode = loss_params['kld_anneal_mode']

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = self.beta * ((epoch + 1) / self.num_epochs)
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1
        ).sum(dim=-1)

    def forward(self, tr, epoch):
        # Reconstruction loss
        image_recon = self.recon_mult * nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')
        image_recon += self.recon_mult * nn.functional.mse_loss(tr['x'], tr['x_recon'], reduction='mean')

        if 'reward_next' in tr and tr['reward_next'] is not None and 'reward_pred' in tr:
            reward_recon = self.reward_mult * nn.functional.mse_loss(tr['reward_next'], tr['reward_pred'], reduction='mean')
            if 'reward' in tr and tr['reward'] is not None and 'reward_recon' in tr:
                reward_recon += self.reward_mult * nn.functional.mse_loss(tr['reward'], tr['reward_recon'], reduction='mean')
        else:
            reward_recon = torch.zeros((), device=tr['x'].device)

        # KLD loss
        kld = self.kl_divergence(
            tr["mu_posts"],
            tr["log_var_posts"],
            tr["mu_priors"],
            tr["log_var_priors"]
        )
        kld = torch.clamp(kld, min=self.free_nats)
        kld = kld.mean()
        kld = self.kld_anneal(epoch) * kld

        loss = image_recon + reward_recon + kld
        if torch.isnan(loss):
            raise ValueError("Loss is NaN!")

        loss_return = {
            "Image Reconstruction Loss": image_recon.detach().cpu().item(),
            "Reward Reconstruction Loss": reward_recon.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
        }
        return loss, loss_return