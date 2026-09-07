"""A simple convolutional Variational Autoencoder (VAE)."""
import torch.nn as nn
import torch

from models.encoder import ConvEncoder
from models.decoder import ConvDecoder


class ConvVAE(nn.Module):
  def __init__(self, latent_size, in_channels, conv_params, device):
    super().__init__()
    self.device = device

    self.latent_size = latent_size

    self.encoder = ConvEncoder(self.latent_size, in_channels, conv_params)

    # ConvEncoder already projects to latent_size internally, so mu/log_var take that as input.
    self.mu = nn.Linear(self.latent_size, self.latent_size)
    self.log_var = nn.Linear(self.latent_size, self.latent_size)

    self.decoder = ConvDecoder(
        self.latent_size, 
        conv_params, 
        self.encoder.out_dim_flat, 
        self.encoder.out_shape
    )
    self.out_image_shape = self.decoder.out_image_shape

  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    # Encode input
    encoded = self.encoder(x)
    flattened = encoded.view(encoded.size(0), -1)

    # Get latent variables
    mu = self.mu(flattened)
    log_var = self.log_var(flattened)
    z = self.reparameterize(mu, log_var)

    decoded = self.decoder(z)
    return encoded, decoded, mu, log_var

  def sample(self, num_samples):
    """Generate random samples"""
    with torch.no_grad():
      z = torch.randn(num_samples, self.latent_size).to(self.device)
      decoded = self.decoder(z)
      samples = decoded.view(-1, self.out_image_shape[0], self.out_image_shape[1], self.out_image_shape[2])

    return samples.cpu()