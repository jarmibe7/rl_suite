"""
A convolutional decoder.

Authors: Jared Berry
"""
from torch import nn

class ConvDecoder(nn.Module):
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__()

        self.latent_size = latent_size

        # CNN parameters
        k = conv_params['dec_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.out_image_shape = conv_params['out_image_shape']
        self.enc_out_shape = enc_out_shape

        self.fc_decode = nn.Linear(self.latent_size, enc_out_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(self.latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, enc_out_dim),
            nn.ReLU(),
        )

        # Define convolutional decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.out_image_shape[0], kernel_size=conv_params['enc_kernel_size'], stride=s-1, padding=p),
            nn.Sigmoid(), # Keep between 0 and 1
        )

    def forward(self, z):
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        decoded = self.decoder_cnn(to_decode)
        return decoded