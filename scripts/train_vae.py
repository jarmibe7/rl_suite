"""
Train the convolutional VAE on MNIST.

python scripts/train_vae.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vae import ConvVAE

DATA_DIR = Path("data")
RUN_DIR = Path("runs/vae")
OUTPUTS = RUN_DIR / "mnist.pt"
RENDER_DIR = RUN_DIR / "renders"
NUM_EPOCHS = 10
BATCH_SIZE = 128
LATENT_SIZE = 16
LEARNING_RATE = 1e-3
BETA = 1.0
SEED = 0
DEVICE = "cuda:1"
RENDER_EVERY = 1
NUM_RENDER = 8


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def render_reconstructions(model, dataset, epoch, device):
	"""Save a grid of ground truth images alongside their reconstructions (ground truth top)."""
	model.eval()
	indices = random.sample(range(len(dataset)), NUM_RENDER)
	images = torch.stack([dataset[i][0] for i in indices])
	with torch.no_grad():
		images = images.to(device)
		_, reconstruction, _, _ = model(images)
	comparison = torch.cat([images.cpu(), reconstruction.cpu()], dim=0)
	RENDER_DIR.mkdir(parents=True, exist_ok=True)
	vutils.save_image(
		comparison, RENDER_DIR / f"epoch_{epoch:03d}.png", nrow=NUM_RENDER
	)


def vae_loss(reconstruction, target, mu, log_var, beta):
	reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')
	kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
	kl_loss = 1e-4 * (kl_loss / target.size(0))
	return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def main():
	set_seed(SEED)

	if "cuda" in DEVICE and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	else:
		device = torch.device(DEVICE)

	# Encoder/decoder only works for image size divisible by 8
	transform = transforms.Compose([
		transforms.Pad(2),
		transforms.ToTensor(),
	])
	train_set = datasets.MNIST(
		root=DATA_DIR, train=True, download=True, transform=transform
	)
	train_loader = DataLoader(
		train_set,
		batch_size=BATCH_SIZE,
		shuffle=True,
		pin_memory=device.type == "cuda",
	)

	conv_params = {
		"in_image_shape": (1, 32, 32),
		"out_image_shape": (1, 32, 32),
		"enc_kernel_size": 3,
		"dec_kernel_size": 4,
		"stride": 2,
		"pad": 1,
	}
	model = ConvVAE(LATENT_SIZE, 1, conv_params, device).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="training")
	for epoch in epoch_bar:
		model.train()
		total_loss = 0.0
		total_reconstruction = 0.0
		total_kl = 0.0

		for images, _ in train_loader:
			images = images.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)
			_, reconstruction, mu, log_var = model(images)
			loss, reconstruction_loss, kl_loss = vae_loss(
				reconstruction, images, mu, log_var, BETA
			)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			total_reconstruction += reconstruction_loss.item()
			total_kl += kl_loss.item()

		batches = len(train_loader)
		epoch_bar.set_postfix(
			loss=f"{total_loss / batches:.4f}",
			reconstruction=f"{total_reconstruction / batches:.4f}",
			kl=f"{total_kl / batches:.4f}",
		)

		if epoch % RENDER_EVERY == 0:
			render_reconstructions(model, train_set, epoch, device)

	# Render new sample
	model.eval()
	with torch.no_grad():
		sample = torch.randn(NUM_RENDER, LATENT_SIZE, device=device)
		sample = model.conv_decoder(sample)
		RENDER_DIR.mkdir(parents=True, exist_ok=True)
		vutils.save_image(
			sample.cpu(), RENDER_DIR / f"sample.png", nrow=NUM_RENDER
		)
	
	
	OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"latent_size": LATENT_SIZE,
			"conv_params": conv_params,
			"args": {
				"num_epochs": NUM_EPOCHS,
				"batch_size": BATCH_SIZE,
				"latent_size": LATENT_SIZE,
				"learning_rate": LEARNING_RATE,
				"beta": BETA,
				"seed": SEED,
				"device": DEVICE,
			},
		},
		OUTPUTS,
	)
	print(f"Saved checkpoint to {OUTPUTS}")


if __name__ == "__main__":
	main()
