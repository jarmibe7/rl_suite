"""Train the convolutional VAE on MNIST."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vae import ConvVAE

DATA_DIR = Path("data")
OUTPUTS = Path("runs/vae/mnist.pt")
NUM_EPOCHS=10
BATCH_SIZE = 128
LATENT_SIZE = 16
LEARNING_RATE = 1e-3
BETA = 1.0
SEED = 0
DEVICE = "cuda:0"


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def vae_loss(reconstruction, target, mu, log_var, beta):
	reconstruction_loss = F.binary_cross_entropy(
		reconstruction, target, reduction="sum"
	) / target.size(0)
	kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
	kl_loss = kl_loss / target.size(0)
	return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def main():
	set_seed(SEED)

	if "cuda" in DEVICE and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	if DEVICE == "auto":
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

	for epoch in range(1, NUM_EPOCHS + 1):
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
		print(
			f"epoch {epoch:03d}/{NUM_EPOCHS:03d} "
			f"loss={total_loss / batches:.4f} "
			f"reconstruction={total_reconstruction / batches:.4f} "
			f"kl={total_kl / batches:.4f}"
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
