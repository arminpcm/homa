import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl


class DiffusionModel(pl.LightningModule):
    def __init__(self, image_size=64, channels=1, timesteps=1000, lr=1e-3):
        """
        PyTorch Lightning-based Diffusion Model.

        Args:
            image_size (int): Size of the input image.
            channels (int): Number of channels in the input image.
            timesteps (int): Number of diffusion timesteps.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.lr = lr

        # Define the U-Net for noise prediction
        self.model = UNet(channels=channels, timesteps=timesteps)

        # Precompute the beta schedule and related terms
        self.betas = torch.linspace(1e-4, 0.02, timesteps)  # Linear beta schedule
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        """
        Forward process for predicting noise.

        Args:
            x (torch.Tensor): Noisy input image.
            t (torch.Tensor): Timestep (batch of indices).

        Returns:
            torch.Tensor: Predicted noise.
        """
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input image and noise.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, noise = batch
        batch_size = x.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device).long()

        # Compute noisy images
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        noisy_images = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        # Predict the noise
        predicted_noise = self(noisy_images, t)

        # Mean squared error loss
        loss = F.mse_loss(predicted_noise, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for PyTorch Lightning.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample(self, shape, device):
        """
        Generate samples from the model.

        Args:
            shape (tuple): Shape of the output image.
            device (torch.device): Device to run the sampling.

        Returns:
            torch.Tensor: Generated image.
        """
        x = torch.randn(shape, device=device)  # Start from pure noise

        for t in reversed(range(self.timesteps)):
            alpha_t = self.alpha_cumprod[t]
            beta_t = self.betas[t]
            x = (
                x
                - torch.sqrt(1 - alpha_t)
                * self(x, torch.tensor([t], device=device).long())
            ) / torch.sqrt(alpha_t)
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta_t) * noise
        return x


class UNet(nn.Module):
    def __init__(self, channels, timesteps):
        """
        A simple U-Net architecture for predicting noise.

        Args:
            channels (int): Number of input/output channels.
            timesteps (int): Number of diffusion timesteps.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, channels, 3, padding=1)
        self.time_embedding = nn.Embedding(timesteps, 64)

    def forward(self, x, t):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image.
            t (torch.Tensor): Timestep embeddings.

        Returns:
            torch.Tensor: Predicted noise.
        """
        time_emb = self.time_embedding(t).view(t.size(0), -1, 1, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
