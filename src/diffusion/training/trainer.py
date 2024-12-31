import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.diffusion.datasets.simulated_dataset import SimulatedDataset
from src.diffusion.modules.diffusion_model import DiffusionModel


class DiffusionTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model.compute_loss(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_diffusion():
    dataset = SimulatedDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = DiffusionModel()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(DiffusionTrainer(model), dataloader)
