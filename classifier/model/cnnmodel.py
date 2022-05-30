import pytorch_lightning as pl
import torch
import torch.nn as nn

class CNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
            nn.Linear(9 * 64, 128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)

    def step(self, batch, mode='train'):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)

        _, pred = torch.max(output, dim=1)
        acc = torch.sum(pred == y.data) / (y.shape[0] * 1.0)

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_accuracy", acc)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
