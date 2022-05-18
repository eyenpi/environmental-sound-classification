import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.model = resnet18(progress=False, pretrained=False)
        self.model.fc = nn.Linear(512, 10)
        self.model.conv1 = nn.Conv2d(1,
                                     64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3),
                                     bias=False)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
