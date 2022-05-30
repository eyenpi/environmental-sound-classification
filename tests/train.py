from classifier.model.resnet18 import CNNModel
from classifier.dataset.dataset import AudioDataModule

import pytorch_lightning as pl

model = CNNModel()
datamod = AudioDataModule("./urbansound8k", n_fft=1024, hop_len=256, n_mels=128, sample_rate=16000, chunk_size=32000)
trainer = pl.Trainer(gpus=1, max_epochs=10)
trainer.fit(model, datamodule=datamod)