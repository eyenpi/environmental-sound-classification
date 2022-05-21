from classifier.model.cnnmodel import CNNModel
from classifier.dataset.dataset import AudioDataModule

import pytorch_lightning as pl

model = CNNModel()
datamod = AudioDataModule("./urbansound8k/", n_fft=2048, hop_len=512, n_mels=64, sample_rate=16000, chunk_size=64000)
trainer = pl.Trainer(gpus=1, max_epochs=3)
trainer.fit(model, datamodule=datamod)