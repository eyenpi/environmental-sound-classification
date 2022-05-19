import torch

from model.cnnmodel import CNNModel
from utility.wav2spec import wav2spec

model = CNNModel().load_from_checkpoint("lightning_logs\checkpoints\epoch=2-step=702.ckpt")

spec = wav2spec("lightning_logs\\102871-8-0-10.wav", n_fft=2048, hop_len=512, n_mels=64)
output = model(spec)
_, pred = torch.max(output, dim=1)
print(pred)