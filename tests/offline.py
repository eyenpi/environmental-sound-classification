import torch

from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import wav2spec

model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\epoch=2-step=702.ckpt")

spec = wav2spec("lightning_logs\\102871-8-0-10.wav",
                n_fft=2048, hop_len=512, n_mels=64, sample_rate=16000, chunk_size=64000)
output = model(spec)
_, pred = torch.max(output, dim=1)
print(pred)
