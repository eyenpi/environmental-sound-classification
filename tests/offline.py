import torch
import torch.nn.functional as F
# from classifier.model.cnnmodel import CNNModel
from classifier.model.resnet18 import CNNModel
from classifier.utility.wav2spec import wav2spec

model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\\resnetlast.ckpt")

model.eval()

spec = wav2spec("lightning_logs\long\\2f.wav",
                n_fft=1024, hop_len=256, n_mels=128, sample_rate=16000, chunk_size=32000)
output = F.softmax(model(spec), dim=1)
alpha = 0.2
smoothed = ((1-alpha) * output) + alpha / 10
prob, pred = torch.max(output, dim=1)
print(pred, prob)
print(smoothed)
