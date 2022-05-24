import torch
import torch.nn.functional as F
import torchaudio
from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import wav2spec

model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\epoch=9-step=2340.ckpt")

model.eval()

sample_rate = 16000
chunk_size = 8000
buffer_size = 32000
frame = buffer_size // chunk_size

buffer = torch.rand([1, buffer_size])

wave, sr = torchaudio.load('lightning_logs\long.wav')
audio_mono = torch.mean(wave, dim=0, keepdim=True)
audio_mono = torchaudio.transforms.Resample(sr, 16000)(audio_mono)

for i in range(24):
    buffer = torch.roll(buffer, (frame-1)*chunk_size, dims=1)
    aud = audio_mono[:, i*chunk_size:(i+1)*chunk_size]
    buffer[:, -chunk_size:] = aud
    spec = wav2spec(buffer, n_fft=2048, hop_len=512,
                    n_mels=64, sample_rate=16000, chunk_size=buffer_size, online=True)

    output = F.softmax(model(spec), dim=1)
    prob, pred = torch.max(output, dim=1)
    # if prob > 0.9:
    print("prediciton:", pred.item(), "probability:", prob.item())
