import torch
import torch.nn.functional as F
from classifier.audio.microphone import AudioModule
from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import wav2spec
import librosa
import librosa.display
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.ion()
plt.show()

model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\epoch=9-step=2340.ckpt")

model.eval()

sample_rate = 16000
chunk_size = 16000
buffer_size = 32000
frame = buffer_size // chunk_size

buffer = torch.rand([1, buffer_size])

microphone = AudioModule(rate=sample_rate, chunk=chunk_size)
aud = microphone.get_audio()

for i in range(3000):
    buffer = torch.roll(buffer, (frame-1)*chunk_size, dims=1)
    buffer[:, -chunk_size:] = torch.tensor(next(aud)).reshape(1, -1)
    spec = wav2spec(buffer, n_fft=2048, hop_len=512,
                    n_mels=64, sample_rate=16000, chunk_size=buffer_size, online=True)
    output = F.softmax(model(spec), dim=1)
    prob, pred = torch.max(output, dim=1)
    if prob > 0.60:
        print("prediciton:", pred.item(), "probability:", prob.item())
    
    librosa.display.specshow(spec.detach().reshape(64, -1).numpy(), sr=16000, hop_length=512, ax=ax)
    plt.pause(0.005)