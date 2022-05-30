from time import sleep
import torch
import torch.nn.functional as F
import torchaudio
from classifier.model.resnet18 import CNNModel
# from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import wav2spec
import matplotlib.pyplot as plt
import librosa
import librosa.display
from playsound import playsound
from threading import Thread

def plays():
    playsound('lightning_logs\long\longlong.wav', block= False)

idtoclass = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling', 6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}

fig, ax = plt.subplots()
plt.ion()
plt.show()

model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\\resnetlast.ckpt")

model.eval()

sample_rate = 16000
chunk_size = 16000
buffer_size = 32000
frame = buffer_size // chunk_size

buffer = torch.zeros([1, buffer_size])

wave, sr = torchaudio.load('lightning_logs\long\longlong.wav')
audio_mono = torch.mean(wave, dim=0, keepdim=True)
audio_mono = torchaudio.transforms.Resample(sr, 16000)(audio_mono)

T = Thread(name='sound',target=plays)
T.start()
for i in range(1000):
    buffer = torch.roll(buffer, (frame-1)*chunk_size, dims=1)
    aud = audio_mono[:, i*chunk_size:(i+1)*chunk_size]
    if aud.shape[1] < chunk_size:
        break
    buffer[:, -chunk_size:] = aud
    spec = wav2spec(buffer, n_fft=1024, hop_len=256,
                    n_mels=128, sample_rate=16000, chunk_size=buffer_size, online=True)

    output = F.softmax(model(spec), dim=1)
    prob, pred = torch.max(output, dim=1)
    if prob > 0.9:
        print(f"prediciton: {pred.item()} probability: {prob.item():.2f}")
        plt.title(idtoclass[pred.item()])

    librosa.display.specshow(spec.detach().reshape(64, -1).numpy(), sr=16000, hop_length=256, ax=ax)
    plt.pause(0.005)
    if i < 32 or i > 38:
        sleep(0.8)