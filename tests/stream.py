import torch
from classifier.audio.audio import AudioModule
from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import waveform2spec


model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\epoch=2-step=702.ckpt")

microphone = AudioModule(rate=44100, chunk=160000)
for i in range(10):
    aud = microphone.get_audio()
    torchaud = torch.tensor(next(aud)).reshape(1, 160000)
    spec = waveform2spec(torchaud, n_fft=2048, hop_len=512, n_mels=64)
    output = model(spec)
    _, pred = torch.max(output, dim=1)
    print(pred)
