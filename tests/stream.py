import torch
from classifier.audio.audio import AudioModule
from classifier.model.cnnmodel import CNNModel
from classifier.utility.wav2spec import waveform2spec


model = CNNModel().load_from_checkpoint(
    "lightning_logs\checkpoints\epoch=2-step=702.ckpt")

microphone = AudioModule(rate=16000, chunk=64000)
for i in range(10):
    aud = microphone.get_audio()
    torchaud = torch.tensor(next(aud)).reshape(1, 64000)
    spec = waveform2spec(torchaud, n_fft=2048, hop_len=512, n_mels=64, sample_rate=16000, chunk_size=64000)
    output = model(spec)
    _, pred = torch.max(output, dim=1)
    print(pred)
