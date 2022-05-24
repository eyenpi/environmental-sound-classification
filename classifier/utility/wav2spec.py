import torchaudio
import torch


def wav2spec(filename, n_fft, hop_len, n_mels, sample_rate, chunk_size, online=False):
    if not online:
        wave, sr = torchaudio.load(filename)
    else:
        wave = filename
        sr = sample_rate
    audio_mono = torch.mean(wave, dim=0, keepdim=True)
    audio_mono = torchaudio.transforms.Resample(sr, sample_rate)(audio_mono)
    tempData = torch.zeros([1, chunk_size])
    if audio_mono.numel() < chunk_size:
        tempData[:, :audio_mono.numel()] = audio_mono
    else:
        tempData = audio_mono[:, :chunk_size]
    audio_mono = tempData
    melsgram = torchaudio.transforms.MelSpectrogram(
        sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(audio_mono)
    melsgram = torchaudio.transforms.AmplitudeToDB()(melsgram)
    melsgram = melsgram.reshape(1, 1, n_mels, -1)
    return melsgram
