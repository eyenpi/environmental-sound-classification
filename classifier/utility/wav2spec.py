import torchaudio
import torch

def wav2spec(filename, n_fft, hop_len, n_mels):
    wave, sr = torchaudio.load(filename)
    audio_mono = torch.mean(wave, dim=0, keepdim=True)
    audio_mono = torchaudio.transforms.Resample(sr, 44100)(audio_mono)
    tempData = torch.zeros([1, 160000])
    if audio_mono.numel() < 160000:
        tempData[:, :audio_mono.numel()] = audio_mono
    else:
        tempData = audio_mono[:, :160000]
    audio_mono = tempData
    melsgram = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(audio_mono)
    melsgram = torchaudio.transforms.AmplitudeToDB()(melsgram)
    melsgram = melsgram.reshape(1, 1, n_mels, -1)
    return melsgram

def waveform2spec(waveform, n_fft, hop_len, n_mels):
    sr = 44100
    audio_mono = torchaudio.transforms.Resample(sr, 44100)(waveform)
    tempData = torch.zeros([1, 160000])
    if audio_mono.numel() < 160000:
        tempData[:, :audio_mono.numel()] = audio_mono
    else:
        tempData = audio_mono[:, :160000]
    audio_mono = tempData
    melsgram = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(audio_mono)
    melsgram = torchaudio.transforms.AmplitudeToDB()(melsgram)
    melsgram = melsgram.reshape(1, 1, n_mels, -1)
    return melsgram