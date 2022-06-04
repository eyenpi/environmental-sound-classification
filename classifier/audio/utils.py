import torchaudio
import torch

class AudioUtils:
    def open(self, path):
        return torchaudio.load(path)

    def to_mono(self, audio):
        return torch.mean(audio, dim=0, keepdim=True)

    def resample(self, audio, sr, new_sr):
        return torchaudio.transforms.Resample(sr, new_sr)(audio)

    def pad(self, audio, length):
        tempData = torch.zeros([1, length])
        if audio.numel() < length:
            tempData[:, :audio.numel()] = audio
        else:
            tempData = audio[:, :length]
        return tempData

    def melsgram(self, audio, sr, n_fft, hop_length, n_mels):
        melsgram = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(audio)
        melsgram = torchaudio.transforms.AmplitudeToDB()(melsgram)
        return melsgram

    def augment(self, specgram, n_time_mask, time_mask_params, n_freq_mask, freq_mask_params):
        aug_specgram = specgram
        for _ in range(n_time_mask):
            aug_specgram = torchaudio.transforms.TimeMasking(
                time_mask_param=time_mask_params, iid_masks=False)(aug_specgram)
        for _ in range(n_freq_mask):
            aug_specgram = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=freq_mask_params, iid_masks=False)(aug_specgram)
        return aug_specgram

