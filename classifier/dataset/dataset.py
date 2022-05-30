import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split


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


class AudioDataset(Dataset):
    def __init__(self, df, n_fft, hop_len, n_mels, sample_rate, chunk_size, augment=True):
        self.paths = df['path']
        self.targets = df['classID']

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        target = self.targets[item]
        audio_helper = AudioUtils()
        wave, sr = audio_helper.open(self.paths.iloc[item])
        wave = audio_helper.to_mono(wave)
        wave = audio_helper.resample(wave, sr, self.sample_rate)
        wave = audio_helper.pad(wave, self.chunk_size)
        wave = audio_helper.melsgram(
            wave, self.sample_rate, self.n_fft, self.hop_len, self.n_mels)
        if self.augment:
            wave = audio_helper.augment(
                wave, n_time_mask=5, time_mask_params=7, n_freq_mask=5, freq_mask_params=7)
        return wave, target


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, base_path, n_fft, hop_len, n_mels, sample_rate, chunk_size, batch_size=32):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.df = pd.read_csv(f"{self.base_path}/UrbanSound8K.csv")
        self.df['path'] = self.df.apply(
            lambda x: f"{base_path}/fold{x.fold}/{x.slice_file_name}", axis=1)

        df_train, df_test = train_test_split(self.df, test_size=0.05)
        df_train, df_val = train_test_split(df_train, test_size=0.1)

        self.df_train = df_train.reset_index(drop=True)
        self.df_val = df_val.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)

        weights = 1 / self.df_train.groupby(["classID"])['class'].count()
        self.df_train['prob'] = self.df_train.apply(
            lambda x: weights[x['classID']], axis=1)

    def setup(self, stage=None):
        self.trainset = AudioDataset(
            self.df_train, n_fft=self.n_fft, hop_len=self.hop_len,
            n_mels=self.n_mels, sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.valset = AudioDataset(
            self.df_val, n_fft=self.n_fft, hop_len=self.hop_len,
            n_mels=self.n_mels, sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.testset = AudioDataset(
            self.df_test, n_fft=self.n_fft, hop_len=self.hop_len,
            n_mels=self.n_mels, sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        print("len train dataset:", len(self.trainset))
        print("len valid dataset:", len(self.valset))
        print("len test dataset:", len(self.testset))

    def train_dataloader(self):
        probs = torch.tensor(self.df_train['prob']).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            probs, len(probs))
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=2,
                          sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=2)
