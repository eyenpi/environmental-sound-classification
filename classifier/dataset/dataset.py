import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    def __init__(self, df, n_fft, hop_len, n_mels, sample_rate, chunk_size):
        self.paths = df['path']
        self.targets = df['classID']

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        target = self.targets[item]
        wave, sr = torchaudio.load(self.paths.iloc[item])
        audio_mono = torch.mean(wave, dim=0, keepdim=True)
        audio_mono = torchaudio.transforms.Resample(
            sr, self.sample_rate)(audio_mono)
        tempData = torch.zeros([1, self.chunk_size])
        if audio_mono.numel() < self.chunk_size:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :self.chunk_size]
        audio_mono = tempData
        melsgram = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=self.n_fft, hop_length=self.hop_len, n_mels=self.n_mels)(audio_mono)
        melsgram = torchaudio.transforms.AmplitudeToDB()(melsgram)
        return melsgram, target


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
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=2)
