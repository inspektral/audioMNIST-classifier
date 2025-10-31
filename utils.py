import torch
import torchaudio

from mfccdataset import MFCCDataset
from torch.utils.data import DataLoader, random_split

def load_mfcc(audio_path:str, sr:int=16000, n_mfcc:int=20, max_len:float=1.0):
    waveform, sample_rate = torchaudio.load(audio_path)
    max_samples = int(sr*max_len)

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)

    samples = waveform.shape[1] 
    if samples > max_samples:
        waveform = waveform[:, :max_samples]
    elif samples < sr*max_samples:
        padding = max_samples - samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(waveform).squeeze(0)

    return mfcc

def train_test_dataloaders(data_path:str='data'):
    full_dataset = MFCCDataset(data_dir=data_path)
    train_size = int(len(full_dataset)*0.8)
    test_size = len(full_dataset)-train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32)

    return train_data_loader, test_data_loader

def predict_dataloader(predict_path:str, sr:int=16000, n_mfcc:int=20, max_len:float=1.0):
    dataset = MFCCDataset(data_dir=predict_path, sr=sr, n_mfcc=n_mfcc, max_len=max_len, with_labels=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader