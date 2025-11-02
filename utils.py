import torch
import torchaudio

from mfccdataset import MFCCDataset
from torch.utils.data import DataLoader, random_split

from config import SEED, N_MFCC, N_MELS, BATCH_SIZE

def load_mfcc(audio_path:str, sr:int=16000, n_mfcc:int=20, n_mels:int=128, max_len:float=1.0):
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

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs={'n_mels': n_mels})
    mfcc = mfcc_transform(waveform).squeeze(0)

    return mfcc


def train_test_dataloaders(data_path:str='data'):
    full_dataset = MFCCDataset(data_dir=data_path, n_mfcc=N_MFCC, n_mels=N_MELS)
    train_size = int(len(full_dataset)*0.8)
    test_size = len(full_dataset)-train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_data_loader, test_data_loader

def train_test_dataloaders_by_speaker(data_path:str='data'):
    full_dataset = MFCCDataset(data_dir=data_path, n_mfcc=N_MFCC, n_mels=N_MELS)
    speakers = set()
    for i in range(len(full_dataset)):
        filepath, label, speaker = full_dataset.get_annotation(i)
        speakers.add(speaker)

    speakers = list(speakers)
    num_speakers = len(speakers)
    num_train_speakers = int(num_speakers * 0.8)

    torch.manual_seed(SEED)
    speakers = torch.tensor(speakers)[torch.randperm(num_speakers)].tolist()

    train_speakers = speakers[:num_train_speakers]
    test_speakers = speakers[num_train_speakers:]

    train_indices = [i for i in range(len(full_dataset)) if full_dataset.get_annotation(i)[2] in train_speakers]
    test_indices = [i for i in range(len(full_dataset)) if full_dataset.get_annotation(i)[2] in test_speakers]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    train_data_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

    return train_data_loader, test_data_loader

def predict_dataloader(predict_path:str, sr:int=16000, n_mfcc:int=20, max_len:float=1.0):
    dataset = MFCCDataset(data_dir=predict_path, sr=sr, n_mfcc=n_mfcc, max_len=max_len, with_labels=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader

def add_log_entry(logs:list, epoch:int, train_accuracy:float, train_loss:float, test_accuracy:float, test_loss:float):
    
    logs.append({
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'train_loss': train_loss,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    })

    print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}")
    print(f"Epoch {epoch}: Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")