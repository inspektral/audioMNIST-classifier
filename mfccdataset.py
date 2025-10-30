import os
import torch
import torchaudio

from torch.utils.data import Dataset

class MFCCDataset(Dataset):

    def __init__(self, data_dir, sr=16000, n_mfcc = 20):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mfcc = n_mfcc

        self.annotations = []
        self._load_annotations()

    def _load_annotations(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    label = file.split('_')[0]
                    self.annotations.append((file_path, label))

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_path, label = self.annotations[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = resampler(waveform)

        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc)
        mfcc = mfcc_transform(waveform).squeeze(0)

        return mfcc, label
    
if __name__ == "__main__":
    dataset = MFCCDataset(data_dir='data')
    print(f"Dataset size: {len(dataset)}")
    mfcc, label = dataset[0]
    print(f"MFCC shape: {mfcc.shape}, Label: {label}")

