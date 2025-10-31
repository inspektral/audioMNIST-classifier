import os
import torch
import torchaudio

from torch.utils.data import Dataset

import utils

class MFCCDataset(Dataset):

    def __init__(self, data_dir, sr=16000, n_mfcc = 20, max_len = 1.0, with_labels=True):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.with_labels = with_labels

        self.annotations = []
        self._load_annotations()

    def _load_annotations(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    if self.with_labels:
                        label = int(file.split('_')[0])
                        spkr = int(file.split('_')[1])
                        self.annotations.append((file_path, label, spkr))
                    else:
                        self.annotations.append((file_path, None, None))

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        file_path, label, speaker = self.annotations[idx]
        
        mfcc = utils.load_mfcc(file_path, self.sr, self.n_mfcc, self.max_len)
        if self.with_labels:
            label = torch.tensor(label, dtype=torch.long)
            speaker = torch.tensor(speaker, dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)
            speaker = torch.tensor(-1, dtype=torch.long)

        return mfcc, label, speaker, file_path
    
if __name__ == "__main__":
    dataset = MFCCDataset(data_dir='data')
    print(f"Dataset size: {len(dataset)}")
    mfcc, label = dataset[0]
    print(f"MFCC shape: {mfcc.shape}, Label: {label}")

