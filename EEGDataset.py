from torch.utils.data import Dataset
import torch

class EEGDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.data
        x, y = x[idx], y[idx]
        out = {'feature': x, 'label': y}

        return out