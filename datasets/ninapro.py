import os
import numpy as np
from torch.utils.data import Dataset
import torch

class Ninapro(Dataset):
    def __init__(self, emgs, labels) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).float()
        sample = sample.permute(1,0).unsqueeze(0)
        label = self.labels[index]-1
        label = torch.tensor(label).long()
        return sample, label