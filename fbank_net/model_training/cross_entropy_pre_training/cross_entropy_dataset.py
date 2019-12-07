import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder


class FBanksCrossEntropyDataset(Dataset):
    def __init__(self, root):
        self.dataset_folder = DatasetFolder(root=root, loader=FBanksCrossEntropyDataset._npy_loader, extensions='.npy')
        self.len_ = len(self.dataset_folder.samples)

        bin_counts = np.bincount(self.dataset_folder.targets)
        self.num_classes = len(self.dataset_folder.classes)
        self.label_to_index_range = {}
        start = 0
        for i in range(self.num_classes):
            self.label_to_index_range[i] = (start, start + bin_counts[i])
            start = start + bin_counts[i]

    @staticmethod
    def _npy_loader(path):
        sample = np.load(path)
        assert sample.shape[0] == 64
        assert sample.shape[1] == 64
        assert sample.shape[2] == 1

        sample = np.moveaxis(sample, 2, 0)  # pytorch expects input in the format in_channels x width x height
        sample = torch.from_numpy(sample).float()

        return sample

    def __getitem__(self, index):
        return self.dataset_folder[index]

    def __len__(self):
        return self.len_
