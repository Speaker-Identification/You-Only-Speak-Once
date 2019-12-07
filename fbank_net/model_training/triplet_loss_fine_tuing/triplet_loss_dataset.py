import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder


class FBanksTripletDataset(Dataset):
    def __init__(self, root):
        self.dataset_folder = DatasetFolder(root=root, loader=FBanksTripletDataset._npy_loader, extensions='.npy')
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

        sample = np.moveaxis(sample, 2, 0)
        sample = torch.from_numpy(sample).float()

        return sample

    def __getitem__(self, index):
        anchor_x, anchor_y = self.dataset_folder[index]

        # find a positive
        start, end = self.label_to_index_range[anchor_y]
        i = np.random.randint(low=start, high=end)
        positive_x, positive_y = self.dataset_folder[i]

        #  find a negative
        l_ = list(range(self.num_classes))
        l_.pop(anchor_y)
        ny_ = np.random.choice(l_)
        start, end = self.label_to_index_range[ny_]
        i = np.random.randint(low=start, high=end)
        negative_x, negative_y = self.dataset_folder[i]

        return (anchor_x, anchor_y), (positive_x, positive_y), (negative_x, negative_y)

    def __len__(self):
        return self.len_
