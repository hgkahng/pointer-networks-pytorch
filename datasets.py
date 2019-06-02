# -*- coding: utf-8 -*-

import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class SortDataset(Dataset):
    def __init__(self, num_samples, length, random_seed=9):
        super(SortDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.dataset = []
        for _ in tqdm(range(num_samples)):
            x = torch.randperm(length)
            self.dataset.append(x)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':

    train_size = 1000
    data_length = 10
    train_dataset = SortDataset(
        num_samples=train_size,
        length=data_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
    )

    for i, x in enumerate(train_loader):
        print(x)
        break
        