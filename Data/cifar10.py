# cifar10.py
# Taked from https://doi.org/10.48550/arXiv.2503.22725
"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_loaders(batch_size: int = 128, val_split: float = 0.1):
    """Возвращает train/val/test DataLoader'ы для CIFAR‑10."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_full = datasets.CIFAR10(root="./data", train=True, download=True,
                                  transform=transform_train)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True,
                               transform=transform_test)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader