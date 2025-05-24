from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

def get_cifar10_loaders(batch_size: int = 128, val_split: float = 0.1, use_augmix=False):
    """Возвращает train/val/test DataLoader'ы для CIFAR‑10."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    if use_augmix:
        from Regularization.augmix import augmix_train_transform
        transform_train = augmix_train_transform(MEAN, STD)

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