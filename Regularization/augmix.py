from torchvision.transforms import (
    AutoAugment, AutoAugmentPolicy,
    Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
)

def augmix_train_transform(mean, std):
    return Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # proxy-AugMix
        ToTensor(),
        Normalize(mean, std),
    ])