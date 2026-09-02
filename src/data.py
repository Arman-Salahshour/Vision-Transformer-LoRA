"""ImageWoof dataset loading.
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from constants import IMG_SIZE, VIT_MEAN, VIT_STD


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(VIT_MEAN, VIT_STD),
            ]
        )
    return transforms.Compose( # -> standardized val-data, no randomness
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(VIT_MEAN, VIT_STD),
        ]
    )


def get_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 4):
    """data_dir must contain 'train/' and 'val/' subfolders (ImageFolder format)."""
    train_set = ImageFolder(f"{data_dir}/train", transform=get_transforms(train=True))
    val_set = ImageFolder(f"{data_dir}/val", transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_set.classes
