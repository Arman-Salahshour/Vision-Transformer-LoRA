"""ImageWoof dataset loading.
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # required input size for vit_tiny_patch16_224


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
