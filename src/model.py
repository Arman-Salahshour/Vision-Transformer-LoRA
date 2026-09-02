"""Pre-trained ViT loading via timm.
"""

import timm
import torch.nn as nn

from constants import MODEL_NAME, NUM_CLASSES


def get_vit(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model


# used for just train the head of the model.
def freeze_all_but_head(model: nn.Module) -> None:
    """Not used for full fine-tuning, but handy for sanity checks."""
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name
