"""LoRA (Low-Rank Adaptation) wrapper for the pre-trained ViT.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.
    """

    def __init__(self):

        raise NotImplementedError()

    def forward():
        raise NotImplementedError()


def apply_lora():
    raise NotImplementedError()


def get_lora_trainable_params():
    raise NotImplementedError()
