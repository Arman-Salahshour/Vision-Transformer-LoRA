"""LoRA (Low-Rank Adaptation) wrapper for the pre-trained ViT.
"""

import torch
import torch.nn as nn
from constants import device

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = None): 
        super().__init__()

        assert isinstance(linear, nn.Linear), "Input must be a nn.Linear layer"

        self.register_buffer('orig_weight', linear.weight.data.detach())

        if linear.bias is not None:
            self.register_buffer('orig_bias', linear.bias.data.detach())
        else:
            self.register_buffer('orig_bias', None)

        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.alpha = alpha
        self.rank = rank

        self.scaler = alpha / rank if alpha is not None else 1.0

        self.lora_linear = nn.Sequential(
            nn.Linear(self.in_dim, self.rank, bias=False),
            nn.Linear(self.rank, self.out_dim, bias=False)
        )


        self.lora_linear[0].weight.data.normal_(mean=0.0, std=0.01)
        self.lora_linear[1].weight.data.zero_()


    def forward(self, x):
        orig_out = nn.functional.linear(x, self.orig_weight, self.orig_bias)
        lora_out = self.lora_linear(x) * self.scaler
        return orig_out + lora_out



