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



def apply_lora(model, rank: int = 10, alpha: float = None):
    for name, param in model.named_parameters():
        if name in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    for block in model.blocks:
        # Wrap linear layers in the attention block
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha).to(device)
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha).to(device)

        # Unfreeze the attention block
        block.attn.requires_grad_(True)

        # Unfreeze LayerScale layers as well
        block.ls1.requires_grad_(True)
        block.ls2.requires_grad_(True)
            


def get_lora_trainable_params(model):
    """ Returns the trainable parameters of the model after applying LoRA """
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    return trainable_parameters