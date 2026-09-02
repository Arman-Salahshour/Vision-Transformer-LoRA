# Method: Low-Rank Adaptation of ViT-Tiny

## 1. Formulation

Let $W_0 \in \mathbb{R}^{d \times k}$ be a frozen pre-trained projection. Full fine-tuning optimises
$W_0 + \Delta W$ over all $dk$ entries. LoRA [1] restricts the update to a rank-$r$ subspace,
$\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$:

$$h = W_0 x + \frac{\alpha}{r} B A x .$$

Cost per adapted layer drops from $dk$ to $r(d + k)$, and the $\alpha / r$ scaling keeps the update
magnitude independent of rank, so sweeping $r$ does not implicitly re-tune the learning rate.

We initialise $A \sim \mathcal{N}(0, 0.01^2)$ and $B = 0$, which makes $\Delta W = 0$ at step 0. The
adapted network therefore starts out identical to the pre-trained checkpoint. The asymmetry
matters: zero-initialising both factors would sit on a saddle point where neither gets gradient.

## 2. Implementation

`LoRALinear` (in `src/lora.py`) is a drop-in replacement for `nn.Linear`. Three decisions are worth
stating.

**Frozen weights are buffers, not parameters.** The wrapper registers `orig_weight` as a buffer
rather than keeping the original `nn.Parameter`. Buffers are excluded from `model.parameters()`, so
the backbone cannot receive gradients whatever happens to `requires_grad` later. That is stronger
than `requires_grad = False`, which a later `requires_grad_(True)` on a parent module would quietly
undo. Buffers still appear in `state_dict()`, so checkpoints stay self-contained.

**Bias handling covers both branches.** With no bias on the wrapped layer, `orig_bias` is registered
as `None`, which `F.linear` accepts, so the forward path stays uniform. Both `qkv` and `proj` in
`vit_tiny_patch16_224` do carry biases, but the branch keeps the wrapper correct for
`qkv_bias=False` and for the MLP layers.

**The forward pass adds a parallel branch** instead of reconstructing $W_0 + BA$. Computing $B(Ax)$
costs $\mathcal{O}(r(d+k))$ per token against $\mathcal{O}(dk)$, and avoids allocating a full
$d \times k$ tensor per layer per step.

`apply_lora` freezes the backbone first, then substitutes the wrapped modules. The order is
load-bearing: adapters are built after the freeze loop, so they inherit `requires_grad=True`.
Reversing the stages would sweep `blocks.*.attn.{qkv,proj}.lora_linear.*` into the freeze and leave
only the head trainable, a failure that raises no error and degrades silently into linear probing.

## 3. Adaptation surface

We adapt the attention projections only (`attn.qkv`, $\mathbb{R}^{192 \to 576}$, and `attn.proj`,
$\mathbb{R}^{192 \to 192}$), following the ablation in [1], which finds this keeps most of the
achievable accuracy at a fraction of the cost. The MLP sublayers stay frozen.

Two groups are unfrozen alongside the adapters. `head`, necessarily, since timm builds a randomly
initialised 10-way classifier when `num_classes=10` is passed to `create_model`, and a frozen random
head cannot be trained. And `norm`, the final LayerNorm, a cheap concession at 384 parameters that
lets the network rescale features for the new label space. The second is a deliberate deviation from
vanilla LoRA. `ls1` and `ls2` are unfrozen defensively but resolve to `nn.Identity` here, since
LayerScale is only instantiated when `init_values` is set.

## 4. Parameter accounting

ViT-Tiny uses $D = 192$, $L = 12$ blocks, $\text{mlp\_ratio} = 4$, $16 \times 16$ patches at
$224 \times 224$. With a 10-way head the full model holds **5,526,346** parameters. Per block the
adapters add $r(D + 3D) + r(D + D) = 1152r$, so the budget across 12 blocks is exactly
$13{,}824\,r$. At $r = 10$ with $\alpha$ unset (scaling factor 1.0):

| Component | Parameters | Trainable |
|---|---:|:---:|
| Patch embedding | 147,648 | ✗ |
| `cls_token` + positional embedding | 38,016 | ✗ |
| Transformer blocks (frozen weights, now buffers) | 5,338,368 | ✗ |
| LoRA adapters, `qkv` (12 × 7,680) | 92,160 | ✓ |
| LoRA adapters, `proj` (12 × 3,840) | 46,080 | ✓ |
| Final LayerNorm | 384 | ✓ |
| Classification head | 1,930 | ✓ |
| **Trainable total** | **140,554** | |
| **Full fine-tuning total** | **5,526,346** | |

LoRA at $r = 10$ trains **2.54%** of what full fine-tuning updates, a **39.3×** smaller optimisation
problem. Since AdamW keeps two moments per trainable parameter, optimiser state falls from roughly
42 MB to 1.1 MB in fp32.

The buffer decision has one accounting consequence: after injection, 1,778,688 backbone weights are
no longer `nn.Parameter` objects, so `sum(p.numel() for p in model.parameters())` returns 3,885,898
rather than 5,526,346. We report percentages against the unwrapped count, the meaningful denominator
for a full-versus-LoRA comparison.

![Trainable parameter budget against LoRA rank. Left: absolute counts (log-log) with the full
fine-tuning reference. Right: the same figures as a percentage of full fine-tuning.](imgs/lora_param_scaling.png)

**Figure 1.** Trainable parameters scale linearly in $r$ (the budget is $13{,}824\,r + 2{,}314$), so
rank is a direct dial on the capacity/cost trade-off. Even at $r = 32$ it stays under 8.1% of full
fine-tuning.


### References

[1] E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
[2] A. Dosovitskiy et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at
Scale." ICLR, 2021.
