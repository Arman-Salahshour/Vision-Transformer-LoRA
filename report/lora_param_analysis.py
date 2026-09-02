"""Parameter accounting and rank-scaling figure for vit_tiny_patch16_224 + LoRA.

Derived analytically from the timm ViT-Tiny spec:
    embed_dim D = 192, depth L = 12, heads = 3, mlp_ratio = 4,
    patch = 16, img = 224 -> 196 patches + 1 cls token = 197 tokens.

LoRA is applied to the attention projections (qkv, proj) only; the final
LayerNorm and the classification head are additionally trainable.

Usage:
    python lora_param_analysis.py                 # print table + write figure
    python lora_param_analysis.py --no-figure     # print table only
    python lora_param_analysis.py --rank 16       # highlight a different rank
"""

import argparse

D = 192          # embed_dim
L = 12           # depth
MLP = 4 * D      # 768
TOKENS = 197     # 196 patches + cls
NUM_CLASSES = 10
PATCH = 16
IN_CH = 3

RANKS = [1, 2, 4, 8, 10, 16, 32]
FIGURE_PATH = "lora_param_scaling.png"


def backbone_params():
    """Parameter counts for the unwrapped ViT-Tiny with a NUM_CLASSES head."""
    patch_embed = IN_CH * D * PATCH * PATCH + D
    cls_token = D
    pos_embed = TOKENS * D

    norm1 = 2 * D
    qkv = D * (3 * D) + 3 * D
    proj = D * D + D
    norm2 = 2 * D
    fc1 = D * MLP + MLP
    fc2 = MLP * D + D
    block = norm1 + qkv + proj + norm2 + fc1 + fc2

    final_norm = 2 * D
    head = D * NUM_CLASSES + NUM_CLASSES

    total = patch_embed + cls_token + pos_embed + L * block + final_norm + head
    return {
        "patch_embed": patch_embed,
        "cls_token": cls_token,
        "pos_embed": pos_embed,
        "block": block,
        "blocks_total": L * block,
        "qkv_per_block": qkv,
        "proj_per_block": proj,
        "final_norm": final_norm,
        "head": head,
        "total": total,
    }


def lora_params(rank):
    """Adapter parameters across all blocks. A and B are bias-free.

    qkv: A (D -> r), B (r -> 3D)   |   proj: A (D -> r), B (r -> D)
    Per block this is r(D + 3D) + r(D + D) = 1152r.
    """
    qkv_lora = rank * D + rank * (3 * D)
    proj_lora = rank * D + rank * D
    return L * (qkv_lora + proj_lora)


def report(rank):
    b = backbone_params()
    head_norm = b["final_norm"] + b["head"]
    full = b["total"]

    print("Component breakdown (unwrapped model)")
    for k, v in b.items():
        print(f"  {k:>16}: {v:,}")

    displaced = L * (b["qkv_per_block"] + b["proj_per_block"])
    adapters = lora_params(rank)
    trainable = adapters + head_norm

    print()
    print(f"{'rank':>6} {'adapters':>12} {'+norm/head':>12} {'% of full':>12}")
    for r in RANKS:
        t = lora_params(r) + head_norm
        print(f"{r:>6} {lora_params(r):>12,} {t:>12,} {100 * t / full:>11.3f}%")

    print()
    print(f"  full fine-tuning trainable : {full:,}")
    print(f"  rank={rank} trainable{'':<12}: {trainable:,}")
    print(f"  share of full              : {100 * trainable / full:.3f}%")
    print(f"  compression factor         : {full / trainable:.1f}x")
    print(f"  weights displaced to buffers: {displaced:,}")
    print(f"  -> len(model.parameters())  : {full - displaced + adapters:,}")


def make_figure(path=FIGURE_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = backbone_params()
    head_norm = b["final_norm"] + b["head"]
    full = b["total"]

    trainable = [lora_params(r) + head_norm for r in RANKS]
    pct = [100 * t / full for t in trainable]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax1.plot(RANKS, trainable, marker="o", color="#1f4e79", lw=1.8,
             label="LoRA (qkv + proj) + norm + head")
    ax1.axhline(full, color="#a63603", ls="--", lw=1.5,
                label=f"Full fine-tuning ({full / 1e6:.2f}M)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(RANKS)
    ax1.set_xticklabels(RANKS)
    ax1.set_xlabel("LoRA rank $r$")
    ax1.set_ylabel("Trainable parameters")
    ax1.set_title("Trainable parameter budget", fontsize=10)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(fontsize=7, loc="upper left")

    ax2.bar([str(r) for r in RANKS], pct, color="#1f4e79", width=0.6)
    ax2.axhline(100, color="#a63603", ls="--", lw=1.5)
    ax2.set_xlabel("LoRA rank $r$")
    ax2.set_ylabel("% of full fine-tuning")
    ax2.set_title("Relative parameter cost", fontsize=10)
    ax2.grid(True, axis="y", ls=":", alpha=0.4)
    for i, p in enumerate(pct):
        ax2.text(i, p + 0.25, f"{p:.2f}", ha="center", fontsize=7)
    ax2.set_ylim(0, 9.5)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--out", default=FIGURE_PATH)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    report(args.rank)
    if not args.no_figure:
        make_figure(args.out)


if __name__ == "__main__":
    main()
