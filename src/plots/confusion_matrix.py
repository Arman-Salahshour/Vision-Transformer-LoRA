"""Generates a confusion matrix for a trained model.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from constants import CLASS_NAMES, DEFAULT_DATA_DIR, device  
from data import get_dataloaders 
from model import get_vit
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    _, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)

    model = get_vit(num_classes=len(classes), pretrained=False)
    if args.mode == "lora":
        from lora import apply_lora

        model = apply_lora(model)
    model = load_checkpoint(model, f"output/{args.mode}_model.pt", map_location=device)
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalized: recall per class

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    readable_names = [CLASS_NAMES[c] for c in classes]
    ax.set_xticklabels(readable_names, rotation=45, ha="right")
    ax.set_yticklabels(readable_names)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({args.mode} fine-tuning)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    plt.tight_layout()
    plt.savefig(f"output/confusion_matrix_{args.mode}.png", dpi=150)
    print(f"Saved output/confusion_matrix_{args.mode}.png")


if __name__ == "__main__":
    main()