import argparse
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from constants import DEFAULT_DATA_DIR, LORA_ALPHA, LORA_RANK, SEED, device
from data import get_dataloaders
from model import get_vit
from utils import AverageMeter, Timer, save_checkpoint, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update((preds == labels).float().mean().item(), images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update((preds == labels).float().mean().item(), images.size(0))
    return loss_meter.avg, acc_meter.avg


def build_model(mode: str, num_classes: int, rank: int = 10, alpha: float = None):
    model = get_vit(num_classes=num_classes, pretrained=True)
    if mode == "lora":
        from lora import apply_lora 

        model = apply_lora(model, rank=rank, alpha=alpha)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora"], default="full")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK, help="LoRA rank r (ignored in full mode)")
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA, help="LoRA alpha scaling (ignored in full mode)")
    args = parser.parse_args()

    set_seed(args.seed)

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    model = build_model(args.mode, num_classes=len(classes), rank=args.lora_rank, alpha=args.lora_alpha).to(device)

    set_seed(args.seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    total_train_time = 0.0
    total_val_time = 0.0
    for epoch in range(args.epochs):
        with Timer() as train_timer:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        with Timer() as val_timer:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        total_train_time += train_timer.elapsed
        total_val_time += val_timer.elapsed
        print(
            f"epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} ({train_timer.elapsed:.1f}s) | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} ({val_timer.elapsed:.1f}s)"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_time_seconds": train_timer.elapsed,
            "val_time_seconds": val_timer.elapsed,
        })

    print(f"Training time ({args.mode}): {total_train_time:.1f}s (val: {total_val_time:.1f}s)")

    ckpt_path = f"output/{args.mode}_model.pt"
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())

    results = {
        "mode": args.mode,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        },
        "training_time_seconds": total_train_time,
        "validation_time_seconds": total_val_time,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "trainable_pct": 100 * trainable_count / total_count,
        "history": history,
        "final_val_acc": history[-1]["val_acc"],
        "final_train_acc": history[-1]["train_acc"],
    }

    results_path = f"output/{args.mode}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
