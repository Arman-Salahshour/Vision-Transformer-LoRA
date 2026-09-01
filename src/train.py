import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dataloaders
from model import get_vit
from utils import AverageMeter, Timer, get_device, save_checkpoint, set_seed

DEFAULT_DATA_DIR = "data/imagewoof2"


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


def build_model(mode: str, num_classes: int):
    model = get_vit(num_classes=num_classes, pretrained=True)
    if mode == "lora":
        from lora import apply_lora  # import lazy ->'full' mode works before lora.py is finished

        model = apply_lora(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora"], default="full")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size)
    model = build_model(args.mode, num_classes=len(classes)).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    with Timer() as t:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"epoch {epoch + 1}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

    print(f"Training time ({args.mode}): {t.elapsed:.1f}s")

    ckpt_path = f"output/{args.mode}_model.pt"
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
