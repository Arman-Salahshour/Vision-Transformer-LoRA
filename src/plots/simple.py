"""Generates comparison plots from output/full_results.json and
output/lora_results.json for the report. Run after both training
runs have completed.
"""

import json

import matplotlib.pyplot as plt

RESULTS = {
    "full": "output/full_results.json",
    "lora": "output/lora_results.json",
}
COLORS = {"full": "tab:blue", "lora": "tab:orange"}


def load_results():
    results = {}
    for mode, path in RESULTS.items():
        try:
            with open(path) as f:
                results[mode] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping {mode}")
    return results


def plot_loss_curves(results):
    plt.figure(figsize=(6, 4))
    for mode, res in results.items():
        epochs = [h["epoch"] for h in res["history"]]
        train_loss = [h["train_loss"] for h in res["history"]]
        val_loss = [h["val_loss"] for h in res["history"]]
        plt.plot(epochs, train_loss, "--", color=COLORS[mode], label=f"{mode} train")
        plt.plot(epochs, val_loss, "-", color=COLORS[mode], label=f"{mode} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/loss_curves.png", dpi=150)
    plt.close()


def plot_accuracy_curves(results):
    plt.figure(figsize=(6, 4))
    for mode, res in results.items():
        epochs = [h["epoch"] for h in res["history"]]
        train_acc = [h["train_acc"] for h in res["history"]]
        val_acc = [h["val_acc"] for h in res["history"]]
        plt.plot(epochs, train_acc, "--", color=COLORS[mode], label=f"{mode} train")
        plt.plot(epochs, val_acc, "-", color=COLORS[mode], label=f"{mode} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/accuracy_curves.png", dpi=150)
    plt.close()


def plot_bar_comparison(results, key, ylabel, title, filename, pct=False):
    modes = list(results.keys())
    values = [results[m][key] for m in modes]
    plt.figure(figsize=(4, 4))
    bars = plt.bar(modes, values, color=[COLORS[m] for m in modes])
    for bar, v in zip(bars, values):
        label = f"{v:.1f}%" if pct else f"{v:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, v, label, ha="center", va="bottom")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"output/{filename}", dpi=150)
    plt.close()


def main():
    results = load_results()
    if len(results) < 2:
        print("Need both full_results.json and lora_results.json for comparison plots.")
        return

    plot_loss_curves(results)
    plot_accuracy_curves(results)
    plot_bar_comparison(results, "final_val_acc", "Validation accuracy", "Final validation accuracy", "final_val_acc.png")
    plot_bar_comparison(results, "training_time_seconds", "Seconds", "Training time", "training_time.png")
    plot_bar_comparison(results, "trainable_pct", "% of parameters", "Trainable parameters", "trainable_pct.png", pct=True)

    print("Saved plots to output/: loss_curves.png, accuracy_curves.png, final_val_acc.png, training_time.png, trainable_pct.png")


if __name__ == "__main__":
    main()