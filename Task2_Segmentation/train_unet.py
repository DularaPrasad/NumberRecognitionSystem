"""
train_unet.py  -  TRAIN the U-Net segmentation network
======================================================
What happens, step by step:
  1. Make a fixed VALIDATION set (600 examples) to check progress honestly.
  2. For every EPOCH (one round of training):
       - make fresh training examples from MNIST train digits,
       - show them to the network in BATCHES of 32,
       - the network guesses a label for every pixel,
       - the LOSS measures how wrong the guesses were,
       - the OPTIMISER (Adam) nudges the network's numbers to reduce the loss.
  3. After every epoch, measure the loss and accuracy on the validation set.
  4. Keep the version of the network that did best on validation
     (saved as models/unet_seg.pt) and draw the training curve.

RUN IT (inside Task2_Segmentation, with the venv switched on):
    python train_unet.py                 # default: 12 epochs (about 15-20 min on a laptop CPU)
    python train_unet.py --epochs 3      # quick test run
OUTPUT
    models/unet_seg.pt                   # the trained network (used by segmentation.py)
    results/unet_training_history.csv    # loss and accuracy for every epoch
    results/unet_training_curve.png      # figure for the report
"""

import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mnist_loader import load_mnist
from unet_data import make_batch
from unet_model import UNet

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "unet_seg.pt")
RESULTS = os.path.join(HERE, "results")

# Border pixels are rare (about 0.2% of pixels) but they matter most, so a
# mistake on a border pixel costs more than a mistake on a background pixel.
CLASS_WEIGHTS = [0.25, 1.0, 4.0]   # background, interior, border


def evaluate(net, x_val, y_val, loss_fn, batch=100):
    """Loss, ink-pixel accuracy and border recall on the validation set."""
    net.eval()
    total_loss, correct, ink, border_hit, border_all = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(x_val), batch):
            xb = torch.from_numpy(x_val[i:i + batch])
            yb = torch.from_numpy(y_val[i:i + batch])
            out = net(xb)
            total_loss += loss_fn(out, yb).item() * len(xb)
            pred = out.argmax(1)
            on_ink = yb > 0
            correct += (pred[on_ink] == yb[on_ink]).sum().item()
            ink += on_ink.sum().item()
            is_border = yb == 2
            border_hit += (pred[is_border] == 2).sum().item()
            border_all += is_border.sum().item()
    net.train()
    return total_loss / len(x_val), correct / max(1, ink), border_hit / max(1, border_all)


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(epochs, [h["train_loss"] for h in history], color="#2a78d6", lw=2, marker="o", ms=4, label="Training loss")
    a1.plot(epochs, [h["val_loss"] for h in history], color="#eb6834", lw=2, marker="o", ms=4, label="Validation loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss (lower is better)"); a1.legend(frameon=False)
    a1.set_title("Loss", loc="left")
    a2.plot(epochs, [100 * h["val_ink_acc"] for h in history], color="#2a78d6", lw=2, marker="o", ms=4, label="Ink pixels labelled correctly")
    a2.plot(epochs, [100 * h["val_border_recall"] for h in history], color="#eb6834", lw=2, marker="o", ms=4, label="Border pixels found")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation (%)"); a2.set_ylim(0, 100); a2.legend(frameon=False)
    a2.set_title("Validation accuracy", loc="left")
    for a in (a1, a2):
        a.grid(True, color="#e5e5e5"); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the U-Net segmentation network")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    print("Loading MNIST training digits ...")
    images, _ = load_mnist(train=True)
    # Hold out the last 6,000 training digits for validation, so validation
    # examples never contain a digit the network trained on.
    train_digits, val_digits = images[:54000], images[54000:]

    print("Making the validation set (600 examples) ...")
    x_val, y_val = make_batch(val_digits, np.random.default_rng(12345), 600)

    net = UNet()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS))
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    # Lower the learning rate gradually so the last epochs make fine adjustments.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history, best = [], float("inf")
    print(f"Training for {args.epochs} epochs x {args.steps} batches of {args.batch} ...\n")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)   # new examples every epoch
        running = 0.0
        for step in range(args.steps):
            xb, yb = make_batch(train_digits, rng, args.batch)
            xb, yb = torch.from_numpy(xb), torch.from_numpy(yb)
            optimiser.zero_grad()
            loss = loss_fn(net(xb), yb)     # 1) guess  2) measure how wrong
            loss.backward()                 # 3) work out which way to nudge each number
            optimiser.step()                # 4) nudge
            running += loss.item()
        scheduler.step()

        val_loss, ink_acc, border_rec = evaluate(net, x_val, y_val, loss_fn)
        row = {"epoch": epoch, "train_loss": round(running / args.steps, 4), "val_loss": round(val_loss, 4),
               "val_ink_acc": round(ink_acc, 4), "val_border_recall": round(border_rec, 4),
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        saved = ""
        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), MODEL_PATH)
            saved = "  <- best so far, saved"
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"ink acc {100 * ink_acc:5.1f}% | border found {100 * border_rec:5.1f}% | {row['seconds']}s{saved}")

    with open(os.path.join(RESULTS, "unet_training_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_history(history, os.path.join(RESULTS, "unet_training_curve.png"))
    print(f"\nDone. Best model saved to {MODEL_PATH}")
    print("Saved results/unet_training_history.csv and results/unet_training_curve.png")


if __name__ == "__main__":
    main()
