"""
train_seg.py  -  TRAIN any of the four segmentation networks
============================================================
All four networks are trained in EXACTLY the same way, so the comparison
between them is fair:
  * the same training examples (made from MNIST training digits, a new set
    every epoch, same random seed),
  * the same validation set (600 examples from held-out digits),
  * the same loss (weighted cross-entropy; border pixels weighted 4x because
    they are only about 0.2% of all pixels),
  * the same optimiser (Adam, learning rate 0.001 lowered gradually),
  * the same number of epochs and batches.
The version with the lowest validation loss is kept.

RUN IT (inside Task2_Segmentation, with the venv switched on):
    python train_seg.py --model unet
    python train_seg.py --model resnet
    python train_seg.py --model mobilenet
    python train_seg.py --model fcdensenet
    python train_seg.py --model all          (trains all four, one after another)

OUTPUT
    models/<model>_seg.pt                       the trained network
    results/training_history_<model>.csv        loss and accuracy per epoch
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mnist_loader import load_mnist
from seg_models import MODEL_NAMES, build_model
from unet_data import make_batch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
CLASS_WEIGHTS = [0.25, 1.0, 4.0]   # background, inside a digit, border


def model_path(name):
    return os.path.join(MODELS, f"{name}_seg.pt")


def evaluate(net, x_val, y_val, loss_fn, batch=100):
    """Validation loss, ink-pixel accuracy and border recall."""
    net.eval()
    total, correct, ink, b_hit, b_all = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(x_val), batch):
            xb, yb = torch.from_numpy(x_val[i:i + batch]), torch.from_numpy(y_val[i:i + batch])
            out = net(xb)
            total += loss_fn(out, yb).item() * len(xb)
            pred = out.argmax(1)
            on_ink, border = yb > 0, yb == 2
            correct += (pred[on_ink] == yb[on_ink]).sum().item()
            ink += on_ink.sum().item()
            b_hit += (pred[border] == 2).sum().item()
            b_all += border.sum().item()
    net.train()
    return total / len(x_val), correct / max(1, ink), b_hit / max(1, b_all)


def train(name, epochs, steps, batch, lr, seed):
    torch.manual_seed(seed)
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    images, _ = load_mnist(train=True)
    train_digits, val_digits = images[:54000], images[54000:]
    x_val, y_val = make_batch(val_digits, np.random.default_rng(12345), 600)

    net = build_model(name)
    n_params = sum(p.numel() for p in net.parameters())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS))
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    print(f"\n=== {MODEL_NAMES[name]}  ({n_params:,} parameters) ===")
    history, best, t_start = [], float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng = np.random.default_rng(seed * 1000 + epoch)
        running = 0.0
        for _ in range(steps):
            xb, yb = make_batch(train_digits, rng, batch)
            xb, yb = torch.from_numpy(xb), torch.from_numpy(yb)
            optimiser.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item()
        scheduler.step()
        val_loss, ink_acc, border_rec = evaluate(net, x_val, y_val, loss_fn)
        row = {"epoch": epoch, "train_loss": round(running / steps, 4), "val_loss": round(val_loss, 4),
               "val_ink_acc": round(ink_acc, 4), "val_border_recall": round(border_rec, 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        saved = ""
        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), model_path(name))
            saved = "  <- best so far, saved"
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"ink acc {100 * ink_acc:5.1f}% | border found {100 * border_rec:5.1f}% | {row['seconds']}s{saved}")

    with open(os.path.join(RESULTS, f"training_history_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    minutes = (time.time() - t_start) / 60
    print(f"Done in {minutes:.1f} min. Best model saved to {model_path(name)}")


def main():
    ap = argparse.ArgumentParser(description="Train a segmentation network")
    ap.add_argument("--model", required=True, choices=list(MODEL_NAMES) + ["all"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = list(MODEL_NAMES) if a.model == "all" else [a.model]
    for name in names:
        train(name, a.epochs, a.steps, a.batch, a.lr, a.seed)


if __name__ == "__main__":
    main()
