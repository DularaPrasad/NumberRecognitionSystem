"""
compare_methods.py  -  TASK 2 EXPERIMENT: which segmentation technique is best?
==============================================================================
The brief asks us to "research and experiment with different image
segmentation techniques and select the appropriate technique". This script
is that experiment.

HOW IT WORKS
  1. Build hundreds of number images from real MNIST digits (2-6 digits each)
     under 5 conditions, from easy to hard:
        - Well spaced   : clear gaps between digits
        - Close         : gaps of 0-3 pixels
        - Touching      : digits overlap / touch each other
        - Noisy         : random dots of "dust" added
        - Paper photo   : dark ink on unevenly lit paper with grain
  2. Because WE built the images, we know where every digit really is.
  3. Run all 5 techniques (4 rule-based + the trained U-Net) on every
     image and score them. The test numbers are built from MNIST TEST
     digits, which the U-Net never saw during training.

HOW A RESULT IS SCORED
  A digit counts as correctly segmented when one segment contains
      >= 85% of that digit's ink   (coverage - nothing important cut off)
  and that segment is >= 90% made of that digit   (purity - no neighbour bits).
  A NUMBER is correct only if the count is right AND every digit is correct,
  in the right left-to-right order.

RUN IT
    python compare_methods.py              (300 images per condition)
    python compare_methods.py --n 100      (faster)
OUTPUT (in results/)
    segmentation_results.csv      the numbers, for your report table
    segmentation_comparison.png   bar chart, for your report
    segmentation_examples.png     picture of each method on each condition
"""

import argparse
import csv
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")  # draw charts to files (no window needed)
import matplotlib.pyplot as plt
import numpy as np

from mnist_loader import load_mnist
from number_builder import build_number
from segmentation import METHODS, draw_boxes, segment_number

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CONDITIONS = {
    "Well spaced": dict(gap=(4, 10)),
    "Close": dict(gap=(0, 3)),
    "Touching": dict(gap=(-5, -1)),
    "Noisy": dict(gap=(2, 8), specks=25),
    "Paper photo": dict(gap=(2, 8), paper=True),
}
METHOD_NAMES = {
    "projection": "Projection profile",
    "connected_components": "Connected components",
    "contours": "Contours",
    "hybrid": "Hybrid (rule-based)",
    "unet": "U-Net (trained)",
}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # one colour per method


def score(segments, binary, owner, n_digits, min_cov=0.85, min_pur=0.90):
    """Compare the segments with the answer key.

    Returns (number_correct: bool, digits_correct: int, count_correct: bool)
    """
    visible = (binary > 0) & (owner >= 0)
    total_ink = np.bincount(owner[visible], minlength=n_digits)  # ink per digit

    good = {}          # digit index -> True if a segment captured it well
    order = []         # majority digit of each segment, left to right
    for s in segments:
        x, y, w, h = s.box
        ys, xs = np.nonzero(s.mask)
        owners = owner[ys + y, xs + x]
        owners = owners[owners >= 0]
        if len(owners) == 0:
            order.append(-1)          # segment of pure noise
            continue
        counts = np.bincount(owners, minlength=n_digits)
        k = int(counts.argmax())
        purity = counts[k] / len(owners)
        coverage = counts[k] / max(1, total_ink[k])
        order.append(k)
        if purity >= min_pur and coverage >= min_cov:
            good[k] = True

    count_ok = len(segments) == n_digits
    number_ok = count_ok and order == list(range(n_digits)) and len(good) == n_digits
    return number_ok, len(good), count_ok


def run(n_per_condition=300, seed=42):
    images, labels = load_mnist(train=False)
    rng = np.random.default_rng(seed)
    rows = []
    examples = {}

    for cond, kwargs in CONDITIONS.items():
        # Make the test images for this condition once, reuse for every method.
        tests = []
        for _ in range(n_per_condition):
            n = int(rng.integers(2, 7))
            idx = rng.integers(0, len(images), n)
            img, owner = build_number([images[i] for i in idx], rng=rng, **kwargs)
            tests.append((img, owner, n, "".join(str(labels[i]) for i in idx)))
        examples[cond] = tests[0]

        for method in METHODS:
            num_ok = dig_ok = cnt_ok = total_digits = 0
            start = time.perf_counter()
            for img, owner, n, _ in tests:
                segs, binary = segment_number(img, method=method, return_binary=True)
                a, b, c = score(segs, binary, owner, n)
                num_ok += a
                dig_ok += b
                cnt_ok += c
                total_digits += n
            ms = 1000 * (time.perf_counter() - start) / n_per_condition
            rows.append({
                "condition": cond,
                "method": METHOD_NAMES[method],
                "number_accuracy_%": round(100 * num_ok / n_per_condition, 1),
                "digit_accuracy_%": round(100 * dig_ok / total_digits, 1),
                "count_accuracy_%": round(100 * cnt_ok / n_per_condition, 1),
                "ms_per_image": round(ms, 2),
            })
            print(f"{cond:12s} | {METHOD_NAMES[method]:22s} | "
                  f"number {rows[-1]['number_accuracy_%']:5.1f}% | "
                  f"digit {rows[-1]['digit_accuracy_%']:5.1f}% | {ms:5.2f} ms")
    return rows, examples


def save_csv(rows):
    path = os.path.join(RESULTS, "segmentation_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", path)


def save_chart(rows):
    """Grouped bar chart: number accuracy for each condition and method."""
    conds = list(CONDITIONS)
    methods = list(METHOD_NAMES.values())
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m, (name, colour) in enumerate(zip(methods, COLOURS)):
        vals = [next(r["number_accuracy_%"] for r in rows
                     if r["condition"] == c and r["method"] == name) for c in conds]
        xs = np.arange(len(conds)) + (m - (len(methods) - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=colour, label=name)
    ax.set_xticks(np.arange(len(conds)), conds)
    ax.set_ylabel("Numbers segmented perfectly (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Segmentation technique comparison on MNIST-built numbers", loc="left")
    ax.yaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    path = os.path.join(RESULTS, "segmentation_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


def save_examples(examples):
    """Grid of pictures: one row per condition, one column per method."""
    fig, axes = plt.subplots(len(examples), len(METHODS), figsize=(3.5 * len(METHODS), 2.0 * len(examples)))
    for r, (cond, (img, owner, n, text)) in enumerate(examples.items()):
        for c, method in enumerate(METHODS):
            segs = segment_number(img, method=method)
            vis = draw_boxes(img, segs)
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(METHOD_NAMES[method], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cond}\n(true: {text})", fontsize=9)
            ax.set_xlabel(f"found {len(segs)} of {n}", fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS, "segmentation_examples.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="test images per condition")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rows, examples = run(args.n, args.seed)
    save_csv(rows)
    save_chart(rows)
    save_examples(examples)
