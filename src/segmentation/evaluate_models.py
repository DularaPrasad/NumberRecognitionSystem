"""
evaluate_models.py  -  COMPREHENSIVE comparison of the four segmentation networks
=================================================================================
Accuracy alone does not describe a segmentation model well, so the four
trained networks (U-Net, ResNet-18 FCN, MobileNetV3 LR-ASPP, FC-DenseNet) are
compared on five groups of measures. The rule-based hybrid method is included
only as a REFERENCE baseline (it is not a neural network and has no pixel labels).

1. PIXEL-LEVEL (semantic segmentation) - how well each pixel is labelled
   * IoU (Intersection over Union, also called the Jaccard index) per class:
       IoU = correct pixels of a class / (pixels predicted as it + pixels that
       really are it - correct ones). The standard measure in PASCAL VOC.
   * mIoU: the mean IoU over the 3 classes, and frequency-weighted IoU (FCN paper).
   * Dice coefficient (= F1 score) per class, and its mean.
   * Precision and recall per class - the BORDER class matters most, because
     it is what separates touching digits.
   * Pixel accuracy and mean class accuracy (reported, but pixel accuracy is
     misleading here: about 95% of pixels are background).

2. INSTANCE-LEVEL (each digit as one object)
   Predicted digits are matched one-to-one to the real digits (Hungarian
   matching on mask IoU); a match counts when IoU > 0.5.
   * Digit precision, recall and F1 (F1 here = Recognition Quality, RQ).
   * Segmentation Quality SQ (mean IoU of the matched digits) and
     Panoptic Quality PQ = SQ x RQ (Kirillov et al., 2019).
   * Boundary F1 (Csurka et al., 2013): how closely the outline of each
     matched digit follows the real outline (within 2 pixels).
   * Digit-count accuracy, over-segmentation (too many pieces) and
     under-segmentation (digits merged) rates.

3. TASK-LEVEL
   * Number accuracy: whole number correct (right count, every digit has
     >= 85% of its ink and is >= 90% pure, right order) - with a 95%
     confidence interval (Wilson).

4. EFFICIENCY
   * Parameters, model file size, FLOPs for one 64x192 image,
     CPU time per image (network + post-processing), training time.

5. ROBUSTNESS
   * All measures for each condition: well spaced, close, touching, noisy,
     paper photo.

The test numbers are built from MNIST TEST digits (never seen in training).

RUN:   python evaluate_models.py            (300 numbers per condition, a few minutes)
       python evaluate_models.py --n 100    (faster)
OUTPUT (results/):
    model_comparison_overall.csv, model_comparison_by_condition.csv,
    pixel_metrics_per_class.csv, model_comparison.md,
    fig_metric_overview.png, fig_by_condition.png, fig_efficiency.png (size and speed),
    fig_training_curves.png, fig_pixel_examples.png
"""

import argparse
import csv
import math
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import nn_segment
import segmentation as seg
from compare_methods import CONDITIONS, score
from mnist_loader import load_mnist
from number_builder import build_number
from seg_models import MODEL_NAMES
from unet_data import digit_owner_map, make_labels

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
NETWORKS = list(MODEL_NAMES)                      # unet, resnet, mobilenet, fcdensenet
REFERENCE = "hybrid"
LABELS = {**MODEL_NAMES, REFERENCE: "Rule-based hybrid (reference)"}
COLOURS = {"unet": "#2a78d6", "resnet": "#eb6834", "mobilenet": "#1baf7a",
           "fcdensenet": "#eda100", REFERENCE: "#9a9a9a"}
CLASS_NAMES = ["background", "inside digit", "border"]


# --------------------------------------------------------------------------- helpers
BF_TOLERANCE = 2   # pixels: a boundary point counts as correct if within 2 px


def _outline(mask):
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_f1(pred_mask, gt_mask, tol=BF_TOLERANCE):
    """Boundary F1 (BF score, Csurka et al., 2013): how well the OUTLINE of a
    predicted digit follows the outline of the real digit, within `tol` pixels."""
    pb, gb = _outline(pred_mask), _outline(gt_mask)
    if not pb.any() or not gb.any():
        return 0.0
    d_to_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    d_to_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    precision = np.mean(d_to_g[pb] <= tol)
    recall = np.mean(d_to_p[gb] <= tol)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def instance_stats(segments, gt_ids, n_digits):
    """Match predicted digit masks to real digit masks; return counts and IoUs."""
    pred = np.zeros(gt_ids.shape, np.int32)
    for k, s in enumerate(segments, start=1):
        x, y, w, h = s.box
        region = pred[y:y + h, x:x + w]
        region[s.mask > 0] = k
    n_pred = len(segments)
    iou = np.zeros((n_digits, n_pred))
    for g in range(n_digits):
        gm = gt_ids == g
        if not gm.any():
            continue
        for p in range(n_pred):
            pm = pred == p + 1
            inter = np.count_nonzero(gm & pm)
            if inter:
                iou[g, p] = inter / np.count_nonzero(gm | pm)
    tp, iou_sum, bf_sum = 0, 0.0, 0.0
    if n_digits and n_pred:
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] > 0.5:
                tp += 1
                iou_sum += iou[r, c]
                bf_sum += boundary_f1(pred == c + 1, gt_ids == r)
    return {"tp": tp, "fp": n_pred - tp, "fn": n_digits - tp, "iou_sum": iou_sum, "bf_sum": bf_sum,
            "count_ok": int(n_pred == n_digits), "over": int(n_pred > n_digits), "under": int(n_pred < n_digits)}


def pixel_metrics(conf):
    """IoU, Dice, precision, recall per class from a 3x3 confusion matrix."""
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    pred_tot, true_tot = conf.sum(0), conf.sum(1)
    iou = tp / np.maximum(pred_tot + true_tot - tp, 1)
    dice = 2 * tp / np.maximum(pred_tot + true_tot, 1)
    prec = tp / np.maximum(pred_tot, 1)
    rec = tp / np.maximum(true_tot, 1)
    freq = true_tot / true_tot.sum()
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec,
            "miou": iou.mean(), "fwiou": (freq * iou).sum(), "mdice": dice.mean(),
            "pixel_acc": tp.sum() / conf.sum(), "mean_class_acc": rec.mean()}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def summarise(stats):
    """Turn a list of per-image instance stats into the instance/task measures."""
    tp = sum(s["tp"] for s in stats); fp = sum(s["fp"] for s in stats); fn = sum(s["fn"] for s in stats)
    iou_sum = sum(s["iou_sum"] for s in stats)
    n = len(stats)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    sq = iou_sum / max(1, tp)
    bf = sum(s["bf_sum"] for s in stats) / max(1, tp)
    num_ok = sum(s["number_ok"] for s in stats)
    lo, hi = wilson(num_ok, n)
    return {"digit_precision": 100 * prec, "digit_recall": 100 * rec, "digit_f1_RQ": 100 * f1,
            "SQ": 100 * sq, "PQ": 100 * sq * f1, "boundary_F1": 100 * bf,
            "count_accuracy": 100 * sum(s["count_ok"] for s in stats) / n,
            "over_segmentation": 100 * sum(s["over"] for s in stats) / n,
            "under_segmentation": 100 * sum(s["under"] for s in stats) / n,
            "number_accuracy": 100 * num_ok / n, "number_acc_ci_low": lo, "number_acc_ci_high": hi,
            "ms_per_image": 1000 * sum(s["seconds"] for s in stats) / n}


def model_cost(name):
    """Parameters, file size (MB) and FLOPs for one 64x192 image."""
    from torch.utils.flop_counter import FlopCounterMode
    net = nn_segment.load_model(name)
    params = sum(p.numel() for p in net.parameters())
    size_mb = os.path.getsize(nn_segment.model_path(name)) / 1e6
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            net(torch.zeros(1, 1, 64, 192))
    return params, size_mb, fc.get_total_flops() / 1e9


def training_minutes(name):
    path = os.path.join(RESULTS, f"training_history_{name}.csv")
    if not os.path.exists(path):
        return float("nan"), []
    rows = list(csv.DictReader(open(path)))
    return sum(float(r["seconds"]) for r in rows) / 60, rows


# --------------------------------------------------------------------------- main experiment
def run(n_per_condition, seed):
    images, _ = load_mnist(train=False)
    rng = np.random.default_rng(seed)
    methods = NETWORKS + [REFERENCE]
    per = {m: {c: [] for c in CONDITIONS} for m in methods}           # instance stats per image
    conf = {m: {c: np.zeros((3, 3), np.int64) for c in CONDITIONS} for m in NETWORKS}
    examples = {}
    # load every network and run it once, so loading time is not counted as speed
    for m in NETWORKS:
        nn_segment.run(np.pad(np.full((20, 14), 255, np.uint8), 20), m)

    for cond, kw in CONDITIONS.items():
        print(f"Condition: {cond}")
        for i in range(n_per_condition):
            n = int(rng.integers(2, 7))
            idx = rng.integers(0, len(images), n)
            img, owner = build_number([images[j] for j in idx], rng=rng, **kw)
            binary = seg.prepare_binary(img)
            gt_ids = digit_owner_map(binary, owner)
            gt_cls = make_labels(binary, owner, n)
            if i == 0:
                examples[cond] = (img, binary, gt_cls, {})
            for m in methods:
                t0 = time.perf_counter()
                if m == REFERENCE:
                    segs = seg.segment_hybrid(binary)
                    cls = None
                else:
                    segs, cls = nn_segment.run(binary, m)
                elapsed = time.perf_counter() - t0
                segs.sort(key=lambda s: s.box[0])
                st = instance_stats(segs, gt_ids, n)
                st["number_ok"] = int(score(segs, binary, owner, n)[0])
                st["seconds"] = elapsed
                per[m][cond].append(st)
                if cls is not None:
                    conf[m][cond] += np.bincount(gt_cls.ravel() * 3 + cls.ravel(), minlength=9).reshape(3, 3)
                    if i == 0:
                        examples[cond][3][m] = cls
    return per, conf, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="test numbers per condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots-only", action="store_true", help="redraw the charts from the saved CSV files")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.plots_only:
        overall = list(csv.DictReader(open(os.path.join(RESULTS, "model_comparison_overall.csv"))))
        by_cond = list(csv.DictReader(open(os.path.join(RESULTS, "model_comparison_by_condition.csv"))))
        for r in overall:
            for k, v in r.items():
                if k != "model":
                    r[k] = float(v) if v not in ("", None) else float("nan")
        plot_overview(overall)
        plot_by_condition(by_cond)
        plot_efficiency(overall)
        plot_training()
        print("Charts redrawn.")
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))

    per, conf, examples = run(args.n, args.seed)
    methods = NETWORKS + [REFERENCE]

    # ---------------- overall table
    overall, by_cond, per_class = [], [], []
    for m in methods:
        allstats = [s for c in CONDITIONS for s in per[m][c]]
        row = {"model": LABELS[m]}
        if m in NETWORKS:
            pm = pixel_metrics(sum(conf[m].values()))
            row.update({"pixel_acc": 100 * pm["pixel_acc"], "mean_class_acc": 100 * pm["mean_class_acc"],
                        "mIoU": 100 * pm["miou"], "fwIoU": 100 * pm["fwiou"], "mean_dice": 100 * pm["mdice"],
                        "IoU_inside": 100 * pm["iou"][1], "IoU_border": 100 * pm["iou"][2],
                        "border_precision": 100 * pm["precision"][2], "border_recall": 100 * pm["recall"][2],
                        "dice_border": 100 * pm["dice"][2]})
            for k, cname in enumerate(CLASS_NAMES):
                per_class.append({"model": LABELS[m], "class": cname, "IoU": 100 * pm["iou"][k],
                                  "Dice_F1": 100 * pm["dice"][k], "precision": 100 * pm["precision"][k],
                                  "recall": 100 * pm["recall"][k]})
        row.update(summarise(allstats))
        if m in NETWORKS:
            params, size_mb, gflops = model_cost(m)
            mins, _ = training_minutes(m)
            row.update({"parameters": params, "size_MB": size_mb, "GFLOPs": gflops, "train_minutes": mins})
        overall.append(row)
        for c in CONDITIONS:
            r = {"model": LABELS[m], "condition": c}
            r.update(summarise(per[m][c]))
            if m in NETWORKS:
                pm = pixel_metrics(conf[m][c])
                r.update({"mIoU": 100 * pm["miou"], "IoU_border": 100 * pm["iou"][2]})
            by_cond.append(r)

    def write_csv(rows, name):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        print("Saved results/" + name)

    write_csv(overall, "model_comparison_overall.csv")
    write_csv(by_cond, "model_comparison_by_condition.csv")
    write_csv(per_class, "pixel_metrics_per_class.csv")
    write_markdown(overall, by_cond)
    plot_overview(overall)
    plot_by_condition(by_cond)
    plot_efficiency(overall)
    plot_training()
    plot_examples(examples)
    print_summary(overall)


# --------------------------------------------------------------------------- reporting
def fmt(v, d=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def write_markdown(overall, by_cond):
    nets = [r for r in overall if r["model"] != LABELS[REFERENCE]]
    lines = ["# Task 2 - Comparison of segmentation networks", "",
             "Test set: 1,500 numbers (2-6 digits) built from MNIST test digits, 300 per condition.", "",
             "## 1. Pixel-level metrics (%)", "",
             "| Model | Pixel acc. | Mean class acc. | mIoU | fwIoU | Mean Dice | IoU inside | IoU border | Border precision | Border recall | Border Dice |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in nets:
        lines.append(f"| {r['model']} | {fmt(r['pixel_acc'])} | {fmt(r['mean_class_acc'])} | {fmt(r['mIoU'])} | {fmt(r['fwIoU'])} | "
                     f"{fmt(r['mean_dice'])} | {fmt(r['IoU_inside'])} | {fmt(r['IoU_border'])} | {fmt(r['border_precision'])} | "
                     f"{fmt(r['border_recall'])} | {fmt(r['dice_border'])} |")
    lines += ["", "## 2. Instance-level and task-level metrics (%)", "",
              "| Model | Digit precision | Digit recall | Digit F1 (RQ) | SQ | PQ | Boundary F1 | Count acc. | Over-seg. | Under-seg. | Number acc. (95% CI) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in overall:
        lines.append(f"| {r['model']} | {fmt(r['digit_precision'])} | {fmt(r['digit_recall'])} | {fmt(r['digit_f1_RQ'])} | "
                     f"{fmt(r['SQ'])} | {fmt(r['PQ'])} | {fmt(r['boundary_F1'])} | {fmt(r['count_accuracy'])} | {fmt(r['over_segmentation'])} | "
                     f"{fmt(r['under_segmentation'])} | {fmt(r['number_accuracy'])} ({fmt(r['number_acc_ci_low'])}-{fmt(r['number_acc_ci_high'])}) |")
    lines += ["", "## 3. Efficiency", "",
              "| Model | Parameters | File size (MB) | GFLOPs per image | CPU ms per image | Training time (min) |",
              "|---|---|---|---|---|---|"]
    for r in nets:
        lines.append(f"| {r['model']} | {r['parameters']:,} | {fmt(r['size_MB'], 2)} | {fmt(r['GFLOPs'], 3)} | "
                     f"{fmt(r['ms_per_image'])} | {fmt(r['train_minutes'])} |")
    lines += ["", "## 4. Robustness: PQ and number accuracy per condition (%)", "",
              "| Model | " + " | ".join(f"{c} PQ / num." for c in CONDITIONS) + " |",
              "|---|" + "---|" * len(CONDITIONS)]
    for r in overall:
        cells = []
        for c in CONDITIONS:
            rc = next(x for x in by_cond if x["model"] == r["model"] and x["condition"] == c)
            cells.append(f"{fmt(rc['PQ'])} / {fmt(rc['number_accuracy'])}")
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    open(os.path.join(RESULTS, "model_comparison.md"), "w").write("\n".join(lines) + "\n")
    print("Saved results/model_comparison.md")


def _style(ax):
    ax.yaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_overview(overall):
    metrics = [("IoU_border", "Border IoU"), ("mIoU", "mIoU"), ("digit_f1_RQ", "Digit F1"),
               ("PQ", "Panoptic Quality"), ("number_accuracy", "Number accuracy")]
    methods = NETWORKS + [REFERENCE]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, m in enumerate(methods):
        r = next(x for x in overall if x["model"] == LABELS[m])
        vals = [r.get(key, float("nan")) for key, _ in metrics]
        xs = np.arange(len(metrics)) + (k - (len(methods) - 1) / 2) * width
        ax.bar(xs, [0 if math.isnan(v) else v for v in vals], width * 0.92, color=COLOURS[m], label=LABELS[m])
    ax.set_xticks(np.arange(len(metrics)), [t for _, t in metrics])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Segmentation networks compared on pixel, digit and number level", loc="left")
    _style(ax)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    ax.text(0.99, 0.98, "Pixel metrics are not defined for the rule-based reference", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_metric_overview.png"), dpi=150)
    plt.close(fig)


def plot_by_condition(by_cond):
    methods = NETWORKS + [REFERENCE]
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.8 / len(methods)
    for ax, key, title in [(axes[0], "PQ", "Panoptic Quality by condition"),
                           (axes[1], "number_accuracy", "Number accuracy by condition")]:
        for k, m in enumerate(methods):
            vals = [float(next(x for x in by_cond if x["model"] == LABELS[m] and x["condition"] == c)[key]) for c in conds]
            xs = np.arange(len(conds)) + (k - (len(methods) - 1) / 2) * width
            ax.bar(xs, vals, width * 0.92, color=COLOURS[m], label=LABELS[m])
        ax.set_xticks(np.arange(len(conds)), conds, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 105)
        _style(ax)
    axes[0].set_ylabel("Score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(methods), loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(RESULTS, "fig_by_condition.png"), dpi=150)
    plt.close(fig)


def plot_efficiency(overall):
    """Four small panels: size and speed of each network (lower is better)."""
    rows = [next(x for x in overall if x["model"] == LABELS[m]) for m in NETWORKS]
    panels = [("parameters", "Parameters (millions)", 1e-6, "{:.2f}"),
              ("GFLOPs", "GFLOPs per 64x192 image", 1, "{:.2f}"),
              ("ms_per_image", "CPU time per image (ms)", 1, "{:.1f}"),
              ("train_minutes", "Training time (minutes)", 1, "{:.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    names = [LABELS[m].split(" (")[0] for m in NETWORKS]
    for ax, (key, title, scale, f) in zip(axes, panels):
        vals = [float(r[key]) * scale for r in rows]
        bars = ax.bar(range(len(vals)), vals, 0.7, color=[COLOURS[m] for m in NETWORKS])
        for bar, v in zip(bars, vals):
            ax.annotate(f.format(v), (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5, color="#333333")
        ax.set_xticks(range(len(vals)), names, fontsize=8.5, rotation=20)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
        _style(ax)
    fig.suptitle("Cost of each network (lower is better)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_efficiency.png"), dpi=150)
    plt.close(fig)


def plot_training():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    plotted = False
    for m in NETWORKS:
        _, rows = training_minutes(m)
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        a1.plot(ep, [float(r["val_loss"]) for r in rows], color=COLOURS[m], lw=2, marker="o", ms=3.5, label=LABELS[m])
        plotted = True
        a2.plot(ep, [100 * float(r["val_border_recall"]) for r in rows], color=COLOURS[m], lw=2, marker="o", ms=3.5)
    a1.set_title("Validation loss (lower is better)", loc="left"); a1.set_xlabel("Epoch"); a1.set_yscale("log")
    a2.set_title("Validation border recall (%)", loc="left"); a2.set_xlabel("Epoch"); a2.set_ylim(0, 100)
    for a in (a1, a2):
        _style(a)
    if plotted:
        a1.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_examples(examples):
    palette = np.array([[0, 0, 0], [42, 120, 214], [235, 104, 52]], np.uint8)   # RGB: bg, inside, border
    conds = [c for c in ("Close", "Touching", "Noisy") if c in examples]
    cols = ["Input", "Ground truth"] + [LABELS[m] for m in NETWORKS]
    fig, axes = plt.subplots(len(conds), len(cols), figsize=(2.6 * len(cols), 1.6 * len(conds) + 0.6))
    for r, c in enumerate(conds):
        img, binary, gt, preds = examples[c]
        pics = [binary, palette[gt]] + [palette[preds[m]] for m in NETWORKS]
        for k, p in enumerate(pics):
            ax = axes[r, k]
            ax.imshow(p, cmap="gray" if p.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[k], fontsize=9)
            if k == 0:
                ax.set_ylabel(c, fontsize=9)
    fig.suptitle("Pixel labels: blue = inside a digit, orange = border between digits", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_pixel_examples.png"), dpi=140)
    plt.close(fig)


def print_summary(overall):
    print("\n" + "-" * 104)
    print(f"{'Model':30s} {'mIoU':>6s} {'bIoU':>6s} {'DigitF1':>8s} {'PQ':>6s} {'Number':>7s} {'Params':>11s} {'ms/img':>7s}")
    for r in overall:
        print(f"{r['model']:30s} {fmt(r.get('mIoU')):>6s} {fmt(r.get('IoU_border')):>6s} {fmt(r['digit_f1_RQ']):>8s} "
              f"{fmt(r['PQ']):>6s} {fmt(r['number_accuracy']):>7s} "
              f"{(format(r['parameters'], ',') if 'parameters' in r else '-'):>11s} {fmt(r['ms_per_image']):>7s}")


if __name__ == "__main__":
    main()
