"""
evaluate_pipeline.py

Task 4: evaluation of the whole system (preprocessing, segmentation, then a digit
model) on images of multi-digit numbers, for every trained model.

Two sources of test images (use either or both):

  --synthetic N   N numbers of 2 to 5 digits, built from MNIST TEST digits, which none
                  of the models were trained on. Half are clean; half have a shadow
                  across the image and some noise (a harder, more photo-like case).
  --photos DIR    real photos. The true number is the digits at the START of the file
                  name, for example 4071_a.jpg or 38-photo2.png. Files whose name does
                  not start with a number are skipped.

For every model and preprocessing setting it reports three numbers:

  number accuracy  the whole number was read exactly right
  count accuracy   segmentation found the right NUMBER OF DIGITS
  digit accuracy   digits read correctly, among numbers whose digit count was right

The last two separate segmentation errors from classification errors.

Examples (run from anywhere):
    python src\\evaluation\\evaluate_pipeline.py --synthetic 200
    python src\\evaluation\\evaluate_pipeline.py --photos TestImages
    python src\\evaluation\\evaluate_pipeline.py --synthetic 200 --photos TestImages --save-failures

Results go to results/evaluation/: a CSV of every image, a summary CSV, and (with
--save-failures) a picture of every number that was read wrongly.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "integration"))

from pipeline import MODEL_NAMES, SEGMENTERS, NumberRecognizer, annotate  # noqa: E402

OUT = ROOT / "results" / "evaluation"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    name: str            # file name, or a generated name
    image: np.ndarray    # BGR image
    truth: str           # the true number, for example "4071"
    condition: str       # "clean" or "shadow" (synthetic), or "photo"


# ------------------------------------------------------------ test images
def make_number_image(digits, pool, rng, shadow):
    """Join real MNIST test digits into one number, drawn as dark ink on light paper."""
    parts = []
    for ch in digits:
        candidates = pool[int(ch)]                                         # all MNIST test images of this digit
        parts.append(candidates[rng.integers(len(candidates))])            # pick one at random
        parts.append(np.zeros((28, int(rng.integers(3, 9))), np.uint8))    # a gap of 3 to 8 pixels
    strip = np.pad(np.hstack(parts[:-1]), 14)                              # drop the last gap, add a blank margin
    big = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = np.repeat((255 - big)[:, :, None], 3, axis=2).astype(np.float32)  # white-on-black -> dark ink on paper
    if shadow:
        ramp = np.linspace(rng.uniform(0.5, 0.85), 1.0, img.shape[1])      # one side of the paper is darker
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        img = img * ramp[None, :, None] + rng.normal(0, 5, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def build_synthetic(pool, n, seed):
    """n numbers: lengths cycle 2,3,4,5 and the first half of every 8 are clean, the rest shadowed."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        length = 2 + i % 4
        shadow = (i // 4) % 2 == 1
        digits = "".join(str(int(d)) for d in rng.integers(0, 10, length))
        samples.append(Sample(f"synthetic_{i:03d}_{digits}", make_number_image(digits, pool, rng, shadow),
                              digits, "shadow" if shadow else "clean"))
    return samples


def load_mnist_test_pool():
    from torchvision import datasets
    data = datasets.MNIST(root=ROOT / "data", train=False, download=True)
    images, labels = data.data.numpy(), data.targets.numpy()
    return {d: images[labels == d] for d in range(10)}


def load_photos(folder):
    samples, skipped = [], []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        match = re.match(r"(\d+)", path.stem)
        if not match:
            skipped.append(f"{path.name} (file name does not start with a number)")
            continue
        image = cv2.imread(str(path))
        if image is None:
            skipped.append(f"{path.name} (cannot be read)")
            continue
        samples.append(Sample(path.name, image, match.group(1), "photo"))
    return samples, skipped


# ------------------------------------------------------------ running and scoring
def score(truth, predicted):
    count_ok = len(predicted) == len(truth)
    digits_right = sum(a == b for a, b in zip(truth, predicted)) if count_ok else 0
    return predicted == truth, count_ok, digits_right


def run(samples, models, preprocessing_options, segmenter, save_failures=False, factory=NumberRecognizer):
    rows, errors = [], 0
    for prep in preprocessing_options:
        recognizer = factory(models[0], segmenter, prep)
        for model in models:
            recognizer.set_model(model)
            for s in samples:
                result, predicted, error = None, "", ""
                try:
                    result = recognizer.recognise(s.image)
                    predicted = result.number
                except Exception as problem:       # one bad image must not stop the whole run
                    error = f"{type(problem).__name__}: {problem}"
                    errors += 1
                number_ok, count_ok, digits_right = score(s.truth, predicted)
                rows.append({"condition": s.condition, "file": s.name, "truth": s.truth, "preprocessing": prep,
                             "model": model, "predicted": predicted, "number_correct": int(number_ok),
                             "count_correct": int(count_ok), "digits_right": digits_right, "error": error})
                if save_failures and not number_ok and result is not None:
                    folder = OUT / "failures"
                    folder.mkdir(parents=True, exist_ok=True)
                    shown = annotate(s.image, result) if result.boxes else s.image
                    cv2.imwrite(str(folder / f"{Path(s.name).stem}_{model}_{prep}.png"), shown)
    return rows, errors


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["preprocessing"], r["model"]), []).append(r)
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    summary = []
    for (condition, prep, model), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 9))):
        n = len(group)
        counted = [r for r in group if r["count_correct"]]
        digits = sum(len(r["truth"]) for r in counted)
        summary.append({
            "condition": condition, "preprocessing": prep, "model": model, "numbers": n,
            "number_accuracy": sum(r["number_correct"] for r in group) / n,
            "count_accuracy": len(counted) / n,
            "digit_accuracy": (sum(r["digits_right"] for r in counted) / digits) if digits else float("nan"),
        })
    return summary


def print_summary(summary):
    print(f"\n{'condition':10s} {'preprocessing':14s} {'model':8s} {'numbers':>8s} {'number acc':>11s} {'count acc':>10s} {'digit acc':>10s}")
    previous = None
    for s in summary:
        if previous is not None and (s["condition"], s["preprocessing"]) != previous:
            print()
        previous = (s["condition"], s["preprocessing"])
        digit = "n/a" if np.isnan(s["digit_accuracy"]) else f"{s['digit_accuracy']:.1%}"
        print(f"{s['condition']:10s} {s['preprocessing']:14s} {s['model']:8s} {s['numbers']:8d} "
              f"{s['number_accuracy']:11.1%} {s['count_accuracy']:10.1%} {digit:>10s}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the whole number-recognition pipeline.")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N", help="number of numbers built from MNIST test digits")
    parser.add_argument("--photos", metavar="DIR", help="folder of real photos named like 4071_a.jpg")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--segmenter", default="unet", choices=SEGMENTERS)
    parser.add_argument("--preprocessing", default="both", choices=("adaptive", "otsu", "both"))
    parser.add_argument("--seed", type=int, default=42, help="seed for the synthetic numbers (same numbers every run)")
    parser.add_argument("--save-failures", action="store_true", help="save a picture of every wrongly read number")
    args = parser.parse_args()
    if not args.synthetic and not args.photos:
        parser.error("give --synthetic N, --photos DIR, or both")

    samples = []
    if args.synthetic:
        samples += build_synthetic(load_mnist_test_pool(), args.synthetic, args.seed)
    if args.photos:
        photos, skipped = load_photos(args.photos)
        samples += photos
        print(f"Photos: {len(photos)} used from {args.photos}")
        for item in skipped:
            print(f"  skipped: {item}")
    if not samples:
        sys.exit("No test images to evaluate.")

    preps = ["otsu", "adaptive"] if args.preprocessing == "both" else [args.preprocessing]
    print(f"Evaluating {len(samples)} images, models {args.models}, segmenter '{args.segmenter}', preprocessing {preps}")
    rows, errors = run(samples, args.models, preps, args.segmenter, args.save_failures)
    summary = summarise(rows)
    print_summary(summary)
    if errors:
        print(f"\n{errors} recognitions raised an error and were counted as wrong (see the 'error' column in the CSV).")

    write_csv(OUT / "pipeline_results.csv", rows)
    write_csv(OUT / "pipeline_summary.csv", summary)
    print(f"\nSaved {OUT / 'pipeline_summary.csv'} and {OUT / 'pipeline_results.csv'}")


if __name__ == "__main__":
    main()
