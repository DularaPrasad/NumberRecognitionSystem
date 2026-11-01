"""
evaluate_pipeline.py

Task 4: evaluation of the whole system (preprocessing, segmentation, then a digit
model) on images of multi-digit numbers, for every trained digit model and, now,
for every segmentation method.

Sources of test images (use any combination):

  --synthetic N   N numbers of 2 to 5 digits, built from MNIST TEST digits, which none
                  of the models were trained on. Half are clean; half have a shadow
                  across the image and some noise. The digits have a 3 to 8 pixel gap.
  --touching N    N more numbers whose digits TOUCH or overlap by 1 to 4 pixels, on
                  clean paper. This is where the segmentation methods differ most.
  --photos DIR    real photos. The true number is the digits at the START of the file
                  name, for example 4071_a.jpg or 38-photo2.png. Files whose name does
                  not start with a number are skipped.

For every segmenter, digit model and preprocessing setting it reports:

  number accuracy  the whole number was read exactly right
  count accuracy   segmentation found the right NUMBER OF DIGITS
  digit accuracy   digits read correctly, among numbers whose digit count was right

The last two separate segmentation errors from recognition errors.

--segmenter all  runs the four segmentation networks (unet, resnet, mobilenet,
fcdensenet) and the rule-based "hybrid" baseline, whichever are available.

Examples (run from anywhere):
    python src\\evaluation\\evaluate_pipeline.py --synthetic 200
    python src\\evaluation\\evaluate_pipeline.py --synthetic 200 --touching 100 --segmenter all --preprocessing adaptive
    python src\\evaluation\\evaluate_pipeline.py --photos TestImages --segmenter all --save-failures

Results go to results/evaluation/: a CSV of every image and a summary CSV (with
"_all" in the file names when several segmenters are compared), and, with
--save-failures, a picture of every number that was read wrongly.
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
NETWORK_SEGMENTERS = ("unet", "resnet", "mobilenet", "fcdensenet")
COMPARED_SEGMENTERS = NETWORK_SEGMENTERS + ("hybrid",)
CONDITION_ORDER = ["clean", "shadow", "touching", "photo"]


@dataclass
class Sample:
    name: str            # file name, or a generated name
    image: np.ndarray    # BGR image
    truth: str           # the true number, for example "4071"
    condition: str       # "clean", "shadow" or "touching" (synthetic), or "photo"


# ------------------------------------------------------------ test images
def make_number_image(digits, pool, rng, shadow, overlap=False):
    """Join real MNIST test digits into one number, drawn as dark ink on light paper.

    overlap=False leaves a gap of 3 to 8 pixels between digits; overlap=True makes
    neighbouring digits overlap by 1 to 4 pixels, so they touch."""
    picks, gaps = [], []
    for ch in digits:
        candidates = pool[int(ch)]                                         # all MNIST test images of this digit
        picks.append(candidates[rng.integers(len(candidates))])            # pick one at random
        gap = int(rng.integers(3, 9)) if not overlap else -int(rng.integers(1, 5))
        gaps.append(gap)
    if overlap:
        # MNIST digits have blank margins, so overlap the INK, not the 28-pixel boxes: crop each
        # digit to its ink columns, then let neighbouring digits overlap by 1 to 4 pixels.
        picks = [image[:, np.where(image.max(axis=0) > 0)[0][0]:np.where(image.max(axis=0) > 0)[0][-1] + 1]
                 for image in picks]
    widths = [image.shape[1] for image in picks]
    strip = np.zeros((28, sum(widths) + sum(gaps[:-1])), np.uint8)
    x = 0
    for image, width, gap in zip(picks, widths, gaps):
        strip[:, x:x + width] = np.maximum(strip[:, x:x + width], image)   # overlapping ink stays ink
        x += width + gap
    strip = np.pad(strip, 14)                                              # add a blank margin
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


def build_touching(pool, n, seed):
    """n numbers on clean paper whose digits touch or overlap. Uses its own random stream, so
    adding this test never changes the numbers that --synthetic produces."""
    rng = np.random.default_rng(seed + 1)
    samples = []
    for i in range(n):
        length = 2 + i % 4
        digits = "".join(str(int(d)) for d in rng.integers(0, 10, length))
        samples.append(Sample(f"touching_{i:03d}_{digits}",
                              make_number_image(digits, pool, rng, shadow=False, overlap=True), digits, "touching"))
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


def resolve_segmenters(choice, available=SEGMENTERS):
    """'all' means the four networks plus the hybrid baseline, whichever this pipeline.py offers."""
    if choice != "all":
        return [choice], ""
    chosen = [s for s in COMPARED_SEGMENTERS if s in available]
    missing = [s for s in COMPARED_SEGMENTERS if s not in available]
    warning = (f"Not available in this pipeline.py: {', '.join(missing)}. "
               "Pull the latest version from GitHub to include them.") if missing else ""
    return chosen, warning


# ------------------------------------------------------------ running and scoring
def score(truth, predicted):
    count_ok = len(predicted) == len(truth)
    digits_right = sum(a == b for a, b in zip(truth, predicted)) if count_ok else 0
    return predicted == truth, count_ok, digits_right


def run(samples, models, preprocessing_options, segmenters, save_failures=False, factory=NumberRecognizer):
    if isinstance(segmenters, str):
        segmenters = [segmenters]
    rows, errors = [], 0
    for seg in segmenters:
        for prep in preprocessing_options:
            recognizer = factory(models[0], seg, prep)
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
                                 "segmenter": seg, "model": model, "predicted": predicted,
                                 "number_correct": int(number_ok), "count_correct": int(count_ok),
                                 "digits_right": digits_right, "error": error})
                    if save_failures and not number_ok and result is not None:
                        folder = OUT / "failures"
                        folder.mkdir(parents=True, exist_ok=True)
                        shown = annotate(s.image, result) if result.boxes else s.image
                        cv2.imwrite(str(folder / f"{Path(s.name).stem}_{seg}_{model}_{prep}.png"), shown)
    return rows, errors


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["preprocessing"], r["segmenter"], r["model"]), []).append(r)

    def position(item, order):
        return order.index(item) if item in order else len(order)

    def sort_key(key):
        condition, prep, seg, model = key
        return (position(condition, CONDITION_ORDER), prep, position(seg, list(COMPARED_SEGMENTERS)),
                position(model, list(MODEL_NAMES)))

    summary = []
    for key in sorted(groups, key=sort_key):
        condition, prep, seg, model = key
        group = groups[key]
        n = len(group)
        counted = [r for r in group if r["count_correct"]]
        digits = sum(len(r["truth"]) for r in counted)
        summary.append({
            "condition": condition, "preprocessing": prep, "segmenter": seg, "model": model, "numbers": n,
            "number_accuracy": sum(r["number_correct"] for r in group) / n,
            "count_accuracy": len(counted) / n,
            "digit_accuracy": (sum(r["digits_right"] for r in counted) / digits) if digits else float("nan"),
        })
    return summary


def print_summary(summary):
    print(f"\n{'condition':10s} {'preprocessing':13s} {'segmenter':11s} {'model':8s} {'numbers':>7s} "
          f"{'number acc':>11s} {'count acc':>10s} {'digit acc':>10s}")
    previous = None
    for s in summary:
        group = (s["condition"], s["preprocessing"], s["segmenter"])
        if previous is not None and group != previous:
            print()
        previous = group
        digit = "n/a" if np.isnan(s["digit_accuracy"]) else f"{s['digit_accuracy']:.1%}"
        print(f"{s['condition']:10s} {s['preprocessing']:13s} {s['segmenter']:11s} {s['model']:8s} {s['numbers']:7d} "
              f"{s['number_accuracy']:11.1%} {s['count_accuracy']:10.1%} {digit:>10s}")


def print_pivot(summary, model):
    """Whole numbers read exactly right, one row per segmenter, for one digit model."""
    cells = {}
    for s in summary:
        if s["model"] == model:
            cells[(s["segmenter"], f"{s['condition']}/{s['preprocessing']}")] = s["number_accuracy"]
    if not cells:
        return

    def column_order(column):
        condition = column.split("/")[0]
        return (CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else 9, column)

    columns = sorted({c for _, c in cells}, key=column_order)
    present = {k[0] for k in cells}
    segmenters = [s for s in COMPARED_SEGMENTERS if s in present] + sorted(present - set(COMPARED_SEGMENTERS))
    print(f"\nWhole numbers read exactly right, with the {model} digit model:")
    print(f"{'segmenter':12s}" + "".join(f"{c:>18s}" for c in columns))
    for seg in segmenters:
        print(f"{seg:12s}" + "".join(f"{cells[(seg, c)]:>18.0%}" if (seg, c) in cells else f"{'':>18s}" for c in columns))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the whole number-recognition pipeline.")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N", help="numbers built from MNIST test digits (gaps between digits)")
    parser.add_argument("--touching", type=int, default=0, metavar="N", help="numbers whose digits touch or overlap")
    parser.add_argument("--photos", metavar="DIR", help="folder of real photos named like 4071_a.jpg")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--segmenter", default="unet", choices=tuple(SEGMENTERS) + ("all",))
    parser.add_argument("--preprocessing", default="both", choices=("adaptive", "otsu", "both"))
    parser.add_argument("--seed", type=int, default=42, help="seed for the synthetic numbers (same numbers every run)")
    parser.add_argument("--pivot-model", default="resnet", choices=MODEL_NAMES, help="digit model shown in the compact table")
    parser.add_argument("--save-failures", action="store_true", help="save a picture of every wrongly read number")
    args = parser.parse_args()
    if not (args.synthetic or args.touching or args.photos):
        parser.error("give --synthetic N, --touching N, --photos DIR, or a combination")

    samples = []
    if args.synthetic or args.touching:
        pool = load_mnist_test_pool()
        if args.synthetic:
            samples += build_synthetic(pool, args.synthetic, args.seed)
        if args.touching:
            samples += build_touching(pool, args.touching, args.seed)
    if args.photos:
        photos, skipped = load_photos(args.photos)
        samples += photos
        print(f"Photos: {len(photos)} used from {args.photos}")
        for item in skipped:
            print(f"  skipped: {item}")
    if not samples:
        sys.exit("No test images to evaluate.")

    segmenters, warning = resolve_segmenters(args.segmenter)
    if warning:
        print(f"Note: {warning}")
    preps = ["otsu", "adaptive"] if args.preprocessing == "both" else [args.preprocessing]
    print(f"Evaluating {len(samples)} images, models {args.models}, segmenters {segmenters}, preprocessing {preps}")
    rows, errors = run(samples, args.models, preps, segmenters, args.save_failures)
    summary = summarise(rows)
    print_summary(summary)
    if args.pivot_model in args.models:
        print_pivot(summary, args.pivot_model)
    if errors:
        print(f"\n{errors} recognitions raised an error and were counted as wrong (see the 'error' column in the CSV).")

    suffix = "_all" if len(segmenters) > 1 else ""
    write_csv(OUT / f"pipeline_results{suffix}.csv", rows)
    write_csv(OUT / f"pipeline_summary{suffix}.csv", summary)
    print(f"\nSaved {OUT / f'pipeline_summary{suffix}.csv'} and {OUT / f'pipeline_results{suffix}.csv'}")


if __name__ == "__main__":
    main()
