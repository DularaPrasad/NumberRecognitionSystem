"""
demo_segment.py  -  try Task 2 segmentation on ONE image and see the result
===========================================================================
EXAMPLES (run inside the project folder with the venv switched on)

  Build a number from the digit folder and segment it:
      python demo_segment.py --number 4071

  Make the digits touch (the hard case):
      python demo_segment.py --number 4071 --gap -3

  Use your OWN image (e.g. a photo of a number you wrote on paper):
      python demo_segment.py --image my_number.jpg --threshold adaptive

  Compare all methods on the same image:
      python demo_segment.py --number 4071 --all

  Also open the picture in a window:
      python demo_segment.py --number 4071 --show

OUTPUT (in outputs/)
  segmented_<name>.png       picture: original -> black/white -> boxes -> 28x28 digits
  digits_<name>/digit_1.png  each digit in MNIST format, ready for the Task 3 model
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np

from number_builder import DIGIT_FOLDER, OUTPUT_FOLDER, build_number_from_folder, export_digit_folder
from segmentation import METHODS, NEURAL_NETWORKS, draw_boxes, segment_number


def show_result(image, method, threshold, title):
    """Run one method and return (figure, segments)."""
    import matplotlib.pyplot as plt

    segments, binary = segment_number(image, method=method, threshold=threshold, return_binary=True)
    n = max(1, len(segments))
    panels = [
        ("1. Input image", image),
        ("2. Black & white (binary) - ink is white", binary),
    ]
    if method in NEURAL_NETWORKS:
        # show what the network decided for every pixel
        from nn_segment import pixel_labels
        colours = np.array([[0, 0, 0], [214, 120, 42], [52, 104, 235]], np.uint8)  # BGR: black, blue, orange
        panels.append((f"{method} pixel labels: blue = inside a digit, orange = border between digits",
                       colours[pixel_labels(binary, method)]))
    panels.append((f"{len(panels) + 1}. Found {len(segments)} digit(s) with '{method}' (numbered left to right)",
                   draw_boxes(image, segments)))
    rows = len(panels) + 1
    fig = plt.figure(figsize=(max(8, 1.3 * n), 1.9 * rows))
    grid = fig.add_gridspec(rows, n)

    for row, (label, pic) in enumerate(panels):
        ax = fig.add_subplot(grid[row, :])
        if pic.ndim == 3:
            ax.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(pic, cmap="gray")
        ax.set_title(label, loc="left", fontsize=10)
        ax.axis("off")

    for i, seg in enumerate(segments):
        ax = fig.add_subplot(grid[rows - 1, i])
        ax.imshow(seg.digit28, cmap="gray")
        ax.set_title(f"#{i + 1}  (28x28)", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, segments


def main():
    ap = argparse.ArgumentParser(description="Segment a handwritten number image")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image file")
    src.add_argument("--number", help="build this number from the digit folder, e.g. 4071")
    ap.add_argument("--method", default="unet", choices=list(METHODS),
                    help="a trained network (unet, resnet, mobilenet, fcdensenet) or a rule-based method")
    ap.add_argument("--all", action="store_true", help="run all methods")
    ap.add_argument("--threshold", default="otsu", choices=["otsu", "adaptive"])
    ap.add_argument("--gap", type=int, default=None, help="with --number: gap in px (negative = touching)")
    ap.add_argument("--paper", action="store_true", help="with --number: paper/photo look")
    ap.add_argument("--seed", type=int, default=None, help="with --number: random seed")
    ap.add_argument("--show", action="store_true", help="open the result in a window")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # just save files, no window
    import matplotlib.pyplot as plt

    # ---- 1. get the input image -------------------------------------------
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        name = os.path.splitext(os.path.basename(args.image))[0]
    else:
        if not os.path.isdir(DIGIT_FOLDER):
            export_digit_folder()
        kw = {"paper": args.paper}
        if args.gap is not None:
            kw["gap"] = (args.gap, args.gap)
        image = build_number_from_folder(args.number, rng=np.random.default_rng(args.seed), **kw)
        name = args.number

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    methods = list(METHODS) if args.all else [args.method]

    # ---- 2. segment with each chosen method --------------------------------
    for method in methods:
        fig, segments = show_result(image, method, args.threshold, f"Task 2 segmentation: {name}")
        tag = f"{name}_{method}" if args.all else name
        fig_path = os.path.join(OUTPUT_FOLDER, f"segmented_{tag}.png")
        fig.savefig(fig_path, dpi=120)
        print(f"[{method}] found {len(segments)} digit(s) -> {fig_path}")

        # ---- 3. save each digit as a 28x28 image for Task 3 ----------------
        digit_dir = os.path.join(OUTPUT_FOLDER, f"digits_{tag}")
        os.makedirs(digit_dir, exist_ok=True)
        for old in os.listdir(digit_dir):  # clear results from a previous run
            os.remove(os.path.join(digit_dir, old))
        for i, seg in enumerate(segments):
            cv2.imwrite(os.path.join(digit_dir, f"digit_{i + 1}.png"), seg.digit28)
        print(f"          28x28 digits saved in {digit_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
