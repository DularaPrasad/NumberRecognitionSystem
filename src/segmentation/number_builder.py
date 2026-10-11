"""
number_builder.py  -  make images of multi-digit NUMBERS from single digits
==========================================================================
The brief says image acquisition must allow "the automatic creation of the
image of a number from a folder of images of individual digits".
This file does exactly that, and we ALSO use it to create test images for
Task 2, because we know the correct answer for every image we build
(which digit is where) - so we can measure how good segmentation is.

WHAT YOU CAN DO
  1. Create a folder of single-digit images from MNIST:
         python number_builder.py --export
     -> digits/0/*.png, digits/1/*.png, ... digits/9/*.png

  2. Build a number image from that folder:
         python number_builder.py 4071
     -> outputs/number_4071.png

  3. Try harder versions (digits touching, photo-like paper background):
         python number_builder.py 4071 --gap -3 --paper
"""

import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIGIT_FOLDER = os.path.join(HERE, "digits")
OUTPUT_FOLDER = os.path.join(HERE, "outputs")


# --------------------------------------------------------------------------
# Part 1: folder of single digits
# --------------------------------------------------------------------------
def export_digit_folder(out_dir=DIGIT_FOLDER, per_class=20, seed=0):
    """Save `per_class` MNIST test images of each digit as PNG files:
    digits/<digit>/<digit>_<n>.png   (white ink on black, 28x28)"""
    from mnist_loader import load_mnist

    images, labels = load_mnist(train=False)
    rng = np.random.default_rng(seed)
    for d in range(10):
        os.makedirs(os.path.join(out_dir, str(d)), exist_ok=True)
        idx = rng.choice(np.where(labels == d)[0], per_class, replace=False)
        for n, i in enumerate(idx):
            cv2.imwrite(os.path.join(out_dir, str(d), f"{d}_{n:03d}.png"), images[i])
    print(f"Saved {per_class} images per digit into {out_dir}")


def load_digit_folder(folder=DIGIT_FOLDER):
    """Read the folder back: returns {digit: [image, image, ...]}.
    Images with a light background (e.g. your own scanned digits on white
    paper) are inverted so every digit is white ink on black."""
    bank = {}
    for d in range(10):
        sub = os.path.join(folder, str(d))
        if not os.path.isdir(sub):
            continue
        for name in sorted(os.listdir(sub)):
            img = cv2.imread(os.path.join(sub, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # not an image file
            if np.median(img) > 127:
                img = 255 - img
            bank.setdefault(d, []).append(img)
    if not bank:
        raise FileNotFoundError(
            f"No digit images in '{folder}'. Run:  python number_builder.py --export"
        )
    return bank


# --------------------------------------------------------------------------
# Part 2: glue digits together into one number image
# --------------------------------------------------------------------------
def _trim(img, level=30):
    """Cut away the empty border around a digit."""
    ys, xs = np.nonzero(img > level)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_number(digit_images, gap=(4, 10), scale=(0.9, 1.2), jitter=3,
                 paper=False, specks=0, margin=12, rng=None):
    """Place digit images side by side to make one number image.

    Args:
        digit_images: list of single-digit images (white ink on black).
        gap:    (min, max) pixels between neighbouring digits. NEGATIVE gaps
                make digits overlap/touch - the hard case for segmentation.
        scale:  random size change per digit (people don't write evenly).
        jitter: random up/down shift per digit, in pixels.
        paper:  True = dark ink on light, unevenly-lit paper (like a photo).
        specks: number of random noise dots to sprinkle on the image.
        rng:    random generator (pass one with a seed for repeatable results).

    Returns:
        image:  the number image (uint8).
        owner:  same size as image; owner[y, x] = which digit (0,1,2...) the
                ink pixel belongs to, or -1 for background. This is the
                "answer key" used to score segmentation.
    """
    rng = rng or np.random.default_rng()
    pieces = []
    for img in digit_images:
        f = rng.uniform(*scale)
        big = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        pieces.append(_trim(big))

    # Work out the canvas size.
    max_h = max(p.shape[0] for p in pieces)
    gaps = [int(rng.integers(gap[0], gap[1] + 1)) for _ in pieces[:-1]]
    total_w = sum(p.shape[1] for p in pieces) + sum(gaps)
    H = max_h + 2 * (margin + jitter)
    W = max(total_w, max(p.shape[1] for p in pieces)) + 2 * margin
    canvas = np.zeros((H, W), np.uint8)
    owner = np.full((H, W), -1, np.int32)

    # Paste each digit, left to right.
    x = margin
    for k, p in enumerate(pieces):
        h, w = p.shape
        y = margin + jitter + (max_h - h) // 2 + int(rng.integers(-jitter, jitter + 1))
        region = canvas[y:y + h, x:x + w]
        wins = p > region                       # where this digit is brighter
        owner[y:y + h, x:x + w][wins & (p > 127)] = k
        np.maximum(region, p, out=region)
        if k < len(gaps):
            x += w + gaps[k]

    # Optional noise dots (dust, pen marks).
    for _ in range(specks):
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        cv2.circle(canvas, (cx, cy), int(rng.integers(0, 2)), 255, -1)

    if paper:
        # Dark-grey ink on light paper, brighter on one side (uneven light),
        # plus grainy sensor noise - similar to a phone photo.
        light = np.linspace(rng.uniform(170, 200), rng.uniform(225, 250), W)[None, :]
        ink_level = canvas.astype(np.float32) / 255.0
        img = light * (1 - 0.75 * ink_level)
        img += rng.normal(0, 6, img.shape)
        canvas = np.clip(img, 0, 255).astype(np.uint8)
    return canvas, owner


def build_number_from_folder(number, folder=DIGIT_FOLDER, rng=None, **kwargs):
    """Build an image of `number` (a string such as "4071") by picking a random
    image of each digit from the digit folder."""
    rng = rng or np.random.default_rng()
    bank = load_digit_folder(folder)
    chosen = []
    for ch in str(number):
        options = bank[int(ch)]
        chosen.append(options[int(rng.integers(len(options)))])
    image, _owner = build_number(chosen, rng=rng, **kwargs)
    return image


# --------------------------------------------------------------------------
# Command-line use
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build number images from digit images")
    ap.add_argument("number", nargs="?", help="number to build, e.g. 4071")
    ap.add_argument("--export", action="store_true", help="create the digits/ folder from MNIST")
    ap.add_argument("--gap", type=int, default=None, help="fixed gap in pixels (negative = touching)")
    ap.add_argument("--paper", action="store_true", help="dark ink on light paper look")
    ap.add_argument("--seed", type=int, default=None, help="random seed (same seed = same image)")
    args = ap.parse_args()

    if args.export:
        export_digit_folder()
    if args.number:
        if not os.path.isdir(DIGIT_FOLDER):
            export_digit_folder()
        kw = {"paper": args.paper}
        if args.gap is not None:
            kw["gap"] = (args.gap, args.gap)
        img = build_number_from_folder(args.number, rng=np.random.default_rng(args.seed), **kw)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        path = os.path.join(OUTPUT_FOLDER, f"number_{args.number}.png")
        cv2.imwrite(path, img)
        print("Saved", path)
    if not args.export and not args.number:
        ap.print_help()
