"""
unet_data.py  -  TRAINING DATA for the U-Net segmentation network
=================================================================
A neural network learns from EXAMPLES WITH ANSWERS. For segmentation, the
"answer" is a label for every single pixel of the image:

      0 = background            (no ink, or a speck of noise)
      1 = inside a digit        (ink that clearly belongs to one digit)
      2 = border between digits (ink where two digits touch or overlap)

We do not have to draw or label anything by hand. number_builder.build_number()
glues MNIST digits together AND tells us which digit owns every pixel, so we
can work out the correct label for every pixel automatically.

IMPORTANT - no cheating:
  * training images are built from MNIST TRAIN digits (60,000 images),
  * compare_methods.py tests on numbers built from MNIST TEST digits,
  so the network is always tested on handwriting it has never seen.

Run this file to see a few examples:   python unet_data.py
"""

import os

import cv2
import numpy as np
from scipy import ndimage

from mnist_loader import load_mnist
from number_builder import build_number
from segmentation import prepare_binary

# Size of every training image (height x width). Both divisible by 8,
# because the U-Net halves the image size three times.
TRAIN_H, TRAIN_W = 64, 192

# The kinds of number images the network should learn to handle.
# Touching digits get the biggest share, because that is where the
# rule-based methods fail.
GAP_CHOICES = [(4, 10), (0, 3), (-5, -1)]   # spaced, close, touching
GAP_PROBS = [0.3, 0.3, 0.4]

BACKGROUND, INTERIOR, BORDER = 0, 1, 2
BORDER_WIDTH = 2   # how many pixels on each side of a contact count as "border"


def make_labels(binary, owner, n_digits):
    """Turn the 'which digit owns this pixel' map into 0/1/2 pixel labels.

    binary : black/white image the network will see (ink = 255)
    owner  : owner[y, x] = digit number (0, 1, 2, ...) or -1 for background
    """
    ink = binary > 0

    # Some ink pixels (soft edges of the pen stroke) have no owner. Give each
    # one the owner of the nearest owned pixel, if that pixel is very close.
    # Ink far from every digit is noise and stays background.
    unowned = owner < 0
    if unowned.all():
        return np.zeros(binary.shape, np.uint8)
    dist, (iy, ix) = ndimage.distance_transform_edt(unowned, return_indices=True)
    full_owner = owner[iy, ix]
    full_owner[~ink | (dist > 2)] = -1

    # A pixel is "border" if it belongs to one digit but is within
    # BORDER_WIDTH pixels of another digit.
    kernel = np.ones((3, 3), np.uint8)
    near_count = np.zeros(binary.shape, np.uint8)
    for k in range(n_digits):
        mask = (full_owner == k).astype(np.uint8)
        near_count += cv2.dilate(mask, kernel, iterations=BORDER_WIDTH) > 0

    labels = np.zeros(binary.shape, np.uint8)
    belongs = full_owner >= 0
    labels[belongs] = INTERIOR
    labels[belongs & (near_count >= 2)] = BORDER
    return labels


def fit_to_canvas(img, rng, h=TRAIN_H, w=TRAIN_W):
    """Place an image on a fixed-size black canvas (crop if too big).
    Returns the canvas and the (top, left) offset used, so labels can follow."""
    ih, iw = img.shape
    top = int(rng.integers(0, h - ih + 1)) if ih <= h else -int(rng.integers(0, ih - h + 1))
    left = int(rng.integers(0, w - iw + 1)) if iw <= w else -int(rng.integers(0, iw - w + 1))
    return top, left


def paste(img, top, left, h=TRAIN_H, w=TRAIN_W):
    out = np.zeros((h, w), img.dtype)
    ys, xs = max(0, top), max(0, left)
    sy, sx = max(0, -top), max(0, -left)
    hh = min(h - ys, img.shape[0] - sy)
    ww = min(w - xs, img.shape[1] - sx)
    out[ys:ys + hh, xs:xs + ww] = img[sy:sy + hh, sx:sx + ww]
    return out


def make_sample(images, rng):
    """Make ONE training example: (binary input image, pixel labels)."""
    n = int(rng.integers(2, 7))                      # 2 to 6 digits
    picks = [images[i] for i in rng.integers(0, len(images), n)]
    gap = GAP_CHOICES[rng.choice(len(GAP_CHOICES), p=GAP_PROBS)]
    specks = int(rng.integers(0, 26)) if rng.random() < 0.3 else 0
    paper = rng.random() < 0.25

    img, owner = build_number(picks, gap=gap, specks=specks, paper=paper, rng=rng)
    binary = prepare_binary(img)
    labels = make_labels(binary, owner, n)

    top, left = fit_to_canvas(binary, rng)
    return paste(binary, top, left), paste(labels, top, left)


def make_batch(images, rng, size):
    """Make a batch of examples as NumPy arrays:
    x: (size, 1, H, W) float32 in 0..1     y: (size, H, W) int64 labels"""
    xs, ys = zip(*(make_sample(images, rng) for _ in range(size)))
    x = np.stack(xs)[:, None].astype(np.float32) / 255.0
    y = np.stack(ys).astype(np.int64)
    return x, y


if __name__ == "__main__":
    # Save a picture of 6 training examples and their labels.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    images, _ = load_mnist(train=True)
    rng = np.random.default_rng(0)
    cmap = ListedColormap(["#000000", "#2a78d6", "#eb6834"])  # bg, interior, border
    fig, axes = plt.subplots(6, 2, figsize=(10, 9))
    for r in range(6):
        x, y = make_sample(images, rng)
        axes[r, 0].imshow(x, cmap="gray")
        axes[r, 1].imshow(y, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        for a in axes[r]:
            a.axis("off")
    axes[0, 0].set_title("Input the network sees")
    axes[0, 1].set_title("Correct answer: blue = inside digit, orange = border")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/unet_training_examples.png", dpi=110)
    print("Saved results/unet_training_examples.png")
