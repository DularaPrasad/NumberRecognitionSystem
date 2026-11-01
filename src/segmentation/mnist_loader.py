"""
mnist_loader.py
================
Loads MNIST digit images as plain NumPy arrays.

WHAT IS MNIST?
    70,000 small pictures (28 x 28 pixels) of handwritten digits 0-9.
    Each picture is white ink on a black background, and comes with a
    "label" saying which digit it is.
        - 60,000 "train" images (used to teach a model - that's Task 3)
        - 10,000 "test"  images (used to check a model)

HOW THIS FILE WORKS
    1. It first looks for the original MNIST files in  data/MNIST/raw/  next
       to this file, then in the group repository's top-level data/MNIST/raw/
       (where torchvision and the other tasks save them).
    2. If they are missing, it asks torchvision to download them.
    3. It reads the files and returns  (images, labels)  as NumPy arrays.

USAGE
    from mnist_loader import load_mnist
    images, labels = load_mnist(train=False)   # 10,000 test digits
    print(images.shape)   # (10000, 28, 28)
    print(labels[:10])    # e.g. [7 2 1 0 4 1 4 9 5 9]
"""

import gzip
import os

import numpy as np

# Folder where MNIST is stored (relative to this file, so it works no matter
# which folder you run the script from).
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "MNIST", "raw")


def _read_idx(path):
    """Read one MNIST file (the 'idx' format) into a NumPy array.

    The idx format is simple: a small header that says how big the data is,
    followed by the raw pixel (or label) bytes.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()

    # Byte 3 tells us how many dimensions the data has (1 = labels, 3 = images)
    n_dims = data[3]
    # Each dimension size is stored as a 4-byte big-endian integer
    shape = tuple(
        int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(n_dims)
    )
    offset = 4 + 4 * n_dims
    return np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(shape)


# In the group repository this file is in src/segmentation/, and the other
# tasks (and torchvision) keep MNIST in the repository's top-level data/ folder.
# Look there too, so MNIST is not downloaded twice.
REPO_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "MNIST", "raw")


def _find_file(stem):
    """Return the path of e.g. 't10k-images-idx3-ubyte' (zipped or not)."""
    for folder in (RAW_DIR, REPO_RAW_DIR):
        for name in (stem, stem + ".gz"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    return None


def load_mnist(train=False):
    """Load MNIST and return (images, labels).

    Args:
        train: True  -> 60,000 training images
               False -> 10,000 test images (default; enough for Task 2)

    Returns:
        images: uint8 array, shape (N, 28, 28), values 0 (black) .. 255 (white)
        labels: uint8 array, shape (N,), values 0..9
    """
    prefix = "train" if train else "t10k"
    img_path = _find_file(f"{prefix}-images-idx3-ubyte")
    lbl_path = _find_file(f"{prefix}-labels-idx1-ubyte")

    # Files missing? Let torchvision download them into data/MNIST/raw/
    if img_path is None or lbl_path is None:
        print("MNIST not found locally - downloading with torchvision ...")
        from torchvision import datasets  # imported here so it's only needed once

        datasets.MNIST(DATA_DIR, train=train, download=True)
        img_path = _find_file(f"{prefix}-images-idx3-ubyte")
        lbl_path = _find_file(f"{prefix}-labels-idx1-ubyte")

    return _read_idx(img_path), _read_idx(lbl_path)


# Run this file directly to check it works:  python mnist_loader.py
if __name__ == "__main__":
    imgs, lbls = load_mnist(train=False)
    print("Images:", imgs.shape, imgs.dtype)
    print("Labels:", lbls.shape, "first ten =", lbls[:10])
