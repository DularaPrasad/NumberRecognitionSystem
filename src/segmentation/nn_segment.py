"""
nn_segment.py  -  split a number into digits with ANY trained segmentation network
==================================================================================
Used by segment_number(image, method="unet" | "resnet" | "mobilenet" | "fcdensenet").

Steps (the same for every network, so they are compared fairly):
  1. Size check: the networks were trained on digits about 21 pixels tall.
     Bigger images (e.g. phone photos) are scaled to that size first; thin
     pen lines are thickened so they do not vanish when shrunk.
  2. The network labels every pixel: background / inside a digit / border.
  3. Each separate patch of "inside a digit" pixels is one digit (a seed).
  4. Border pixels go to the nearest seed, so touching digits are cut where
     the network saw the contact.
  5. The result is mapped back to the original size and tiny specks are
     removed (the same light clean-up for every network).

You need the trained model files in models/ (made by train_seg.py).
"""

import os

import cv2
import numpy as np
import torch
from scipy import ndimage

import segmentation as seg
from seg_models import MODEL_NAMES, build_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TARGET_DIGIT_HEIGHT = 21   # typical digit height (pixels) in the training images
MIN_SEED_AREA = 12         # "inside" patches smaller than this are ignored
TARGET_STROKE = 2.5        # pen thickness (pixels) the networks are used to

_loaded = {}  # each network is loaded once and then reused


def model_path(name):
    return os.path.join(MODELS_DIR, f"{name}_seg.pt")


def load_model(name):
    if name not in _loaded:
        path = model_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained model not found: {path}. Run:  python train_seg.py --model {name}")
        net = build_model(name)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        _loaded[name] = net
    return _loaded[name]


def _stroke_width(binary):
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return 4.0 * float(dist[dist > 0].mean()) if dist.any() else 0.0


def _scale_factor(binary):
    blobs = seg.remove_noise(seg.segment_connected_components(binary))
    if not blobs:
        return 1.0
    factor = TARGET_DIGIT_HEIGHT / seg._typical_height(blobs)
    return 1.0 if 0.75 <= factor <= 1.35 else factor


def predict_classes(binary_small, name):
    """Run the network; returns the class (0/1/2) of every pixel."""
    h, w = binary_small.shape
    pad = 8
    H = (h + 2 * pad + 7) // 8 * 8          # sizes divisible by 8
    W = (w + 2 * pad + 7) // 8 * 8
    canvas = np.zeros((H, W), np.float32)
    canvas[pad:pad + h, pad:pad + w] = binary_small / 255.0
    with torch.no_grad():
        scores = load_model(name)(torch.from_numpy(canvas)[None, None])
    return scores.argmax(1)[0].numpy()[pad:pad + h, pad:pad + w].astype(np.uint8)


def _nearest_fill(ids, todo, max_dist):
    if not (ids > 0).any():
        return ids
    dist, (iy, ix) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    fill = todo & (ids == 0) & (dist <= max_dist)
    ids = ids.copy()
    ids[fill] = ids[iy[fill], ix[fill]]
    return ids


def run(binary, name):
    """Segment a binary image (ink = 255) with network `name`.
    Returns (segments, pixel_labels) - pixel_labels is the network's class
    for every pixel of the ORIGINAL image (0 background, 1 inside, 2 border)."""
    if not binary.any():
        return [], np.zeros(binary.shape, np.uint8)

    # 1. bring the digits to the training scale
    factor = _scale_factor(binary)
    work = binary
    if factor != 1.0:
        needed = TARGET_STROKE / factor
        sw = _stroke_width(binary)
        if sw < needed:
            k = int(round(needed - sw)) + 1
            work = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        size = (max(1, round(binary.shape[1] * factor)), max(1, round(binary.shape[0] * factor)))
        work = cv2.resize(work, size, interpolation=cv2.INTER_AREA)
        work = np.where(work > 60, 255, 0).astype(np.uint8)

    # 2. the network labels every pixel
    cls = predict_classes(work, name)
    ink = work > 0

    # 3. each patch of "inside a digit" pixels is one digit seed
    n, seeds, stats, _ = cv2.connectedComponentsWithStats(((cls == 1) & ink).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_SEED_AREA:
            seeds[seeds == i] = 0

    # 4. border (and other ink) pixels go to the nearest seed
    ids = _nearest_fill(seeds, ink, max_dist=6)

    # 5. back to the original size
    if factor != 1.0:
        size = (binary.shape[1], binary.shape[0])
        ids = cv2.resize(ids.astype(np.uint16), size, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        ids = _nearest_fill(ids, binary > 0, max_dist=max(2, int(np.ceil(3 / factor))))
        cls = cv2.resize(cls, size, interpolation=cv2.INTER_NEAREST)
    else:
        ids = _nearest_fill(ids, binary > 0, max_dist=2)
    labels = np.where(binary > 0, cls, 0).astype(np.uint8)

    segments = []
    for i in np.unique(ids):
        if i == 0:
            continue
        mask = (ids == i) & (binary > 0)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0, y0 = xs.min(), ys.min()
        crop = np.where(mask[y0:ys.max() + 1, x0:xs.max() + 1], 255, 0).astype(np.uint8)
        segments.append(seg.Segment((int(x0), int(y0), crop.shape[1], crop.shape[0]), crop))

    segments = seg.remove_noise(segments)
    segments = seg.merge_broken_pieces(segments)
    segments = seg.attach_fragments(segments)
    return segments, labels


def segment_nn(binary, name):
    return run(binary, name)[0]


def pixel_labels(binary, name):
    return run(binary, name)[1]
