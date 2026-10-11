"""
unet_segment.py  -  use the TRAINED U-Net to split a number into digits
=======================================================================
This is what happens when segment_number(image, method="unet") is called:

  1. Size check: the network was trained on digits about 20 pixels tall.
     A phone photo may have digits 150 pixels tall, so the image is scaled
     to the size the network knows (thin pen lines are thickened first so
     they do not disappear when the image is shrunk).
  2. The U-Net labels every pixel: background / inside a digit / border.
  3. Each separate patch of "inside a digit" pixels is one digit (a "seed").
  4. Border pixels are given to the nearest seed, so touching digits are
     cut exactly where the network saw the contact.
  5. The result is mapped back to the original image size, and the same
     clean-up as the rule-based method is applied (drop specks, join a digit
     drawn in two pieces).

You need the trained model file models/unet_seg.pt (made by train_unet.py).
"""

import os

import cv2
import numpy as np
import torch
from scipy import ndimage

import segmentation as seg
from unet_model import UNet

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "unet_seg.pt")
TARGET_DIGIT_HEIGHT = 21   # typical digit height (pixels) in the training images
MIN_SEED_AREA = 12         # "inside" patches smaller than this are ignored
TARGET_STROKE = 2.5        # pen thickness (pixels) the network is used to

_net = None  # the network is loaded once and then reused


def load_model(path=MODEL_PATH):
    global _net
    if _net is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found at {path}. Run:  python train_unet.py")
        net = UNet()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        _net = net
    return _net


def _stroke_width(binary):
    """Average pen thickness in pixels (from the distance transform)."""
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return 4.0 * float(dist[dist > 0].mean()) if dist.any() else 0.0


def _scale_factor(binary):
    """How much to resize so the digits are about TARGET_DIGIT_HEIGHT tall."""
    blobs = seg.remove_noise(seg.segment_connected_components(binary))
    if not blobs:
        return 1.0
    typical = seg._typical_height(blobs)
    factor = TARGET_DIGIT_HEIGHT / typical
    return 1.0 if 0.75 <= factor <= 1.35 else factor


def _predict(binary_small):
    """Run the network. Returns the class (0/1/2) of every pixel."""
    h, w = binary_small.shape
    pad = 8
    H = (h + 2 * pad + 7) // 8 * 8          # the U-Net needs sizes divisible by 8
    W = (w + 2 * pad + 7) // 8 * 8
    canvas = np.zeros((H, W), np.float32)
    canvas[pad:pad + h, pad:pad + w] = binary_small / 255.0
    with torch.no_grad():
        scores = load_model()(torch.from_numpy(canvas)[None, None])
    return scores.argmax(1)[0].numpy()[pad:pad + h, pad:pad + w]


def _nearest_fill(ids, todo, max_dist):
    """Give every pixel in `todo` the id of the nearest labelled pixel
    (if it is within max_dist pixels)."""
    if not (ids > 0).any():
        return ids
    dist, (iy, ix) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    fill = todo & (ids == 0) & (dist <= max_dist)
    ids = ids.copy()
    ids[fill] = ids[iy[fill], ix[fill]]
    return ids


def _prepare_and_predict(binary):
    """Steps 1-2: resize to the training scale and run the network."""
    # 1. bring the digits to the size the network was trained on
    factor = _scale_factor(binary)
    work = binary
    if factor != 1.0:
        needed = TARGET_STROKE / factor             # stroke width needed before shrinking
        sw = _stroke_width(binary)
        if sw < needed:
            k = int(round(needed - sw)) + 1
            work = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        size = (max(1, round(binary.shape[1] * factor)), max(1, round(binary.shape[0] * factor)))
        work = cv2.resize(work, size, interpolation=cv2.INTER_AREA)
        work = np.where(work > 60, 255, 0).astype(np.uint8)

    # 2. the network labels every pixel
    return factor, work, _predict(work)


def pixel_labels(binary):
    """The network's label for every ink pixel of the ORIGINAL image
    (0 background, 1 inside a digit, 2 border) - used to draw the demo picture."""
    if not binary.any():
        return np.zeros(binary.shape, np.uint8)
    _, _, cls = _prepare_and_predict(binary)
    cls = cv2.resize(cls.astype(np.uint8), (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(binary > 0, cls, 0).astype(np.uint8)


def segment_unet(binary):
    """Split a binary number image (ink = 255) into digit Segments using the U-Net."""
    if not binary.any():
        return []
    factor, work, cls = _prepare_and_predict(binary)
    ink = work > 0

    # 3. each patch of "inside a digit" pixels is one digit seed
    n, seeds, stats, _ = cv2.connectedComponentsWithStats(((cls == 1) & ink).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_SEED_AREA:
            seeds[seeds == i] = 0

    # 4. border (and any other ink) pixels go to the nearest seed
    ids = _nearest_fill(seeds, ink, max_dist=6)

    # 5. back to the original size
    if factor != 1.0:
        ids = cv2.resize(ids.astype(np.uint16), (binary.shape[1], binary.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(np.int32)
        ids = _nearest_fill(ids, binary > 0, max_dist=max(2, int(np.ceil(3 / factor))))
    else:
        ids = _nearest_fill(ids, binary > 0, max_dist=2)

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

    # 6. same clean-up as the rule-based method
    segments = seg.remove_noise(segments)
    segments = seg.merge_broken_pieces(segments)
    segments = seg.attach_fragments(segments)
    return segments
