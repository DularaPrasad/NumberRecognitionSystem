"""
unet_segment.py  -  kept so older code still works.
All networks (including the U-Net) now run through nn_segment.py.
"""

from nn_segment import load_model as _load, pixel_labels as _pixel_labels, segment_nn


def load_model():
    return _load("unet")


def segment_unet(binary):
    return segment_nn(binary, "unet")


def pixel_labels(binary, name="unet"):
    return _pixel_labels(binary, name)
