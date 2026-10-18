"""
test_handoff.py

Smoke test for the handoff between Task 1 (Declan's preprocessing) and Task 2
(Charith's segmentation). It draws synthetic photos of the number 4071 and some
awkward variants, runs preprocessing then segmentation, and checks that the
right number of digits comes out, and that each digit is MNIST-ready.

Segmentation uses the rule-based "hybrid" method so the test needs no trained
network. Synthetic digits drawn with an OpenCV font are not handwriting, so
this tests the handoff (image format, ink colour, lighting), not recognition
accuracy.

Run from anywhere:   python src\\evaluation\\test_handoff.py
"""

import inspect
import sys
from pathlib import Path

import cv2
import numpy as np

# Both teammates' files import each other as plain modules, so their folders
# have to be on the import path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "segmentation"))

from Preprocessing import preprocess_photo   # noqa: E402  (Task 1)
from segmentation import segment_number      # noqa: E402  (Task 2)

# Newer versions of preprocess_photo can take method="adaptive" (better for
# uneven lighting). Older ones cannot, so only try it if it is available.
METHODS = ["otsu"]
if "method" in inspect.signature(preprocess_photo).parameters:
    METHODS.append("adaptive")

rng = np.random.default_rng(0)


def paper(text):
    """Dark ink on light paper, like a photo of a handwritten number."""
    img = np.full((200, 640, 3), 245, np.uint8)
    cv2.putText(img, text, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 4.2, (30, 30, 30), 14, cv2.LINE_AA)
    return img


def with_shadow(img):
    """Darken the left side, as a shadow across a phone photo would."""
    grad = np.linspace(0.45, 1.0, img.shape[1])[None, :, None]
    noisy = img.astype(np.float32) * grad + rng.normal(0, 6, img.shape)
    return noisy.clip(0, 255).astype(np.uint8)


CASES = [
    ("clean photo of 4071",       paper("4071"),                4),
    ("white ink on black, 4071",  255 - paper("4071"),          4),
    ("photo with a shadow, 4071", with_shadow(paper("4071")),   4),
    ("single digit, 7",           paper("7"),                   1),
    ("blank paper (no digits)",   np.full((200, 640, 3), 245, np.uint8), 0),
]

passed = total = 0
print(f"{'case':28s} {'preprocessing':14s} digits found   result")
for name, image, expected in CASES:
    for method in METHODS:
        kwargs = {"method": method} if "method" in inspect.signature(preprocess_photo).parameters else {}
        _, binary = preprocess_photo(image, **kwargs)
        digits = segment_number(binary, method="hybrid")

        ok = len(digits) == expected
        ok = ok and all(d.digit28.shape == (28, 28) and d.digit28.dtype == np.uint8 for d in digits)
        ok = ok and [d.box[0] for d in digits] == sorted(d.box[0] for d in digits)  # left to right

        total += 1
        passed += ok
        print(f"{name:28s} {method:14s} {len(digits)} (expected {expected})   {'PASS' if ok else 'FAIL'}")

print(f"\n{passed} of {total} checks passed")
