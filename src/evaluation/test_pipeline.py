"""
test_pipeline.py

Checks the glue code in src/integration/pipeline.py: that preprocessing,
segmentation and classification are connected correctly. Most checks use a
stand-in classifier (it just returns 0, 1, 2, ...), so they test the wiring and
need no trained model. The last check runs the real model if PyTorch and the
saved weights are available.

Run from anywhere:   python src\\evaluation\\test_pipeline.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "integration"))

import pipeline  # noqa: E402
from pipeline import NumberRecognizer, annotate  # noqa: E402

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not condition else ""))


class StandInRecognizer(NumberRecognizer):
    """The real recognizer, but its classifier just numbers the digits 0, 1, 2, ..."""

    def __init__(self, preprocessing="otsu"):
        self.segmenter = "hybrid"
        self.preprocessing = preprocessing
        self._device = "cpu"
        self.model_name = "stand-in"

    def classify(self, digit_images):
        return list(range(len(digit_images))), [0.9] * len(digit_images)


def photo(text):
    """Dark ink on light paper, like a photo of a handwritten number."""
    img = np.full((200, 640, 3), 245, np.uint8)
    cv2.putText(img, text, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 4.2, (30, 30, 30), 14, cv2.LINE_AA)
    return img


recognizer = StandInRecognizer()
image = photo("4071")
result = recognizer.recognise(image)

check("finds the 4 digits of a clean photo", len(result.digits) == 4, f"found {len(result.digits)}")
check("number is the predictions joined left to right", result.number == "0123", result.number)
check("every result list has one entry per digit",
      len(result.confidences) == len(result.boxes) == len(result.digit_images) == len(result.digits))
check("boxes are in left-to-right order", [b[0] for b in result.boxes] == sorted(b[0] for b in result.boxes))
check("digit images are 28x28 uint8 (MNIST layout)",
      all(d.shape == (28, 28) and d.dtype == np.uint8 for d in result.digit_images))
check("binary image has the same size as the input", result.binary.shape == image.shape[:2])
check("model name is reported", result.model == "stand-in")

blank = recognizer.recognise(np.full((200, 640, 3), 245, np.uint8))
check("blank paper gives an empty result, not an error",
      blank.number == "" and blank.digits == [] and blank.boxes == [])

check("works on a grayscale image too", recognizer.recognise(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).number == "0123")

if pipeline._PREPROCESS_HAS_METHOD:
    check("adaptive preprocessing works", StandInRecognizer("adaptive").recognise(image).number == "0123")
else:
    print("SKIP  adaptive preprocessing (this Preprocessing.py has no method option)")

drawn = annotate(image, result)
check("annotated picture has the same size and shows the boxes",
      drawn.shape == image.shape and not np.array_equal(drawn, image))

try:
    recognizer.recognise(None)
    check("missing image raises a clear error", False)
except ValueError:
    check("missing image raises a clear error", True)

for label, kwargs in [("unknown segmenter", {"segmenter": "bogus"}),
                      ("unknown preprocessing", {"preprocessing": "bogus"}),
                      ("unknown model", {"model": "bogus"})]:
    try:
        NumberRecognizer(**kwargs)
        check(f"{label} is rejected", False)
    except ValueError:
        check(f"{label} is rejected", True)

# Real model, only if PyTorch and the saved weights are available.
try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP  real-model check (PyTorch is not installed here)")
else:
    try:
        real = NumberRecognizer(model="cnn", segmenter="hybrid", preprocessing="otsu")
        out = real.recognise(image)
        print(f"INFO  real CNN read the synthetic photo as '{out.number}' "
              "(not scored: the digits are drawn with a font, not handwriting)")
        check("real model loads and runs through the pipeline", len(out.digits) == len(out.confidences) > 0)
    except Exception as error:   # report any loading problem clearly
        check("real model loads and runs through the pipeline", False, f"{type(error).__name__}: {error}")

print(f"\n{sum(results)} of {len(results)} checks passed")
