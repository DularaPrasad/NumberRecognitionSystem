"""
pipeline.py

Task 4: the whole system in one place.

    image -> preprocessing (Task 1) -> segmentation (Task 2) -> digit model (Task 3) -> number

Use it from Python (this is what the GUI does):

    from pipeline import NumberRecognizer

    recognizer = NumberRecognizer(model="resnet")   # "cnn", "mlp", "lenet5" or "resnet"
    result = recognizer.recognise(image)            # image: a BGR array from cv2.imread
    result.number          # "4071"  ("" if no digits were found)
    result.digits          # [4, 0, 7, 1]
    result.confidences     # [0.99, 0.97, ...]  between 0 and 1
    result.digit_images    # the 28x28 uint8 image each prediction was made from
    result.boxes           # (x, y, w, h) of each digit in the original image
    result.binary          # the black-and-white image after preprocessing

    recognizer.set_model("cnn")                     # switch model without rebuilding

Or from the command line (run from anywhere):

    python src\\integration\\pipeline.py path\\to\\image.png --model resnet --save

Settings that can be compared in the evaluation (Task 4):
    model          which of the four trained digit classifiers to use
    segmenter      "unet" (trained network, default) or a rule-based method such as "hybrid"
    preprocessing  "adaptive" (copes with shadows) or "otsu" (one global threshold)
"""

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# This file lives in src/integration/, so the repository root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# The teammates' modules import each other by plain name (for example
# "import segmentation as seg"), so their folders have to be on the import path.
for _folder in ("preprocessing", "segmentation", "models"):
    sys.path.insert(0, str(ROOT / "src" / _folder))

from Preprocessing import preprocess_photo   # noqa: E402  Task 1
from segmentation import segment_number      # noqa: E402  Task 2

MODEL_NAMES = ("cnn", "mlp", "lenet5", "resnet")
SEGMENTERS = ("unet", "hybrid", "projection", "connected_components", "contours")

# Does the installed Preprocessing.py have the optional method="adaptive"?
_PREPROCESS_HAS_METHOD = "method" in inspect.signature(preprocess_photo).parameters

# Models are loaded once and kept, so switching back and forth is instant.
_LOADED_MODELS = {}


@dataclass
class RecognitionResult:
    number: str          # the recognised number, or "" if no digits were found
    digits: list         # predicted digits, left to right
    confidences: list    # confidence (0 to 1) of each predicted digit
    digit_images: list   # the 28x28 uint8 image behind each prediction
    boxes: list          # (x, y, w, h) of each digit in the original image
    binary: np.ndarray   # black-and-white image after preprocessing
    model: str           # which model made the predictions


def _load_model(name, device):
    """Load one trained model from saved_models/ (cached after the first time)."""
    key = (name, device)
    if key not in _LOADED_MODELS:
        import torch
        from architectures import ARCHITECTURES, WEIGHT_FILES

        path = ROOT / "saved_models" / WEIGHT_FILES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {path}\n"
                "Pull the latest version from GitHub, or train it with "
                f"python src\\models\\{name}_model.py")
        model = ARCHITECTURES[name]()
        # map_location lets weights that were saved on a GPU load on a CPU-only laptop
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        _LOADED_MODELS[key] = model.to(device).eval()   # eval(): BatchNorm must not be in training mode
    return _LOADED_MODELS[key]


class NumberRecognizer:
    def __init__(self, model="resnet", segmenter="unet", preprocessing="adaptive", device=None):
        if segmenter not in SEGMENTERS:
            raise ValueError(f"Unknown segmenter '{segmenter}'. Choose from {SEGMENTERS}")
        if preprocessing not in ("adaptive", "otsu"):
            raise ValueError("preprocessing must be 'adaptive' or 'otsu'")
        if preprocessing == "adaptive" and not _PREPROCESS_HAS_METHOD:
            raise RuntimeError("This Preprocessing.py has no method='adaptive' option. "
                               "Pull the latest version from GitHub, or use preprocessing='otsu'.")
        self.segmenter = segmenter
        self.preprocessing = preprocessing
        self._device = device
        self.model_name = None
        self._model = None
        self.set_model(model)

    def set_model(self, name):
        """Choose which trained model classifies the digits."""
        name = str(name).lower()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = _load_model(name, self._device)
        self.model_name = name

    def classify(self, digit_images):
        """Predict a digit (and a confidence) for each 28x28 uint8 digit image."""
        import torch

        batch = np.stack(digit_images).astype(np.float32) / 255.0      # (N, 28, 28), values 0 to 1
        x = torch.from_numpy(batch).unsqueeze(1).to(self._device)      # (N, 1, 28, 28)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(x), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        return predicted.tolist(), [round(c, 4) for c in confidence.tolist()]

    def recognise(self, image):
        if image is None:
            raise ValueError("image is None. Did cv2.imread fail to find the file?")

        # Task 1: preprocessing. Only the full-size black-and-white image is used
        # (the 28x28 it also returns is a plain resize of the whole picture).
        options = {"method": self.preprocessing} if _PREPROCESS_HAS_METHOD else {}
        _, binary = preprocess_photo(image, **options)

        # Task 2: split the number into digits, left to right. Each one comes back
        # already in MNIST layout (28x28, white on black, centred).
        segments = segment_number(binary, method=self.segmenter)
        if not segments:
            return RecognitionResult("", [], [], [], [], binary, self.model_name)

        # Task 3: classify every digit with the chosen model.
        digit_images = [s.digit28 for s in segments]
        digits, confidences = self.classify(digit_images)
        return RecognitionResult("".join(str(d) for d in digits), digits, confidences,
                                 digit_images, [s.box for s in segments], binary, self.model_name)


def annotate(image, result):
    """A copy of the image with a box and the predicted digit around each digit."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    size = max(out.shape[:2])
    thickness = max(1, round(size / 300))
    font = max(0.5, size / 600)
    for (x, y, w, h), digit, conf in zip(result.boxes, result.digits, result.confidences):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), thickness)
        cv2.putText(out, f"{digit} ({conf:.0%})", (x, max(int(30 * font), y - 4 * thickness)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thickness, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Recognise a handwritten number in an image.")
    parser.add_argument("image", help="path to the image")
    parser.add_argument("--model", default="resnet", choices=MODEL_NAMES)
    parser.add_argument("--segmenter", default="unet", choices=SEGMENTERS)
    parser.add_argument("--preprocessing", default="adaptive", choices=("adaptive", "otsu"))
    parser.add_argument("--save", action="store_true", help="save an annotated picture into results/")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Cannot read the image: {args.image}")

    recognizer = NumberRecognizer(args.model, args.segmenter, args.preprocessing)
    result = recognizer.recognise(image)

    if not result.number:
        print("No digits found in this image.")
        return
    print(f"Number: {result.number}   (model: {result.model})")
    for i, (digit, conf) in enumerate(zip(result.digits, result.confidences), start=1):
        print(f"  digit {i}: {digit}  (confidence {conf:.1%})")

    if args.save:
        RESULTS.mkdir(exist_ok=True)
        out_path = RESULTS / f"{Path(args.image).stem}_{result.model}_{args.segmenter}.png"
        cv2.imwrite(str(out_path), annotate(image, result))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
