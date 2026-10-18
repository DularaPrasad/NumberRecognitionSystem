"""
evaluate_saved_model.py

Re-evaluates ONE saved model on the 10,000 MNIST test images (no training) and
prints its accuracy, its accuracy for each digit and its most common mistakes.
It also saves the model's confusion matrix to results/, replacing the old one,
so the picture always matches the weights in saved_models/.

Run from anywhere (model is cnn, mlp, lenet5 or resnet; default cnn):
    python src\\evaluation\\evaluate_saved_model.py cnn
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw straight to a file; no window opens
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from architectures import ARCHITECTURES, WEIGHT_FILES  # noqa: E402

parser = argparse.ArgumentParser(description="Re-evaluate one saved model on the MNIST test set.")
parser.add_argument("model", nargs="?", default="cnn", choices=sorted(ARCHITECTURES))
name = parser.parse_args().model

device = "cuda" if torch.cuda.is_available() else "cpu"
path = ROOT / "saved_models" / WEIGHT_FILES[name]
if not path.exists():
    sys.exit(f"Saved model not found: {path}")

model = ARCHITECTURES[name]()
model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
model.to(device).eval()

test_data = datasets.MNIST(root=ROOT / "data", train=False, download=True, transform=transforms.ToTensor())
loader = DataLoader(test_data, batch_size=64, shuffle=False)   # same batch size as the training scripts

predictions, truths = [], []
with torch.no_grad():
    for images, labels in loader:
        predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
        truths.extend(labels.tolist())

cm = confusion_matrix(truths, predictions)
correct = int(cm.trace())
print(f"{name}: {correct} of {len(truths)} correct, test accuracy {100 * correct / len(truths):.2f}%\n")

print("Per-digit accuracy:")
per_digit = []
for digit in range(10):
    digit_total = int(cm[digit].sum())
    digit_correct = int(cm[digit][digit])
    per_digit.append(100 * digit_correct / digit_total)
    print(f"  Digit {digit}: {per_digit[-1]:.2f}% ({digit_correct}/{digit_total})")
weakest = min(range(10), key=lambda d: per_digit[d])
print(f"\nWeakest digit: {weakest} ({per_digit[weakest]:.2f}%)")

mistakes = sorted(((int(cm[a][b]), a, b) for a in range(10) for b in range(10) if a != b), reverse=True)[:5]
print("Most common mistakes (actual digit read as predicted digit):")
for count, actual, predicted in mistakes:
    print(f"  {actual} read as {predicted}: {count} images")

# Confusion matrix picture, in the same style as the training scripts.
TITLES = {"cnn": "CNN", "mlp": "MLP", "lenet5": "LeNet-5", "resnet": "ResNet"}
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted digit")
plt.ylabel("Actual digit")
plt.title(f"{TITLES[name]} confusion matrix")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
out = RESULTS / f"{name}_confusion_matrix.png"
plt.savefig(out)
print(f"\nSaved {out}")
