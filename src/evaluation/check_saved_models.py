"""
check_saved_models.py

Loads each trained model from saved_models/ and re-scores it on the 10,000
MNIST test images (no training happens), then compares the result with the
accuracy reported for it. This confirms that the weights the pipeline loads are
the same ones the results in the worklogs and the report came from.

Run from anywhere:   python src\\evaluation\\check_saved_models.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from architectures import ARCHITECTURES, WEIGHT_FILES  # noqa: E402

# Test accuracy (%) of each saved model. The CNN's saved weights come from a repeat of the
# 10-epoch run (98.75%); its first 10-epoch run scored 98.96%. The other three are from
# the runs reported in the worklogs.
REPORTED = {"cnn": 98.75, "mlp": 97.84, "lenet5": 98.87, "resnet": 99.22}

device = "cuda" if torch.cuda.is_available() else "cpu"
test_data = datasets.MNIST(root=ROOT / "data", train=False, download=True, transform=transforms.ToTensor())
# Batch size 64, the same as in the training scripts, so the numbers are computed the same way.
loader = DataLoader(test_data, batch_size=64, shuffle=False)
print(f"Device: {device}\n")

all_match = True
for name, reported in REPORTED.items():
    path = ROOT / "saved_models" / WEIGHT_FILES[name]
    if not path.exists():
        print(f"{name:8s} MISSING: {path}")
        all_match = False
        continue

    model = ARCHITECTURES[name]()
    # map_location lets weights saved on a GPU load on a CPU-only computer
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            correct += (predicted == labels).sum().item()

    expected = round(reported * 100)          # 10,000 test images, so 98.96% = 9,896 correct
    difference = abs(correct - expected)
    if difference == 0:
        verdict = "matches the reported accuracy"
    elif difference <= 2:
        verdict = f"within {difference} image(s) of the reported {reported:.2f}% (GPU rounding)"
    else:
        verdict = f"DIFFERENT from the reported {reported:.2f}%"
        all_match = False
    print(f"{name:8s} {correct:5d} correct   {100 * correct / len(test_data):6.2f}%   {verdict}")

if not all_match:
    print("\nAt least one saved model does not match the accuracy in the worklogs. This usually means the")
    print("model script was run again after the reported results, which replaced the saved weights.")
    print("Either update the reported number for that model, or restore the earlier .pth from Git history.")
