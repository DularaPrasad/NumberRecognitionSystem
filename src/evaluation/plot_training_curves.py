"""
plot_training_curves.py

Draws the training loss of all four Task 3 models (MLP, LeNet-5, CNN, ResNet)
in one graph, so their learning curves can be compared directly.

The loss values are the per-epoch averages printed by each model script
(src/models/*.py) in its 10-epoch run, copied in below. If a model is
re-trained, update its list of 10 numbers and run this script again.

Run from anywhere:   python src\\evaluation\\plot_training_curves.py
Output:              results/all_models_training_curve.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

# This file lives in src/evaluation/, so the repository root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Average training loss at the end of each of the 10 epochs.
LOSSES = {
    "MLP":     [0.3446, 0.1414, 0.0962, 0.0726, 0.0570, 0.0468, 0.0376, 0.0319, 0.0241, 0.0211],
    "LeNet-5": [0.2989, 0.0804, 0.0584, 0.0461, 0.0376, 0.0318, 0.0266, 0.0246, 0.0195, 0.0171],
    "CNN":     [0.2357, 0.0688, 0.0506, 0.0410, 0.0339, 0.0287, 0.0251, 0.0224, 0.0194, 0.0164],
    "ResNet":  [0.2518, 0.0519, 0.0381, 0.0303, 0.0252, 0.0227, 0.0186, 0.0180, 0.0155, 0.0136],
}

# Colour-blind-friendly colours, with a different marker per model so the
# lines can still be told apart in a black-and-white printout.
STYLE = {
    "MLP":     {"color": "#D55E00", "marker": "s"},
    "LeNet-5": {"color": "#009E73", "marker": "^"},
    "CNN":     {"color": "#0072B2", "marker": "o"},
    "ResNet":  {"color": "#CC79A7", "marker": "D"},
}

epochs = list(range(1, 11))

fig, ax = plt.subplots(figsize=(8.5, 5.6))


def draw(axis, size):
    for name, losses in LOSSES.items():
        axis.plot(epochs, losses, linewidth=1.8, markersize=size,
                  label=f"{name} (final loss {losses[-1]:.4f})", **STYLE[name])


draw(ax, 5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (average per batch)")
ax.set_title("Training loss per epoch for all four models (10 epochs each)")
ax.set_xticks(epochs)
ax.set_xlim(0.6, 10.4)
ax.set_ylim(0, 0.36)
ax.grid(True, alpha=0.3)

# Zoomed view of epochs 5 to 10: after epoch 3 the curves sit close together,
# so this is where the differences (and how flat each curve is) show up.
inset = ax.inset_axes([0.43, 0.42, 0.53, 0.50])
draw(inset, 4)
inset.set_xlim(4.8, 10.2)
inset.set_ylim(0.008, 0.062)
inset.set_xticks(range(5, 11))
inset.tick_params(labelsize=8)
inset.set_title("Zoom: epochs 5 to 10", fontsize=9)
inset.grid(True, alpha=0.3)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
fig.subplots_adjust(bottom=0.24)

out = RESULTS / "all_models_training_curve.png"
fig.savefig(out, dpi=300)
print(f"Saved {out}")
