"""
lenet5_model.py

Task 3, comparison model 2 of 3: LeNet-5, adapted for MNIST.
LeNet-5 is the architecture Yann LeCun designed in 1998 specifically for
handwritten digit recognition.

Two deliberate, documented deviations from the 1998 original:
1. ReLU + max-pooling instead of the original tanh + average-pooling -
   modern practice, and kept consistent with our CNN and MLP so the
   activation/pooling choice isn't a confounding variable in the comparison.
2. padding=2 on the first conv layer instead of resizing images to the
   original 32x32 - functionally equivalent, keeps every other layer's
   original size (6 filters, 16 filters, 400 -> 120 -> 84 -> 10) unchanged.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from pathlib import Path

# Paths are worked out from this file's location, so the script works from any
# folder. This file lives in src/<folder>/, so the repository root is two
# levels up.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"              # saved plots
RESULTS.mkdir(exist_ok=True)
SAVED_MODELS = ROOT / "saved_models"    # saved model weights (.pth)
SAVED_MODELS.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.ToTensor()
train_data = datasets.MNIST(root=ROOT / "data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=ROOT / "data", train=False, download=True, transform=transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=2 here has the same effect as padding the input image to
        # 32x32 would have (LeNet-5's original input size) - it keeps every
        # other layer's original dimensions intact for a 28x28 input.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14, 6 channels
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 10x10 -> 5x5, 16 channels
        x = x.view(x.size(0), -1)                  # flatten to 400 (16*5*5)
        x = self.relu(self.fc1(x))                  # 400 -> 120
        x = self.relu(self.fc2(x))                  # 120 -> 84
        x = self.fc3(x)                              # 84 -> 10
        return x


model = LeNet5().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
train_losses = []

start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")

elapsed = time.time() - start
print(f"Training took {elapsed:.1f} seconds")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"LeNet-5 test accuracy: {accuracy:.2f}%")

cm = confusion_matrix(all_labels, all_preds)

print("\nPer-digit accuracy:")
for digit in range(10):
    digit_total = cm[digit].sum()
    digit_correct = cm[digit][digit]
    digit_acc = 100 * digit_correct / digit_total
    print(f"  Digit {digit}: {digit_acc:.2f}% ({digit_correct}/{digit_total})")

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted digit")
plt.ylabel("Actual digit")
plt.title("LeNet-5 confusion matrix")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color=text_color, fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "lenet5_confusion_matrix.png")
print("Saved confusion matrix to results/lenet5_confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("LeNet-5 training loss over epochs")
plt.savefig(RESULTS / "lenet5_training_curve.png")
print("Saved training curve to results/lenet5_training_curve.png")
plt.show()

torch.save(model.state_dict(), SAVED_MODELS / "lenet5_model.pth")
print("Saved trained model to saved_models/lenet5_model.pth")
