"""
resnet_model.py

Task 3, final comparison model: a small ResNet (Residual Network) for MNIST.

The key idea (He et al., "Deep Residual Learning for Image Recognition",
2015): instead of each block learning y = F(x), it learns y = F(x) + x.
The input x is carried around the convolutions by a "skip connection" and
added back onto the output. This helps deeper networks train reliably.

Design decisions (documented so the comparison is fair and defensible):
1. A compact ResNet built from scratch, NOT the full ResNet-18. ResNet-18
   was designed for 224x224 colour photos and has ~11 million parameters,
   which is far more than 28x28 grayscale digits need.
2. Training settings identical to our CNN, MLP and LeNet-5: 10 epochs,
   batch size 64, Adam with lr 0.001, full 60,000-image training set.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from architectures import SmallResNet

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


model = SmallResNet().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
train_losses = []

start = time.time()
for epoch in range(EPOCHS):
    model.train()   # matters here: BatchNorm behaves differently in train vs eval
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
print(f"ResNet test accuracy: {accuracy:.2f}%")

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
plt.title("ResNet confusion matrix")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color=text_color, fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "resnet_confusion_matrix.png")
print("Saved confusion matrix to results/resnet_confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("ResNet training loss over epochs")
plt.savefig(RESULTS / "resnet_training_curve.png")
print("Saved training curve to results/resnet_training_curve.png")
plt.show()

torch.save(model.state_dict(), SAVED_MODELS / "resnet_model.pth")
print("Saved trained model to saved_models/resnet_model.pth")
