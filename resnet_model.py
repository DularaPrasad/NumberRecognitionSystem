"""
resnet_model.py

Task 3, final comparison model: a small ResNet (Residual Network) for MNIST.

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class ResidualBlock(nn.Module):
    """One residual block: two 3x3 convolutions plus a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path F(x): conv -> batchnorm -> relu -> conv -> batchnorm.
        # bias=False because BatchNorm already adds its own learnable shift.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Skip path. If the block keeps the same shape, x is passed through
        # unchanged. If the block changes the number of channels or shrinks
        # the image (stride=2), x no longer matches F(x) in shape and can't be
        # added directly, so a small 1x1 convolution reshapes x to match.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # first half of F(x)
        out = self.bn2(self.conv2(out))             # second half of F(x)
        out = out + self.shortcut(x)                 # the skip connection: F(x) + x
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem: one ordinary conv layer to start (1 grayscale channel -> 16).
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Three residual blocks. Each of the last two doubles the channels
        # and halves the image size (stride=2): 28x28 -> 14x14 -> 7x7.
        self.layer1 = ResidualBlock(16, 16, stride=1)   # 28x28, 16 channels
        self.layer2 = ResidualBlock(16, 32, stride=2)   # 14x14, 32 channels
        self.layer3 = ResidualBlock(32, 64, stride=2)   # 7x7, 64 channels
        # Global average pooling: averages each of the 64 maps down to one
        # number, so no huge flattened layer is needed before the final one.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)              # 64 x 7 x 7 -> 64 x 1 x 1
        x = x.view(x.size(0), -1)        # flatten to 64 numbers per image
        x = self.fc(x)                   # 64 -> 10 digit scores
        return x


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
plt.savefig("resnet_confusion_matrix.png")
print("Saved confusion matrix to resnet_confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("ResNet training loss over epochs")
plt.savefig("resnet_training_curve.png")
print("Saved training curve to resnet_training_curve.png")
plt.show()

torch.save(model.state_dict(), "resnet_model.pth")
print("Saved trained model to resnet_model.pth")
