"""
mlp_model.py

Task 3, comparison model 1 of 3: a Multi-Layer Perceptron (MLP) for MNIST.
No convolution, no pooling - just plain fully connected layers. Trained
and evaluated exactly like the CNN
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


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # No convolution, no pooling - three fully connected layers only.
        # Input is 28*28=784: every pixel treated as a separate, unrelated
        # number, exactly like KNN's flattening. The model has no built-in
        # concept of "these two pixels are physically next to each other."
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten immediately - no conv layers first
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleMLP().to(device)


total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
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
print(f"MLP test accuracy: {accuracy:.2f}%")

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
plt.title("MLP confusion matrix")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color=text_color, fontsize=8)
plt.tight_layout()
plt.savefig("mlp_confusion_matrix.png")
print("Saved confusion matrix to mlp_confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("MLP training loss over epochs")
plt.savefig("mlp_training_curve.png")
print("Saved training curve to mlp_training_curve.png")
plt.show()

torch.save(model.state_dict(), "mlp_model.pth")
print("Saved trained model to mlp_model.pth")
