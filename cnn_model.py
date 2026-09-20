"""
cnn_model.py

Task 3, next step: a Convolutional Neural Network (CNN) for MNIST.
Goal: a first genuinely working CNN, trained on the full training set,
with a real accuracy number to compare against last week's KNN baseline.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Use the GPU if it's available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Same dataset loading as before in KNN model.
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# DataLoader feeds images to the model in small batches instead of all at
# once, and shuffle=True mixes up the order each epoch so the model doesn't
# accidentally learn anything from the order images happen to be in.
BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1: 16 filters, each learning to detect one simple
        # pattern (an edge, a curve). padding=1 keeps the output the same
        # 28x28 size as the input.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Conv block 2: 32 filters, built on top of block 1's output, so
        # they can combine simple patterns into more complex shapes.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Shrinks the image by half each time it's used, keeping only the
        # strongest signal in each small region.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Final decision layer: from 32*7*7 flattened features to 10 scores,
        # one per digit.
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)                 # flatten each image to one row
        x = self.fc1(x)
        return x


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()                      # measures how wrong the predictions are
optimizer = optim.Adam(model.parameters(), lr=0.001)   # adjusts the model to reduce that error

EPOCHS = 5
train_losses = []

start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear the previous batch's calculations
        outputs = model(images)            # forward pass: make predictions
        loss = criterion(outputs, labels)  # how wrong were those predictions?
        loss.backward()                    # work out how to adjust the model to improve
        optimizer.step()                   # actually apply that adjustment

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")

elapsed = time.time() - start
print(f"Training took {elapsed:.1f} seconds")

# Evaluation: no learning happens here, just checking performance.
# This time we also keep every individual prediction and true label, so we
# can break accuracy down by digit afterwards, not just one overall number.
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

        # .cpu() moves the values off the GPU (numpy can't read GPU memory),
        # .numpy() converts to a plain numpy array, .extend() adds each
        # individual value from this batch onto the running list.
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"CNN test accuracy: {accuracy:.2f}%")

# Confusion matrix: a 10x10 table where row = the true digit, column = what
# the model guessed. The diagonal is correct guesses; everything off the
# diagonal is a specific kind of mistake (e.g. how many real 4s got called 9).
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

print("\nPer-digit accuracy:")
for digit in range(10):
    digit_total = cm[digit].sum()      # every test image that was actually this digit
    digit_correct = cm[digit][digit]   # how many of those were guessed correctly
    digit_acc = 100 * digit_correct / digit_total
    print(f"  Digit {digit}: {digit_acc:.2f}% ({digit_correct}/{digit_total})")

# Plot the confusion matrix as a heatmap, with the actual counts written on
# each cell so it's readable as a standalone image .
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted digit")
plt.ylabel("Actual digit")
plt.title("CNN confusion matrix")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        # White text on dark cells, black text on light cells, so the
        # number stays readable regardless of the cell's shade.
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color=text_color, fontsize=8)
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")
print("Saved confusion matrix to cnn_confusion_matrix.png")
plt.show()

# Save a plot showing loss going down over training - visual evidence of learning.
# plt.figure() here starts a brand new, blank figure, so this plot doesn't
# get drawn on top of the confusion matrix figure above.
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("CNN training loss over epochs")
plt.savefig("cnn_training_curve.png")
print("Saved training curve to cnn_training_curve.png")
plt.show()

# Save the trained model so we don't have to retrain it from scratch every
# time we want to use it.
# state_dict() is a dictionary holding every one of the model's learned
# numbers - every filter value, every weight - essentially the entire result
# of training, packaged up.
torch.save(model.state_dict(), "cnn_model.pth")
print("Saved trained model to cnn_model.pth")
