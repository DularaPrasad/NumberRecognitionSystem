"""
baseline_knn.py

Task 3, first approach: a K-Nearest Neighbours (KNN) baseline on MNIST.

Why start here instead of a CNN: KNN is one of the simplest ML methods -
for each test image, it finds the K most visually similar training images
(by raw pixel distance) and takes a majority vote of their labels. No
training/learning happens in the usual sense - it just memorises the data.

Goal today: get ANY real accuracy number, so next week's CNN has something
concrete to compare against and beat.
"""

import time
import numpy as np
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pathlib import Path

# Paths are worked out from this file's location, so the script works from any
# folder. This file lives in src/<folder>/, so the repository root is two
# levels up.
ROOT = Path(__file__).resolve().parents[2]

transform = transforms.ToTensor()
train_data = datasets.MNIST(root=ROOT / "data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=ROOT / "data", train=False, download=True, transform=transform)

# Using a subset for speed on this first pass - KNN compares every test image
# against every training image, so it gets slow with the full 60,000. We can
# scale this up later once the basic pipeline is proven to work.
N_TRAIN = 5000
N_TEST = 1000


def to_numpy(dataset, n):
    # KNN needs flat feature vectors, not 2D images: each 1x28x28 image
    # becomes a single row of 784 numbers (28*28 = 784).
    X = np.stack([dataset[i][0].numpy().flatten() for i in range(n)])
    y = np.array([dataset[i][1] for i in range(n)])
    return X, y


X_train, y_train = to_numpy(train_data, N_TRAIN)
X_test, y_test = to_numpy(test_data, N_TEST)

print(f"Training on {N_TRAIN} images, testing on {N_TEST} images")

start = time.time()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
elapsed = time.time() - start

acc = accuracy_score(y_test, predictions)
print(f"KNN (k=3) baseline accuracy: {acc * 100:.2f}%")
print(f"Time taken: {elapsed:.1f} seconds")
