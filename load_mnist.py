"""
load_mnist.py

Downloads the MNIST dataset and gives us a first look at it:
- prints how many images are in the training/test sets
- prints the shape of a single image 
- saves a 5x5 grid of sample digits as a PNG
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --define how raw images get converted into tensor (which PyTorch can use).
# ToTensor(): converts the image into a tensor (a multi-dimensional array of numbers), and rescales pixel values from 0-255 down to 0.0-1.0.
transform = transforms.ToTensor()

# --download the training and test sets.
# root="data" means "save the downloaded files into a folder called data".
# download=True means "if it's not already there, fetch it from the internet". This is a one time downloads  - then it loads from the local copy.
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# --print basic facts to confirm everything loaded correctly.
print(f"Training set size: {len(train_data)} images")
print(f"Test set size: {len(test_data)} images")

image, label = train_data[0]
print(f"Each image shape: {tuple(image.shape)}")   # expect (1, 28, 28) -> 1 grayscale channel, 28x28 pixels
print(f"First image's label: {label}")

# --save a 5x5 grid of sample digits so we can SEE the data, not just trust the numbers.
fig, axes = plt.subplots(5, 5, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    image, label = train_data[i]
    ax.imshow(image.squeeze(), cmap="gray")   # squeeze() drops the "1 channel" dimension for display.
    ax.set_title(str(label), fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_sample_grid.png")
print("Saved a preview grid to mnist_sample_grid.png")
plt.show()
