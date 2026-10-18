"""
architectures.py

The four digit-recognition network designs from Task 3, defined in one place
so that both the training scripts (cnn_model.py, mlp_model.py, lenet5_model.py,
resnet_model.py) and the pipeline (src/integration/pipeline.py) use exactly
the same code. Importing this file only defines the classes: it does not
download data or train anything.

The class code below is moved unchanged from the training scripts, so the
trained weights in saved_models/ load into these classes without any change.
"""

import torch
import torch.nn as nn


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


# Name used in the pipeline and the GUI -> network class, and its saved weights.
ARCHITECTURES = {
    "cnn": SimpleCNN,
    "mlp": SimpleMLP,
    "lenet5": LeNet5,
    "resnet": SmallResNet,
}
WEIGHT_FILES = {
    "cnn": "cnn_model.pth",
    "mlp": "mlp_model.pth",
    "lenet5": "lenet5_model.pth",
    "resnet": "resnet_model.pth",
}
