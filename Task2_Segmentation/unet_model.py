"""
unet_model.py  -  the U-Net neural network (PyTorch)
====================================================
U-Net is a well-known convolutional neural network for image SEGMENTATION
(Ronneberger et al., 2015). It gives a prediction for EVERY PIXEL.

It is shaped like the letter U:

   input 1 x 64 x 192
     |  [enc1]  8 channels  -------------------------------  [dec1] -> 3 scores per pixel
     |     v  (halve size)                                  ^ (double size)
     |  [enc2] 16 channels  ----------------------  [dec2]
     |     v                                       ^
     |  [enc3] 32 channels  -------------  [dec3]
     |     v                              ^
     +---------> [bottleneck] 64 channels

  * The LEFT side (encoder) shrinks the image and learns WHAT is there
    (strokes, loops, the shape of a digit).
  * The RIGHT side (decoder) grows it back and decides WHERE, pixel by pixel.
  * The horizontal "skip connections" copy fine detail from left to right,
    so thin pen strokes and the exact contact point between digits are not lost.

The output has 3 numbers ("scores") per pixel, one for each class:
background, inside a digit, border between digits. The biggest score wins.
"""

import torch
import torch.nn as nn

N_CLASSES = 3


class DoubleConv(nn.Module):
    """Two rounds of: 3x3 convolution -> batch normalisation -> ReLU.
    A convolution slides a small 3x3 filter over the image to detect patterns."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base=8, n_classes=N_CLASSES):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = DoubleConv(1, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)                       # halves width and height
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)  # doubles width and height
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, n_classes, 1)           # 3 scores per pixel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # skip connection
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    net = UNet()
    n_params = sum(p.numel() for p in net.parameters())
    out = net(torch.zeros(2, 1, 64, 192))
    print(f"U-Net with {n_params:,} learnable numbers (parameters)")
    print("input (2, 1, 64, 192) -> output", tuple(out.shape))
