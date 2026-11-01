"""
seg_models.py  -  the FOUR segmentation neural networks compared in Task 2
==========================================================================
Every network takes a black/white number image and outputs 3 scores for
EVERY pixel: background / inside a digit / border between two digits.

  1. "unet"        U-Net (Ronneberger et al., 2015)
                   Encoder-decoder with skip connections. See unet_model.py.

  2. "resnet"      Standard ResNet-18 (He et al., 2016) + FCN head (Long et al., 2015)
                   The encoder is torchvision's ResNet-18, used UNCHANGED
                   (same layers, same residual blocks as the paper). ResNet
                   is a classification network, so - exactly as in the FCN
                   paper - its class-scoring part is replaced by 1x1
                   convolutions that score every position, and the scores
                   from several depths are added and up-sampled back to the
                   image size ("FCN-8s"-style fusion).

  3. "mobilenet"   Standard MobileNetV3-Large (Howard et al., 2019) + LR-ASPP head
                   torchvision's own segmentation model
                   lraspp_mobilenet_v3_large, used UNCHANGED. This is the
                   segmentation design proposed in the MobileNetV3 paper.

  4. "fcdensenet"  FC-DenseNet, "The One Hundred Layers Tiramisu" (Jegou et al., 2017)
                   A segmentation network built from DenseNet blocks
                   (Huang et al., 2017), where every layer receives the
                   outputs of ALL earlier layers in its block. Same design as
                   the paper (dense blocks, transition down/up, only new
                   feature maps up-sampled, more layers in deeper blocks) but
                   with fewer layers (2-3-4 / 4 / 4-3-2, growth rate 8) so it
                   trains on a laptop CPU.

Notes for a fair comparison
  * ResNet and MobileNet were designed for 3-channel colour photos, so the
    grayscale image is copied into 3 channels (the networks are not changed).
  * They shrink the image 32 times; our images are only 64 pixels tall, so
    the input is first enlarged 2x and the prediction is scaled back down.
  * All four are trained from scratch with the same data, loss, optimiser,
    number of epochs and random seed (see train_seg.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from unet_model import UNet

N_CLASSES = 3
MODEL_NAMES = {
    "unet": "U-Net",
    "resnet": "ResNet-18 (FCN)",
    "mobilenet": "MobileNetV3 (LR-ASPP)",
    "fcdensenet": "FC-DenseNet",
}


# ----------------------------------------------------------------------------
# 2. Standard ResNet-18 + FCN head
# ----------------------------------------------------------------------------
class ResNetFCN(nn.Module):
    def __init__(self, n_classes=N_CLASSES, upscale=2):
        super().__init__()
        r = resnet18(weights=None)          # the standard torchvision ResNet-18
        self.upscale = upscale
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # 1/4 size
        self.layer1, self.layer2 = r.layer1, r.layer2                   # 1/4, 1/8
        self.layer3, self.layer4 = r.layer3, r.layer4                   # 1/16, 1/32
        # FCN head: a 1x1 convolution scores every position at 4 depths
        self.score1 = nn.Conv2d(64, n_classes, 1)
        self.score2 = nn.Conv2d(128, n_classes, 1)
        self.score3 = nn.Conv2d(256, n_classes, 1)
        self.score4 = nn.Conv2d(512, n_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.repeat(1, 3, 1, 1)                                         # gray -> 3 channels
        x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        up = lambda s, ref: F.interpolate(s, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        s = self.score4(f4)
        s = up(s, f3) + self.score3(f3)     # fuse deep (what) with shallower (where) scores
        s = up(s, f2) + self.score2(f2)
        s = up(s, f1) + self.score1(f1)
        return F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
# 3. Standard MobileNetV3-Large + LR-ASPP (torchvision, unchanged)
# ----------------------------------------------------------------------------
class MobileNetLRASPP(nn.Module):
    def __init__(self, n_classes=N_CLASSES, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=n_classes)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        out = self.net(x)["out"]
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
# 4. FC-DenseNet (Tiramisu), reduced depth
# ----------------------------------------------------------------------------
class DenseLayer(nn.Sequential):
    """BN -> ReLU -> 3x3 conv producing `growth` new feature maps."""

    def __init__(self, c_in, growth, drop=0.1):
        super().__init__(nn.BatchNorm2d(c_in), nn.ReLU(inplace=True),
                         nn.Conv2d(c_in, growth, 3, padding=1, bias=False), nn.Dropout2d(drop))


class DenseBlock(nn.Module):
    """Every layer sees the concatenation of the block input and all earlier layers."""

    def __init__(self, c_in, growth, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(DenseLayer(c_in + i * growth, growth) for i in range(n_layers))

    def forward(self, x):
        new = []
        for layer in self.layers:
            out = layer(torch.cat([x] + new, 1))
            new.append(out)
        return torch.cat([x] + new, 1), torch.cat(new, 1)   # (all features, only the new ones)


class TransitionDown(nn.Sequential):
    def __init__(self, c):
        super().__init__(nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Conv2d(c, c, 1, bias=False),
                         nn.Dropout2d(0.1), nn.MaxPool2d(2))


class FCDenseNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, first=16, growth=8, down=(2, 3, 4), bottleneck=4, up=(4, 3, 2)):
        super().__init__()
        self.first = nn.Conv2d(1, first, 3, padding=1, bias=False)
        c = first
        self.down_blocks, self.trans_down, skip_c = nn.ModuleList(), nn.ModuleList(), []
        for n in down:
            self.down_blocks.append(DenseBlock(c, growth, n))
            c += n * growth
            skip_c.append(c)
            self.trans_down.append(TransitionDown(c))
        self.bottleneck = DenseBlock(c, growth, bottleneck)
        prev_new = bottleneck * growth
        self.trans_up, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        for n, sc in zip(up, reversed(skip_c)):
            # transition up: up-sample ONLY the new feature maps (as in the paper)
            self.trans_up.append(nn.ConvTranspose2d(prev_new, prev_new, 3, stride=2, padding=1, output_padding=1))
            c = prev_new + sc
            self.up_blocks.append(DenseBlock(c, growth, n))
            prev_new = n * growth
            c_last = c + n * growth
        self.head = nn.Conv2d(c_last, n_classes, 1)

    def forward(self, x):
        x = self.first(x)
        skips = []
        for block, td in zip(self.down_blocks, self.trans_down):
            x, _ = block(x)
            skips.append(x)
            x = td(x)
        _, new = self.bottleneck(x)
        for tu, block, skip in zip(self.trans_up, self.up_blocks, reversed(skips)):
            x = torch.cat([tu(new), skip], 1)
            x, new = block(x)
        return self.head(x)


def build_model(name):
    """Create an (untrained) network by name."""
    if name == "unet":
        return UNet()
    if name == "resnet":
        return ResNetFCN()
    if name == "mobilenet":
        return MobileNetLRASPP()
    if name == "fcdensenet":
        return FCDenseNet()
    raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_NAMES)}")


if __name__ == "__main__":
    x = torch.zeros(2, 1, 64, 192)
    for name in MODEL_NAMES:
        net = build_model(name)
        n = sum(p.numel() for p in net.parameters())
        print(f"{MODEL_NAMES[name]:24s} {n:>11,} parameters  ->  output {tuple(net(x).shape)}")
