# Task 2 - Comparison of segmentation networks

Test set: 1,500 numbers (2-6 digits) built from MNIST test digits, 300 per condition.

## 1. Pixel-level metrics (%)

| Model | Pixel acc. | Mean class acc. | mIoU | fwIoU | Mean Dice | IoU inside | IoU border | Border precision | Border recall | Border Dice |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | 99.7 | 96.0 | 84.6 | 99.5 | 90.3 | 97.2 | 56.6 | 60.4 | 89.9 | 72.3 |
| ResNet-18 (FCN) | 99.6 | 96.1 | 84.2 | 99.4 | 90.0 | 96.8 | 55.8 | 59.2 | 90.7 | 71.7 |
| MobileNetV3 (LR-ASPP) | 99.4 | 90.6 | 82.8 | 99.0 | 89.1 | 94.9 | 53.8 | 65.1 | 75.6 | 70.0 |
| FC-DenseNet | 99.7 | 95.5 | 84.7 | 99.4 | 90.4 | 96.8 | 57.5 | 62.1 | 88.5 | 73.0 |

## 2. Instance-level and task-level metrics (%)

| Model | Digit precision | Digit recall | Digit F1 (RQ) | SQ | PQ | Boundary F1 | Count acc. | Over-seg. | Under-seg. | Number acc. (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | 98.4 | 98.7 | 98.5 | 97.1 | 95.7 | 98.9 | 93.6 | 3.5 | 2.9 | 85.6 (83.7-87.3) |
| ResNet-18 (FCN) | 98.2 | 98.6 | 98.4 | 97.0 | 95.4 | 98.8 | 92.9 | 4.2 | 2.9 | 85.1 (83.2-86.8) |
| MobileNetV3 (LR-ASPP) | 98.3 | 97.6 | 98.0 | 96.9 | 95.0 | 98.7 | 92.5 | 2.3 | 5.2 | 84.7 (82.8-86.5) |
| FC-DenseNet | 98.4 | 98.3 | 98.3 | 97.0 | 95.4 | 98.8 | 93.2 | 3.2 | 3.6 | 85.9 (84.0-87.5) |
| Rule-based hybrid (reference) | 98.4 | 94.4 | 96.4 | 96.5 | 93.0 | 98.3 | 83.8 | 1.7 | 14.5 | 79.6 (77.5-81.6) |

## 3. Efficiency

| Model | Parameters | File size (MB) | GFLOPs per image | CPU ms per image | Training time (min) |
|---|---|---|---|---|---|
| U-Net | 121,051 | 0.52 | 0.217 | 11.5 | 9.7 |
| ResNet-18 (FCN) | 11,179,404 | 44.80 | 3.555 | 28.2 | 42.1 |
| MobileNetV3 (LR-ASPP) | 3,218,478 | 13.08 | 0.739 | 29.4 | 43.2 |
| FC-DenseNet | 167,155 | 0.74 | 0.678 | 30.3 | 56.4 |

## 4. Robustness: PQ and number accuracy per condition (%)

| Model | Well spaced PQ / num. | Close PQ / num. | Touching PQ / num. | Noisy PQ / num. | Paper photo PQ / num. |
|---|---|---|---|---|---|
| U-Net | 99.9 / 99.3 | 99.2 / 95.0 | 84.1 / 40.0 | 96.1 / 96.3 | 99.3 / 97.3 |
| ResNet-18 (FCN) | 99.8 / 99.3 | 98.9 / 95.0 | 83.5 / 39.0 | 95.9 / 95.0 | 99.2 / 97.0 |
| MobileNetV3 (LR-ASPP) | 99.8 / 98.7 | 98.9 / 94.7 | 80.3 / 36.3 | 96.4 / 97.3 | 99.3 / 96.7 |
| FC-DenseNet | 99.7 / 98.7 | 99.2 / 96.0 | 83.1 / 41.7 | 96.1 / 96.3 | 99.1 / 96.7 |
| Rule-based hybrid (reference) | 99.4 / 97.3 | 98.5 / 93.0 | 67.2 / 15.3 | 97.9 / 95.7 | 99.3 / 96.7 |
