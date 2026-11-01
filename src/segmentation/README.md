# Task 2 – Image Segmentation with Neural Networks (HNRS, COS30018 Option B)

**Owner:** Charith Jayashan · **Folder:** `src/segmentation/`

Task 2 takes the black-and-white image of a whole handwritten number (from Task 1) and splits it into one 28 × 28 image per digit, in left-to-right order, ready for the Task 3 digit models.

Following our tutor's guidance, the segmentation is done by **trained computer-vision neural networks**. Four standard architectures were trained in the same way and compared with a full set of segmentation metrics.

---

## 1. How Task 2 fits the whole system

```
 image ─▶ Task 1: preprocess_photo() ─▶ Task 2: segment_number() ─▶ Task 3: digit model ─▶ "4071"
          (src/preprocessing)            (src/segmentation)          (src/models)
                 binary image            list of digits, each with
                 (white ink on black)    .box  and  .digit28 (28x28 uint8)
```

The team pipeline (`src/integration/pipeline.py`), the GUI (`src/gui/app.py`) and the evaluation scripts (`src/evaluation/`) all call Task 2 through one function:

```python
from segmentation import segment_number
segments = segment_number(binary, method="unet")   # or "resnet", "mobilenet", "fcdensenet"
for s in segments:
    s.box        # (x, y, w, h) of the digit in the image
    s.digit28    # 28x28 uint8, white digit on black, centred like MNIST
```

---

## 2. The four neural networks

Every network outputs a label for **every pixel**:

| Label | Meaning |
|---|---|
| 0 – background | no ink, or a speck of noise |
| 1 – inside a digit | ink that clearly belongs to one digit |
| 2 – border | ink where two digits touch or overlap |

Each separate patch of "inside" pixels then becomes one digit, and the border pixels are given to the nearest digit, so touching digits are cut exactly where the network saw the contact.

| Network | Key idea | Reference |
|---|---|---|
| **U-Net** | Encoder–decoder with skip connections that carry fine detail from the encoder to the decoder | Ronneberger, Fischer & Brox (2015) |
| **ResNet-18 (FCN head)** | The **standard torchvision ResNet-18**, unchanged, used as the encoder. As in the FCN paper, the classification layer is replaced by 1 × 1 convolutions that score every position; scores from four depths are added and up-sampled | He et al. (2016); Long, Shelhamer & Darrell (2015) |
| **MobileNetV3-Large (LR-ASPP)** | torchvision's **standard `lraspp_mobilenet_v3_large`** segmentation model, unchanged. Depthwise-separable convolutions make it light and fast | Howard et al. (2019) |
| **FC-DenseNet** | "The One Hundred Layers Tiramisu": dense blocks where every layer receives the outputs of all earlier layers. Reduced depth (2-3-4 / 4 / 4-3-2 layers, growth rate 8) so it trains on a laptop CPU | Jégou et al. (2017); Huang et al. (2017) |

Notes:
- ResNet and MobileNet were designed for 3-channel colour photos, so the grayscale image is copied into 3 channels; the networks themselves are not modified.
- They shrink the image 32 times and our images are only 64 pixels tall, so their input is enlarged 2× and the prediction is scaled back.

**Training data.** No hand labelling is needed. `number_builder.py` glues MNIST digits into numbers and records which digit owns every pixel, and `unet_data.py` turns that into the correct label for every pixel. A fresh set of examples is generated every epoch.

**Same training for all four** (`train_seg.py`): MNIST training digits (54,000 for training, 6,000 held out for a 600-example validation set), weighted cross-entropy loss (border pixels × 4, because they are only about 0.3 % of pixels), Adam optimiser (learning rate 0.001, cosine schedule), 10 epochs × 120 batches of 32, random seed 0. The version with the lowest validation loss is kept.

**No cheating:** all testing uses numbers built from MNIST **test** digits, which no network saw during training.

---

## 3. Evaluation metrics (and why)

Accuracy alone does not describe a segmentation model well, so `evaluate_models.py` measures five groups of metrics.

| Group | Metrics | What they tell us |
|---|---|---|
| **Pixel level** | IoU (Jaccard) per class, mean IoU, frequency-weighted IoU, Dice/F1 per class, precision and recall per class, pixel accuracy, mean class accuracy | How well each pixel is labelled. The **border class** matters most, because it separates touching digits. Pixel accuracy is reported but is misleading here (about 95 % of pixels are background). |
| **Digit (instance) level** | Digit precision, recall, F1 (= Recognition Quality RQ), Segmentation Quality SQ, **Panoptic Quality PQ = SQ × RQ**, Boundary F1 | Whether each digit is found as exactly one object, and how accurately its shape and outline are cut. Digits are matched one-to-one (Hungarian matching on mask IoU, match if IoU > 0.5), as in the PQ definition. |
| **Count errors** | Digit-count accuracy, over-segmentation rate, under-segmentation rate | Whether a digit is split into pieces or two digits are merged. |
| **Task level** | Number accuracy (whole number correct) with a 95 % Wilson confidence interval | What matters to the user. |
| **Efficiency** | Parameters, file size, GFLOPs per image, CPU time per image, training time | Cost of using the network in the GUI. |

All metrics are also reported for each of the five test conditions (well spaced, close, touching, noisy, paper photo) to show **robustness**.

---

## 4. Results

Test set: **1,500 numbers** (2–6 digits) built from MNIST test digits, 300 for each of five conditions. Full tables: `results/model_comparison.md` and the CSV files in `results/`.

### 4.1 Pixel level (%)

| Network | mIoU | fwIoU | Mean Dice | IoU inside | **IoU border** | Border precision | Border recall | Pixel acc. |
|---|---|---|---|---|---|---|---|---|
| U-Net | 84.6 | 99.5 | 90.3 | **97.2** | 56.6 | 60.4 | 89.9 | 99.7 |
| ResNet-18 (FCN) | 84.2 | 99.4 | 90.0 | 96.8 | 55.8 | 59.2 | **90.7** | 99.6 |
| MobileNetV3 (LR-ASPP) | 82.8 | 99.0 | 89.1 | 94.9 | 53.8 | **65.1** | 75.6 | 99.4 |
| FC-DenseNet | **84.7** | 99.4 | **90.4** | 96.8 | **57.5** | 62.1 | 88.5 | 99.7 |

### 4.2 Digit level and number level (%)

| Network | Digit F1 (RQ) | SQ | **PQ** | Boundary F1 | Count acc. | Over-seg. | Under-seg. | **Number acc.** (95% CI) |
|---|---|---|---|---|---|---|---|---|
| U-Net | **98.5** | **97.1** | **95.7** | **98.9** | **93.6** | 3.5 | **2.9** | 85.6 (83.7–87.3) |
| ResNet-18 (FCN) | 98.4 | 97.0 | 95.4 | 98.8 | 92.9 | 4.2 | **2.9** | 85.1 (83.2–86.8) |
| MobileNetV3 (LR-ASPP) | 98.0 | 96.9 | 95.0 | 98.7 | 92.5 | **2.3** | 5.2 | 84.7 (82.8–86.5) |
| FC-DenseNet | 98.3 | 97.0 | 95.4 | 98.8 | 93.2 | 3.2 | 3.6 | **85.9** (84.0–87.5) |
| *Rule-based hybrid (reference)* | 96.4 | 96.5 | 93.0 | 98.3 | 83.8 | 1.7 | 14.5 | 79.6 (77.5–81.6) |

### 4.3 Efficiency

| Network | Parameters | File size | GFLOPs / image | CPU ms / image | Training time |
|---|---|---|---|---|---|
| U-Net | **121,051** | **0.52 MB** | **0.22** | **11.5** | **9.7 min** |
| ResNet-18 (FCN) | 11,179,404 | 44.80 MB | 3.56 | 28.2 | 42.1 min |
| MobileNetV3 (LR-ASPP) | 3,218,478 | 13.08 MB | 0.74 | 29.4 | 43.2 min |
| FC-DenseNet | 167,155 | 0.74 MB | 0.68 | 30.3 | 56.4 min |

*Times measured on a 2-core CPU; they will be lower on a typical laptop, but the ranking is what matters.*

### 4.4 Robustness: PQ / number accuracy per condition (%)

| Network | Well spaced | Close | **Touching** | Noisy | Paper photo |
|---|---|---|---|---|---|
| U-Net | 99.9 / 99.3 | 99.2 / 95.0 | **84.1** / 40.0 | 96.1 / 96.3 | 99.3 / 97.3 |
| ResNet-18 (FCN) | 99.8 / 99.3 | 98.9 / 95.0 | 83.5 / 39.0 | 95.9 / 95.0 | 99.2 / 97.0 |
| MobileNetV3 (LR-ASPP) | 99.8 / 98.7 | 98.9 / 94.7 | 80.3 / 36.3 | 96.4 / 97.3 | 99.3 / 96.7 |
| FC-DenseNet | 99.7 / 98.7 | 99.2 / 96.0 | 83.1 / **41.7** | 96.1 / 96.3 | 99.1 / 96.7 |
| *Rule-based hybrid* | 99.4 / 97.3 | 98.5 / 93.0 | 67.2 / 15.3 | 97.9 / 95.7 | 99.3 / 96.7 |

Figures: `fig_metric_overview.png`, `fig_by_condition.png`, `fig_efficiency.png`, `fig_training_curves.png`, `fig_pixel_examples.png`.

### 4.5 What the results show

1. **All four networks clearly beat the rule-based method**, mainly on touching digits (36–42 % of numbers fully correct against 15 %, PQ 80–84 against 67) and in under-segmentation (3–5 % against 14.5 %).
2. **The four networks are very close in accuracy.** Number accuracy is 84.7–85.9 % and the 95 % confidence intervals overlap, so on this test set the differences in number accuracy are not statistically significant. The finer metrics separate them more clearly:
   - **FC-DenseNet** has the best pixel-level scores (mIoU 84.7, border IoU 57.5) and the best number accuracy on touching digits (41.7 %). Dense connections reuse features from every earlier layer, which helps at the thin contact points.
   - **U-Net** has the best digit-level scores (digit F1 98.5, PQ 95.7, boundary F1 98.9, count accuracy 93.6). Its skip connections keep the exact outline of each digit.
   - **ResNet-18** is almost the same as U-Net in accuracy, but is about 90 times larger and needs 16 times more computation. Its features come from 1/4 resolution, so very thin borders are harder for it.
   - **MobileNetV3** is the weakest. It finds only 75.6 % of border pixels (lowest border recall), so touching digits are merged more often (under-segmentation 5.2 %). LR-ASPP predicts at a coarse resolution, which suits large objects in photos better than 2-pixel-wide contacts. It also needed the most epochs to start detecting borders (see `fig_training_curves.png`).
3. **Efficiency differs a lot.** U-Net is the smallest, fastest and quickest to train. MobileNetV3 needs fewer FLOPs than ResNet-18 but is not faster on a CPU, because depthwise convolutions use the processor less efficiently. FC-DenseNet has few parameters but is slow, because it concatenates many full-resolution feature maps.
4. Pixel accuracy (99.4–99.7 %) and frequency-weighted IoU (99.0–99.5 %) hardly separate the networks, because about 95 % of pixels are background. This is why IoU per class, border metrics and instance metrics are needed.
5. On noisy images the networks sometimes label specks as "inside a digit" (`fig_pixel_examples.png`). Most are removed by the size filter, but the rule-based method is slightly better there (PQ 97.9 against about 96).

### 4.6 Selected network: U-Net

The U-Net is selected as the default segmenter. Its accuracy is equal to the best network within the confidence interval, it has the best digit-level quality (PQ, digit F1, boundary F1, count accuracy), and it is 2.5 times faster, about 90 times smaller than ResNet-18 and the quickest to train, which keeps the GUI responsive. **FC-DenseNet** is the best alternative when pixel-level border accuracy matters most. All four networks stay selectable in the pipeline and GUI.

---

## 5. Tested with the whole team pipeline

The new Task 2 was placed in the team repository (`src/segmentation/`) and run with Task 1, all four Task 3 digit models, the team pipeline and the team's own tests. **No Task 1 or Task 3 file was changed.** The only change outside `src/segmentation/` is one line in `src/integration/pipeline.py`, adding the new networks to the list of allowed segmenters:

```python
SEGMENTERS = ("unet", "resnet", "mobilenet", "fcdensenet", "hybrid", "projection", "connected_components", "contours")
```

### 5.1 Team tests

| Test | Result |
|---|---|
| `src/segmentation/check_task2.py` (Task 2 self-test, all 4 networks) | **10 / 10 passed** |
| `tests/test_pipeline.py` (team end-to-end tests) | **16 / 16 passed** |
| `tests/test_handoff.py` (Task 1 → Task 2 hand-off) | 9 / 10 passed. The one failure is a shadow photo with Task 1's **Otsu** setting: Otsu turns the shadow into a large block of "ink" before Task 2 sees it. The same photo passes with Task 1's **adaptive** setting, and the result is the same with every segmenter, so this is a Task 1 setting issue, not a segmentation error. |
| `src/integration/pipeline.py my_number.png` | Segmented into the correct 4 digits; read as "6989" (Task 3 reads the curly first 6 as an 8). |

### 5.2 End-to-end accuracy with the team evaluation script

`python src/evaluation/evaluate_pipeline.py --synthetic 200 --segmenter <name>` was run for each segmenter: 200 numbers built from MNIST test digits (100 clean, 100 with a shadow), Task 1 with both settings, and all four Task 3 models. The tables show the percentage of **whole numbers read correctly**.

Using Task 3's ResNet digit model:

| Segmenter | Clean, adaptive | Shadow, adaptive | Clean, Otsu | Shadow, Otsu |
|---|---|---|---|---|
| U-Net | 96 | 92 | 97 | 79 |
| ResNet-18 (FCN) | 97 | 89 | 97 | 79 |
| MobileNetV3 (LR-ASPP) | 97 | 93 | 97 | 81 |
| FC-DenseNet | 93 | 88 | 94 | 79 |
| *Rule-based hybrid* | 97 | 92 | 97 | 79 |

Averaged over both conditions and both Task 1 settings, for each Task 3 model:

| Segmenter | CNN | MLP | LeNet-5 | ResNet | Digit-count acc. |
|---|---|---|---|---|---|
| U-Net | 88.8 | 83.2 | 88.5 | 91.0 | 94.5 |
| ResNet-18 (FCN) | 88.5 | 83.0 | 88.0 | 90.5 | 94.2 |
| MobileNetV3 (LR-ASPP) | 89.8 | 84.0 | 89.8 | 92.0 | 95.8 |
| FC-DenseNet | 86.8 | 80.8 | 86.0 | 88.5 | 92.0 |
| *Rule-based hybrid* | 89.8 | 82.8 | 89.0 | 91.2 | 95.0 |

Per-segmenter CSV files are in `results/evaluation/by_segmenter/` in the team repository.

**What this shows**

- Every network works end to end with Task 1 and every Task 3 model, without errors.
- The team script puts a **3–8 pixel gap between digits**, so its numbers never touch. On this easy case the rule-based method already does well, and all segmenters score within a few points of each other. With only 100 numbers per cell, differences of about 5 points are within normal random variation. The networks' real advantage shows on touching digits (Section 4.4), which this script does not test.
- FC-DenseNet is 2–4 points lower than the others here, which is close to the size of random variation for 100 numbers. On the larger Task 2 test (1,500 numbers, Section 4) it was the most accurate network, so this small drop is noted but was not investigated further.
- The best full system is **Task 1 adaptive → Task 2 U-Net (or MobileNetV3) → Task 3 ResNet**, which reads 92–96 % of the test numbers correctly.

---

## 6. Files

```
src/segmentation/
├── segmentation.py      ★ segment_number(image, method=...) – used by the pipeline and GUI
├── seg_models.py        the 4 networks (U-Net, ResNet-18 FCN, MobileNetV3 LR-ASPP, FC-DenseNet)
├── unet_model.py        the U-Net architecture
├── nn_segment.py        runs a trained network and turns pixel labels into digits
├── unet_data.py         makes training examples with per-pixel labels
├── train_seg.py         trains any of the networks (same settings for all)
├── evaluate_models.py   comprehensive comparison of the 4 networks
├── check_task2.py       10-check self-test
├── demo_segment.py      shows each step on one image
├── compare_methods.py   number accuracy of every method, incl. the rule-based baseline
├── number_builder.py    builds number images from digit images (also for the GUI's image acquisition)
├── mnist_loader.py      reads MNIST
├── unet_segment.py, train_unet.py   kept so older code still works
├── models/              the 4 trained networks (*_seg.pt)
└── results/             tables (CSV, Markdown) and figures for the report
```

---

## 7. How to run (from the repository root, venv active)

```
python src\segmentation\check_task2.py                       10-check self-test (about 1 minute)
python src\segmentation\evaluate_models.py                   full comparison (about 10-15 minutes)
python src\segmentation\demo_segment.py --image src\segmentation\my_number.png --method resnet
python src\segmentation\train_seg.py --model all             retrain all 4 networks (several hours on a CPU)
python src\integration\pipeline.py photo.jpg --segmenter fcdensenet --model resnet --save
```

The trained networks are included in `models/`, so retraining is only needed if something changes.

---

## 8. Limitations

- Training and the main evaluation use numbers built from MNIST digits. Real photos look different (paper texture, pen, slant, lighting); a larger set of real photos would make the results stronger.
- Where two strokes truly cross, each pixel can only belong to one digit in our labels, so some ink is always lost at the crossing. Instance segmentation (a separate mask per digit) would be a next step.
- The networks were trained on digits about 21 pixels tall. Larger images are resized automatically, but very small or very large writing may still cause errors.
- Task 1's Otsu setting turns large shadows into ink, which no segmentation can undo; Task 1's adaptive setting handles shadows.

---

## References

- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Howard, A. et al. (2019). Searching for MobileNetV3. *ICCV*.
- Huang, G., Liu, Z., van der Maaten, L. and Weinberger, K. (2017). Densely connected convolutional networks. *CVPR*.
- Jégou, S., Drozdzal, M., Vazquez, D., Romero, A. and Bengio, Y. (2017). The one hundred layers tiramisu: fully convolutional DenseNets for semantic segmentation. *CVPR Workshops*.
- Kirillov, A., He, K., Girshick, R., Rother, C. and Dollár, P. (2019). Panoptic segmentation. *CVPR*.
- Long, J., Shelhamer, E. and Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *CVPR*.
- Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
- Csurka, G., Larlus, D. and Perronnin, F. (2013). What is a good evaluation measure for semantic segmentation? *BMVC*.
- Everingham, M. et al. (2010). The PASCAL Visual Object Classes (VOC) challenge. *IJCV*.
