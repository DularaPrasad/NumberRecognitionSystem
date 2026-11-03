# Handwritten Number Recognition System (HNRS)

Group project for **COS30018 Intelligent Systems** (Option B), Swinburne University of Technology.
Tutor: Hy Nguyen.

HNRS reads a handwritten number (for example `4071`) from a photo, a scan, or a picture built from single-digit
images. It finds each digit, reads each digit with a machine learning model, and gives back the number as text.
It has a graphical interface, and every stage can be swapped for a different technique so that the techniques
can be compared.

This is the only README for the whole project. The full discussion of the methods and results is in the project
report.

## Contents

1. [Team](#team) 2. [Quick start](#quick-start) 3. [Using the GUI](#using-the-gui)
4. [How the system works](#how-the-system-works) 5. [Results at a glance](#results-at-a-glance)
6. [Command line and Python use](#command-line-and-python-use) 7. [Checking that everything works](#checking-that-everything-works)
8. [Evaluating the system](#evaluating-the-system) 9. [Retraining the models](#retraining-the-models)
10. [Repository structure](#repository-structure) 11. [Troubleshooting](#troubleshooting)
12. [Limitations](#limitations) 13. [Research extension](#research-extension) 14. [References](#references)

## Team

| Member | Responsibility | Folders |
|---|---|---|
| Declan McNamara | Task 1: image preprocessing. The GUI, including both ways of getting an input image. | `src/preprocessing`, `src/gui` |
| Charith Jayashan | Task 2: image segmentation (rule-based methods and four trained networks) | `src/segmentation` |
| Dulara Prasad | Task 3: digit recognition models. Task 4: integration and evaluation. | `src/models`, `src/integration`, `src/evaluation` |

## Quick start

You need **Python 3.12** and **Git**. On Windows:

```
git clone https://github.com/DularaPrasad/NumberRecognitionSystem.git
cd NumberRecognitionSystem
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\gui\app.py
```

- You must see `(venv)` at the start of the command line before you run anything. If you do not, run
  `venv\Scripts\activate` again.
- The trained models are included in the repository (`saved_models/` for the digit models and
  `src/segmentation/models/` for the segmentation networks), so nothing has to be trained before use.
- Reading images with the GUI and the pipeline does not need the MNIST dataset. The training and evaluation scripts,
  and the GUI's "Create sample folder from MNIST" button, download MNIST the first time they run (internet needed once).
- A GPU is optional. The code uses an NVIDIA GPU if PyTorch can see one and the CPU otherwise.
- On macOS or Linux use forward slashes (`src/gui/app.py`) and `source venv/bin/activate`.

## Using the GUI

```
python src\gui\app.py
```

**Pipeline settings (top).** Choose the digit model (`cnn`, `mlp`, `lenet5` or `resnet`), the segmentation network and
the thresholding method (`adaptive` or `otsu`). Changing a setting re-reads the image that is on screen, which makes
it easy to compare methods on the same picture. The defaults are `resnet`, U-Net and `adaptive`. The GUI offers the
four trained segmentation networks (U-Net, ResNet-18 FCN, MobileNetV3 LR-ASPP and FC-DenseNet). The rule-based
methods are used by the command line and the evaluation scripts.

**Image source: two ways to get an input image** (the brief asks for both).

| Control | What it does |
|---|---|
| **Load Image...** | Opens an image file (jpg, png, bmp or webp), for example a photo or scan of a handwritten number. |
| **Build From Digit Folder...** | Builds the picture of a number automatically from a folder of pictures of single digits, then reads it. See below. |
| **Rebuild (new digits)** | Builds the same number again with new random pictures of each digit. Available after a build. |
| **Save Image...** | Saves the image that is being read (the loaded or built picture, without the boxes) as a PNG or JPG. |
| **View: Annotated / Preprocessed** | *Annotated* shows the image with a green box around each digit and the predicted digit and its confidence above it. *Preprocessed* shows the black and white image that segmentation receives, which shows why a photo went wrong. |

**Build From Digit Folder.** A dialog asks for:

- **Number to build:** 1 to 12 digits (the default is 4071).
- **Digit folder:** a folder with sub-folders named `0` to `9`, each holding pictures of that digit. The first time,
  click **Create sample folder from MNIST**, which fills the default folder with 20 MNIST pictures of every digit.
  This downloads MNIST, so it needs internet once. The default folder is set by `DIGIT_FOLDER` in
  `src/segmentation/number_builder.py`, it is shown in the dialog, and it is not stored in GitHub. **Browse...**
  chooses another folder, for example one with your own digits.
- **Spacing:** *Normal* (a gap of 4 to 10 pixels), *Close* (1 to 3 pixels) or *Touching* (the digits overlap, which is
  the hardest case for segmentation).
- **Photo-style paper background:** dark ink on uneven light paper, like a photo.

A random picture of each digit is used every time, so `7447` uses two different sevens and **Rebuild** gives
different handwriting.

**Results.** The recognised number is shown in large text, with a table of the confidence of every digit. The status
bar at the bottom shows where the image came from, the settings used and the time taken in milliseconds.

## How the system works

```
 image  ->  Task 1: preprocessing  ->  Task 2: segmentation  ->  Task 3: digit model  ->  "4071"
            (src/preprocessing)        (src/segmentation)         (src/models)
            black and white image      one 28x28 image per        predicts 0 to 9 for
            (white ink on black)       digit, left to right       each digit image
```

**Image acquisition.** Load an image file, or build a picture from a folder of single-digit images (the GUI's
*Build From Digit Folder*, which uses `number_builder.py` from Task 2).

**Task 1: preprocessing** (`src/preprocessing/Preprocessing.py`). One small function for each step: grayscale,
3 x 3 median blur (removes noise and keeps the edges sharp), thresholding, deskew (off by default), resize to
28 x 28 and scaling to 0 to 1. `preprocess_photo(image, method=...)` chains them for photos and returns the
full-size black and white image that segmentation needs. Thresholding is `otsu` (one threshold chosen for the
whole image) or `adaptive` (a different threshold for each small area, which copes with shadows). The ink is made
white on black like MNIST: if the median brightness is above 127 the paper is light and the image is inverted.

**Task 2: segmentation** (`src/segmentation`). `segment_number(binary_image, method=...)` cuts the number into
single digits, left to right, and converts each one to the MNIST layout (28 x 28, white digit on black, fitted
into a 20 x 20 box and centred on its centre of mass; thin pen lines are thickened first).

- Eight methods can be chosen: `unet`, `resnet`, `mobilenet`, `fcdensenet` (trained networks) and `hybrid`,
  `projection`, `connected_components`, `contours` (rule-based).
- The rule-based methods work on the ink itself. `projection` counts the ink in every column and cuts at empty
  columns. `connected_components` treats every group of touching ink pixels as one digit. `contours` traces the
  outline of each shape. `hybrid` is connected components plus repair steps (remove dust, join a digit drawn in two
  pieces, attach short detached strokes, and split blobs that are too wide with a shortest-path cut).
- The four networks are a U-Net, a standard torchvision ResNet-18 with an FCN head, a standard torchvision
  MobileNetV3-Large (LR-ASPP) and a small FC-DenseNet. Each one labels **every pixel** as 0 background, 1 inside a
  digit, or 2 border (ink where two digits touch). Each separate patch of "inside" pixels becomes one digit, and the
  border pixels go to the nearest digit, so touching digits are cut where the network saw the contact.
- No image was labelled by hand. `number_builder.py` glues MNIST digits into numbers and records which digit owns
  every pixel, so the correct label of every pixel is known. A fresh set of examples is made every epoch. The GUI
  uses the same builder to make a number from a folder of single digits.
- All four networks were trained with the same settings (`train_seg.py`): 54,000 MNIST training digits for
  training and 6,000 held out for validation, weighted cross-entropy loss (border pixels x 4, because they are only
  about 0.3% of the pixels), Adam (learning rate 0.001, cosine schedule), 10 epochs of 120 batches of 32, seed 0.
  The version with the lowest validation loss is kept. All testing uses numbers built from MNIST **test** digits.

```python
import cv2
from segmentation import segment_number
segments = segment_number(binary_image, method="unet")      # a list, left to right
for s in segments:
    s.box        # (x, y, w, h) of the digit in the image
    s.digit28    # 28x28 uint8 array, white digit on black, ready for the digit model
```

**Task 3: digit recognition** (`src/models`). Four models classify a 28 x 28 digit, all trained on MNIST with the
same settings (10 epochs, batch size 64, Adam, learning rate 0.001, cross-entropy loss, the full 60,000 training
images). The classes are in `architectures.py`, so training and the pipeline use exactly the same code.

| Model | Architecture |
|---|---|
| `mlp` | Fully connected: 784 -> 128 -> 64 -> 10 with ReLU. Every pixel is a separate input. |
| `cnn` | Two 3 x 3 convolution layers (16 and 32 filters) with max pooling, then one linear layer. |
| `lenet5` | Two 5 x 5 convolution layers (6 and 16 filters) with pooling, then 120 -> 84 -> 10. ReLU and max pooling are used so the comparison is fair. |
| `resnet` | A small residual network: a 16-channel convolution, three residual blocks (16, 32 and 64 channels) with batch normalisation, global average pooling and a linear layer. |

A k-nearest-neighbour baseline (`baseline_knn.py`, k = 3, on 5,000 training and 1,000 test images) was used to check
the data loading and to have a number to beat. It is not part of the final comparison.

**Task 4: integration and evaluation** (`src/integration/pipeline.py`, `src/evaluation`). `NumberRecognizer` joins
the three stages and takes the digit model, the segmentation method and the thresholding method as settings. The
GUI, the command line and the evaluation scripts all use it, so they always give the same answers. Models are
loaded once and kept in memory. The result holds the number, each digit with its confidence (the softmax
probability), the 28 x 28 images the predictions came from, the bounding boxes and the black and white image.

## Results at a glance

All numbers come from the saved models in this repository. The test numbers are built from MNIST **test** digits,
which no model saw in training.

**Digit models** (MNIST test set, 10,000 images)

| Model | Parameters | Test accuracy |
|---|---|---|
| MLP | 109,386 | 97.84% |
| Simple CNN | 20,490 | 98.75% (saved model; an earlier identical run scored 98.96%) |
| LeNet-5 | 61,706 | 98.87% |
| Small ResNet | 77,754 | **99.22%** |

**Segmentation networks** (1,500 test numbers of 2 to 6 digits; share of numbers segmented perfectly)

| Method | Parameters | Numbers perfect | CPU time per image |
|---|---|---|---|
| U-Net (default) | 121,051 | 85.6% | 11.5 ms |
| ResNet-18 (FCN) | 11,179,404 | 85.1% | 28.2 ms |
| MobileNetV3 (LR-ASPP) | 3,218,478 | 84.7% | 29.4 ms |
| FC-DenseNet | 167,155 | 85.9% | 30.3 ms |
| Rule-based hybrid (reference) | none | 79.6% | under 1 ms |

The differences between the four networks are within chance. We chose the U-Net because it is the smallest and
fastest. Five conditions were tested (well spaced, close, touching, noisy and a simulated paper photo), and the
networks are clearly better than the rule-based method only on touching digits.

**Whole system** (ResNet digit model, adaptive preprocessing, 100 numbers per column; whole number read exactly right)

| Segmentation | Clean | Shadowed | Touching digits |
|---|---|---|---|
| U-Net | 96% | 92% | 53% |
| ResNet-18 (FCN) | 97% | 89% | 52% |
| MobileNetV3 | 97% | 93% | 43% |
| FC-DenseNet | 93% | 88% | 52% |
| Hybrid (rule-based) | 97% | 92% | 29% |

Adaptive thresholding raised shadowed numbers from 79% (Otsu) to 92% with the U-Net. All of these results use
numbers built from MNIST digits, so accuracy on real photos is expected to be lower.

## Command line and Python use

```
python src\integration\pipeline.py photo.jpg --model resnet --segmenter unet --preprocessing adaptive --save
```

This prints the number and the confidence of each digit. With `--save`, a copy of the image with the boxes drawn
on it is saved in `results/`.

```python
import sys, cv2
sys.path.insert(0, "src/integration")        # pipeline.py adds the other source folders itself
from pipeline import NumberRecognizer

rec = NumberRecognizer(model="resnet", segmenter="unet", preprocessing="adaptive")
result = rec.recognise(cv2.imread("photo.jpg"))
print(result.number, result.confidences)
rec.set_model("cnn")        # switch the digit model without reloading everything
```

## Checking that everything works

| Command | Expected result |
|---|---|
| `python src\evaluation\test_pipeline.py` | 16 of 16 checks pass |
| `python src\evaluation\test_handoff.py` | 9 of 10 pass. The one failure is a known limitation: Otsu thresholding on a shadowed photo. |
| `python src\evaluation\check_saved_models.py` | the four saved digit models reproduce their reported MNIST accuracy |
| `python src\segmentation\check_task2.py` | 10 of 10 checks pass |
| `python src\evaluation\test_expression.py` | 36 of 36 checks pass (the prototype in `src/extension`) |

## Evaluating the system

```
python src\evaluation\evaluate_pipeline.py --synthetic 200 --touching 100 --segmenter all --models resnet --preprocessing adaptive
python src\evaluation\evaluate_pipeline.py --photos TestImages --segmenter all --save-failures
```

- `--synthetic N` builds N clean and N shadowed numbers from MNIST test digits. `--touching N` adds N numbers whose
  digits overlap.
- `--photos FOLDER` scores real photos. **Name each file starting with the true number**, for example
  `4071_a.jpg`. With `--save-failures`, pictures of the wrong results are saved for inspection.
- Results are saved in `results/evaluation/`. Three measures are reported: whole numbers right, digit count right
  (segmentation), and digits right when the count was right (recognition). Together they show which stage made a
  mistake.

Other evaluation scripts:

```
python src\evaluation\evaluate_saved_model.py cnn      per-digit accuracy and confusion matrix of one saved model
python src\evaluation\plot_training_curves.py          training curves of the four digit models
python src\segmentation\evaluate_models.py             full comparison of the four segmentation networks (10 to 15 minutes)
python src\segmentation\demo_segment.py --image my_number.png --method resnet   every segmentation step on one image
```

The segmentation comparison uses five groups of measures: pixel level (IoU, Dice, precision and recall per class,
with the border class the most important), digit level (precision, recall, panoptic quality), count errors
(over- and under-segmentation), whole-number accuracy with a 95% Wilson confidence interval, and efficiency
(parameters, file size, GFLOPs, CPU time).

## Retraining the models

```
python src\models\cnn_model.py          (also mlp_model.py, lenet5_model.py, resnet_model.py)
python src\segmentation\train_seg.py --model all      several hours on a CPU
```

**Warning:** retraining replaces the saved weights, and training is random, so the results change slightly (our CNN
scored 98.96% and 98.75% on two identical runs). The numbers in this README and in the report belong to the weights
in this repository. After any retraining, run `check_saved_models.py` and the evaluation again.

## Repository structure

```
src/
  preprocessing/   Preprocessing.py            Task 1: grayscale, blur, thresholding, deskew, resize
  segmentation/    segmentation.py             Task 2: segment_number(), the four rule-based methods
                   seg_models.py, train_seg.py, nn_segment.py, evaluate_models.py   the four trained networks
                   number_builder.py, unet_data.py, mnist_loader.py                  training data from MNIST;
                                               number_builder.py also builds numbers for the GUI
                   check_task2.py, demo_segment.py, compare_methods.py, models/*.pt
  models/          architectures.py            the four digit models (shared by training and the pipeline)
                   cnn_model.py, mlp_model.py, lenet5_model.py, resnet_model.py, baseline_knn.py
  integration/     pipeline.py                 NumberRecognizer: joins preprocessing, segmentation, recognition
  evaluation/      evaluate_pipeline.py        whole-system evaluation
                   test_pipeline.py, test_handoff.py, test_expression.py,
                   check_saved_models.py, evaluate_saved_model.py, plot_training_curves.py
  gui/             app.py                      the Tkinter GUI (load an image, or build a number from a digit folder)
  exploration/     load_mnist.py               downloads MNIST and shows sample digits
  extension/       expression.py               unconnected prototype for the maths extension (see below)
saved_models/      trained digit models (.pth)
results/           plots, confusion matrices, training curves, evaluation tables (CSV)
TestImages/        photos of handwritten numbers for testing
requirements.txt
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` or `'torch'` | The virtual environment is not active. Run `venv\Scripts\activate` and check for `(venv)`. |
| `ModuleNotFoundError: No module named 'pipeline'` or `'segmentation'` | You are in the wrong folder. Run the commands from the repository root (the folder that contains `src`). |
| `Trained model not found ... saved_models` | Pull the latest version from GitHub, or train it with `python src\models\<name>_model.py`. |
| `Trained model not found ... src/segmentation/models` | Pull the latest version from GitHub, or run `python src\segmentation\train_seg.py --model all`. |
| `No module named 'scipy'` | `pip install scipy` (it normally comes with scikit-learn). |
| The GUI's Segmenter list shows only U-Net | Your `src/integration/pipeline.py` is out of date. Pull the latest version. |
| Build From Digit Folder says "The folder has no images for the digit(s)" or "No digit images were found" | The folder needs sub-folders named `0` to `9`, each with pictures of that digit. Click **Create sample folder from MNIST**, or choose another folder with **Browse...**. |
| "Could not create the digit folder" | Creating the sample folder downloads MNIST, so it needs internet once. Check the connection, or choose a folder that already has digits. |
| A photo gives too many or too few digits | Try the other thresholding method, crop closer to the number, use a darker pen and leave gaps between the digits. |
| Every digit is right on its own but the number is wrong | The error is usually in segmentation (see Limitations). Use `--save-failures` and look at the boxes. |
| The GUI does not start and mentions `tkinter` | Install Python from python.org (Tkinter is included on Windows). On Linux: `sudo apt install python3-tk`. |

## Limitations

- The system reads one line of the digits 0 to 9. It does not read letters or symbols, or several lines.
- Almost all testing used numbers built from MNIST digits. Real handwriting and real photos are harder, and the
  results on them are lower.
- Touching or overlapping digits are the weakest point: only about half of the touching test numbers were read
  exactly right.
- The networks were trained on digits about 21 pixels tall. Larger images are resized automatically, but very
  small or very large writing may still cause errors.
- Each model was trained once (the CNN twice), so small differences between close models are not reliable.

## Research extension

We looked at both extension options (handwritten text, and arithmetic expressions). We chose the text option for
the reasons given in the report, but ran out of time to build it, so the submitted system recognises digits only.
`src/extension/expression.py` is a small tested prototype from the maths option (it calculates the answer from
recognised symbols). It is not connected to the system. The report explains the decision.

## References

- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Howard, A. et al. (2019). Searching for MobileNetV3. *ICCV*.
- Huang, G., Liu, Z., van der Maaten, L. and Weinberger, K. (2017). Densely connected convolutional networks. *CVPR*.
- Jégou, S., Drozdzal, M., Vazquez, D., Romero, A. and Bengio, Y. (2017). The one hundred layers tiramisu. *CVPR Workshops*.
- LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), pp. 2278-2324.
- LeCun, Y., Cortes, C. and Burges, C. (2010). *The MNIST database of handwritten digits*. http://yann.lecun.com/exdb/mnist/
- Long, J., Shelhamer, E. and Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *CVPR*.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), pp. 62-66.
- Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
