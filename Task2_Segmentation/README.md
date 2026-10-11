# Task 2 – Image Segmentation (HNRS, COS30018 Option B)

A beginner-friendly guide to what Task 2 is, how the code works, and how to run it, including **training the U-Net neural network**.

---

## Part 1 – What is Task 2? (the big picture)

Our system has to read a handwritten **number** like `4071`. The Task 3 model is trained on **MNIST**, which only contains **single digits** (one digit per 28 × 28 picture). It has never seen a picture with four digits in it.

So someone has to **cut the number picture into single-digit pictures** first. That is **segmentation**, and it is Task 2.

```
  Task 1                  Task 2                             Task 3
 ┌──────────┐   ┌──────────────────────────────┐   ┌──────────────────┐
 │ clean up │──▶│ find each digit, cut it out, │──▶│ model says which │──▶  "4071"
 │ the image│   │ put them in left-to-right    │   │ digit each       │
 └──────────┘   │ order, resize to 28x28       │   │ picture is       │
                └──────────────────────────────┘   └──────────────────┘
```

The brief asks Task 2 to research, compare and select a segmentation technique, then implement it. Our tutor also asked us to **train a computer vision neural network** for this step instead of relying only on rule-based methods.

This folder therefore contains:
- **4 rule-based techniques** (no learning). These are the baseline.
- **A trained U-Net** neural network (the selected technique).
- **An experiment** that scores all five on 1,500 test images.

---

## Part 2 – Key ideas in plain English

**A picture is just a grid of numbers.** A grayscale image is a table where every pixel is a number from 0 (black) to 255 (white). In Python it is a **NumPy array**.

**Binary image.** Every pixel becomes 0 or 255 (black or white). This is **thresholding**. **Otsu's method** chooses the threshold automatically. **Adaptive thresholding** uses a different threshold in each small area, which helps with shadowy photos.

**Ink is white, background is black**, like MNIST. Photos of dark ink on white paper are flipped automatically.

**Bounding box.** The smallest rectangle around a digit: `(x, y, w, h)`.

**MNIST format.** Each digit handed to Task 3 must look like MNIST: thick pen line (thin lines are thickened by `normalise_stroke()`), fitted inside a 20 × 20 box, placed on a 28 × 28 black square, and centred on its centre of mass. `to_mnist_format()` does this.

**Neural network, training, epoch, loss.** A neural network is a function with many adjustable numbers (parameters). **Training** shows it thousands of examples with the correct answer. After each guess, the **loss** measures how wrong it was, and the parameters are nudged to make the next guess less wrong. One full round of training is an **epoch**.

---

## Part 3 – The five techniques

### A – Vertical projection profile (rule-based)
Count the ink in every **column**. Empty columns are gaps between digits.
✅ Simple, fast, perfect when digits are well spaced. ❌ Fails when digits touch or lean over each other.

### B – Connected component labelling (rule-based)
Every group of touching ink pixels becomes one blob, and one blob is one digit.
✅ Works when digits overlap sideways without touching. ❌ Two touching digits become one blob, and a broken digit becomes two blobs.

### C – Contour detection (rule-based)
Trace the outline of each shape with `cv2.findContours` and cut out its rectangle.
✅ Standard OpenCV. ❌ Same problems as B, and the rectangle can include bits of the next digit.

### D – Hybrid (best rule-based method)
Connected components plus repair steps: remove dust, merge a digit drawn in two pieces, attach short detached strokes, and split blobs that are too wide using a **shortest-path cut** (dynamic programming finds the cut through the least ink).

### E – U-Net neural network (trained, SELECTED ✅)
A **U-Net** is a convolutional neural network built for segmentation (Ronneberger et al., 2015). Instead of rules written by hand, it **learns from examples** to label **every pixel** as one of:

| Label | Meaning |
|---|---|
| 0 – background | no ink, or a speck of dust |
| 1 – inside a digit | ink that clearly belongs to one digit |
| 2 – border | ink where two digits touch or overlap |

Each separate patch of "inside a digit" pixels then becomes one digit, and border pixels are given to the nearest digit. So touching digits are cut exactly where the network saw the contact.

```
   input image ──▶ [encoder: shrinks, learns WHAT]  ──▶ bottleneck
                          │ skip connections (keep fine detail) │
   3 labels per pixel ◀── [decoder: grows back, decides WHERE] ◀┘
```

Our U-Net is small (about 121,000 parameters, 8 → 16 → 32 → 64 channels), so it trains on a laptop CPU in about 15–20 minutes.

**Where does the training data come from?** We don't label anything by hand. `number_builder.py` glues MNIST digits into numbers **and records which digit owns every pixel**, so `unet_data.py` can compute the correct label for every pixel automatically. A new set of examples is made for every epoch.

**No cheating:** the U-Net is trained on numbers built from MNIST **training** digits. The experiment tests on numbers built from MNIST **test** digits, which the network has never seen.

---

## Part 4 – What's in this folder

```
Task2_Segmentation/
├── segmentation.py      ★ main entry point: segment_number(image, method=...) + MNIST format
├── unet_model.py        the U-Net network (PyTorch)
├── unet_data.py         makes training examples with per-pixel labels
├── train_unet.py        trains the U-Net and saves models/unet_seg.pt
├── unet_segment.py      uses the trained U-Net to split a number into digits
├── compare_methods.py   the experiment: scores all 5 techniques
├── demo_segment.py      try segmentation on one image and see each step
├── number_builder.py    builds number images from single-digit images
├── mnist_loader.py      reads MNIST into NumPy arrays
├── models/unet_seg.pt   the trained network (about 0.5 MB)
├── results/             CSV tables and figures for the report
├── data/  digits/  outputs/   generated locally, not uploaded to GitHub
└── README.md            this guide
```

---

## Part 5 – Running it, step by step

### Step 1 – Open a terminal and switch the venv on
```
cd "E:\SWIN\Interligent System\Project files\NRS-github\Task2_Segmentation"
& "E:\SWIN\Interligent System\Project files\Task 2\NumberRecognitionSystem-main\venv\Scripts\Activate.ps1"
```
You must see `(venv)` at the start of the line.

### Step 2 – Check MNIST loads
```
python mnist_loader.py
```
The first time, MNIST downloads into `data/` (needs internet).

### Step 3 – Look at the training examples
```
python unet_data.py
```
Open `results/unet_training_examples.png`. On the left is what the network sees, and on the right the correct answer (blue = inside a digit, orange = border).

### Step 4 – Train the U-Net
```
python train_unet.py
```
It prints one line per epoch, for example:
```
epoch  1 | train loss 0.6068 | val loss 0.3215 | ink acc  91.9% | border found  82.5% | 86.8s  <- best so far, saved
...
epoch 12 | train loss 0.0301 | val loss 0.0272 | ink acc  96.4% | border found  93.4% | 85.8s
```
- **loss** going down means the network is learning.
- **val** (validation) numbers are measured on examples the network does not train on, so they show whether it really learned or just memorised.
- The best version is saved to `models/unet_seg.pt`.
- The curve is saved to `results/unet_training_curve.png` (a figure for the report).

A trained model is already included, so you only need to retrain if you change something. `python train_unet.py --epochs 3` is a quick test run.

### Step 5 – Segment a number and see what the network did
```
python demo_segment.py --number 4071 --gap -3
```
Open `outputs/segmented_4071.png`. You will see the input, the binary image, **the U-Net's pixel labels** (blue/orange), the boxes around each digit, and each digit in 28 × 28 format. The digits are also saved in `outputs/digits_4071/` for Task 3.

Other examples:
```
python demo_segment.py --image my_number.png                 (your own drawing)
python demo_segment.py --number 2380 --gap -3 --all          (all 5 techniques)
python demo_segment.py --image photo.jpg --threshold adaptive  (a phone photo)
```

### Step 6 – Run the experiment (for your report)
```
python compare_methods.py
```
About a minute. It creates `results/segmentation_results.csv`, `segmentation_comparison.png` and `segmentation_examples.png`.

### Step 7 – Save to GitHub
```
cd ..
git add Task2_Segmentation
git commit -m "Task 2: trained U-Net segmentation network and comparison"
git push origin main
```

---

## Part 6 – The experiment and what it found

**Test data.** 300 numbers of 2 to 6 digits for each of 5 conditions, built from MNIST **test** digits (1,500 images, about 6,000 digits). Seed 42, so you get the same numbers.

| Condition | What it simulates |
|---|---|
| Well spaced | neat handwriting, clear gaps (4–10 px) |
| Close | small gaps (0–3 px) |
| Touching | digits overlapping by 1–5 px |
| Noisy | 25 random dots of dust |
| Paper photo | dark ink on unevenly lit, grainy paper |

**Scoring.** A digit is correct when one segment holds at least 85 % of its ink and is at least 90 % made of that digit. A number is correct only when the count, every digit and the order are right.

**Results: % of numbers segmented perfectly**

| Condition | Projection | Conn. comp. | Contours | Hybrid | **U-Net** |
|---|---|---|---|---|---|
| Well spaced | **100.0** | 97.0 | 97.0 | 97.3 | 99.3 |
| Close | 77.0 | 82.7 | 82.7 | 93.0 | **94.7** |
| Touching | 0.7 | 5.3 | 3.7 | 15.3 | **41.0** |
| Noisy | 81.7 | 90.3 | 90.3 | 95.7 | **96.0** |
| Paper photo | **99.3** | 97.7 | 97.7 | 96.7 | 97.0 |
| **Average** | 71.7 | 74.6 | 74.3 | 79.6 | **85.6** |
| Digits correct (average) | 76.5 | 80.5 | 80.1 | 85.6 | **93.1** |

Speed on a laptop CPU: rule-based about 0.5–1 ms per image, U-Net about 10–20 ms per image. Both are fast enough for the GUI.

**What this tells us:**
1. **The U-Net has the best average** (85.6 % against 79.6 % for the best rule-based method) and the best digit accuracy (93.1 % against 85.6 %).
2. **The biggest gain is on touching digits**: 41.0 % of numbers against 15.3 %, and 69.3 % of digits against 34.0 %. The rule-based methods can only cut where the ink is thin. The network has learned what digit strokes look like, so it can tell where one digit ends and the next begins.
3. On easy images all methods score 97–100 %. Projection is still slightly better on well spaced and paper-photo numbers, because when there is a clean gap a simple rule is perfect.
4. **Touching digits are still the hardest case.** 59 % of touching numbers are still not perfect. Most failures have the right digit count, but where two strokes really cross, some crossing pixels go to the wrong digit, and the strict 90 % purity rule then fails the digit.

**Selected technique: U-Net**, because it has the highest average accuracy and is the only method that handles touching digits reasonably. The rule-based hybrid remains available as a fast fallback.

---

## Part 7 – Limitations and ideas for the future

- **Synthetic data.** Training and testing use MNIST digits glued together. Real photos of handwritten numbers may look different (paper texture, pen types, slant). Testing on 30–50 real photos from the team, and adding some to training, would make the results stronger.
- **Overlapping strokes.** Where strokes truly cross, pixels can only belong to one digit in our labels, so some ink is always lost. A next step is to predict a separate mask per digit (instance segmentation, e.g. Mask R-CNN) or to use the Task 3 model's confidence to choose between cuts.
- **Scale.** The network was trained on digits about 21 pixels tall. Larger images are resized automatically, but very small or very large writing may still cause errors.
- **One line only.** The code assumes one horizontal line of digits.
- **Shared preprocessing.** `prepare_binary()` does the minimum preprocessing. Agree with the Task 1 teammate whether to use their function instead.

---

## Part 8 – How your teammates use the code

```python
import cv2
from segmentation import segment_number

image = cv2.imread("some_number.png")
digits = segment_number(image, method="unet")   # list, left to right

for d in digits:
    print(d.box)        # (x, y, w, h) in the image
    pic = d.digit28     # 28x28 uint8 array, white digit on black, ready for Task 3
```

Options (useful for the GUI):
- `method="unet"` (trained network) or `"hybrid"`, `"projection"`, `"connected_components"`, `"contours"`
- `threshold="otsu" | "adaptive"` (adaptive for phone photos)

`number_builder.build_number_from_folder("4071")` creates a number image from the digit folder, which covers the brief's image-acquisition option.

---

## Part 9 – If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` / `torch` | The venv isn't on. Activate it and check for `(venv)`. |
| `ModuleNotFoundError: No module named 'segmentation'` | You're in the wrong folder. `cd` into `Task2_Segmentation` first. |
| `Trained model not found ... models/unet_seg.pt` | Run `python train_unet.py`. |
| `No module named 'scipy'` | `pip install scipy` (normally installed with scikit-learn). |
| Training is very slow | Use fewer batches: `python train_unet.py --steps 80`. |
| Your own photo finds too many/too few digits | Try `--threshold adaptive`, crop closer to the number, use a darker pen. |
| `--show` window doesn't appear | Leave out `--show` and open the PNG in `outputs/`. |
