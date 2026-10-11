# Task 2 – Image Segmentation (HNRS, COS30018 Option B)

A beginner-friendly guide to what Task 2 is, how the code works, and how to run it step by step.

---

## Part 1 – What is Task 2? (the big picture)

Our system has to read a handwritten **number** like `4071`. But the ML model from Task 3 is trained on **MNIST**, and MNIST only contains **single digits** (one digit per 28 × 28 picture). The model has never seen a picture with four digits in it.

So someone has to **cut the number picture into single-digit pictures** first. That cutting is called **segmentation**, and it's your job.

```
  Task 1                  Task 2 (YOU)                       Task 3
 ┌──────────┐   ┌──────────────────────────────┐   ┌──────────────────┐
 │ clean up │──▶│ find each digit, cut it out, │──▶│ model says which │──▶  "4071"
 │ the image│   │ put them in left-to-right    │   │ digit each       │
 └──────────┘   │ order, resize to 28x28       │   │ picture is       │
                └──────────────────────────────┘   └──────────────────┘
```

The brief says for Task 2 (5 marks) you must:

1. **Research and experiment** with different segmentation techniques
2. **Compare** them
3. **Select** the best one and **implement** it

This folder does all three. It contains 4 techniques, an experiment that scores them on thousands of images, and the selected technique ready for the team to use.

---

## Part 2 – Key ideas in plain English

**A picture is just a grid of numbers.** A grayscale image is a table where every cell (pixel) is a number from 0 (black) to 255 (white). In Python we hold it as a **NumPy array**. `image[y, x]` is the pixel in row `y`, column `x`.

**Binary image.** We turn every pixel into either 0 or 255 (pure black or pure white). This is called **thresholding**: "if the pixel is brighter than T, make it white, otherwise black". **Otsu's method** picks T automatically by looking at the image. **Adaptive thresholding** picks a different T for each small area, which helps with photos that have shadows.

**Ink is white, background is black.** MNIST uses white digits on a black background. If you photograph a number written on white paper, it's the opposite, so the code flips (inverts) it automatically.

**Bounding box.** The smallest rectangle around a digit, written as `(x, y, w, h)`: top-left corner, width and height.

**MNIST format.** For the Task 3 model to work, each digit must look like MNIST: fit inside a 20 × 20 box (keeping its shape), sit on a 28 × 28 black square, and be centred by its "centre of mass" (the average position of its ink). The function `to_mnist_format()` does this.

---

## Part 3 – The four techniques

### Technique A – Vertical projection profile
Count how much ink is in every **column** of the image. Columns with zero ink are gaps between digits.

```
image:     ██  ███     █        column ink:  2 2 0 3 3 3 0 0 1
           ██  █ █     █        gaps are the 0s → cut there
```
✅ Very simple and fast. Perfect when digits are well spaced.
❌ Fails when digits touch, or when a slanted digit leans over its neighbour (no empty column).

### Technique B – Connected component labelling
Give every group of **touching ink pixels** its own label (like flood-filling each blob with a different colour). One blob = one digit.
✅ Works even when digits overlap sideways (as long as they don't touch). Each crop contains only that digit's pixels.
❌ A digit drawn in two pieces (e.g. a "5" whose top bar doesn't join) becomes two blobs. Two touching digits become one blob.

### Technique C – Contour detection
Trace the **outline** of each shape with `cv2.findContours`, then cut out its bounding rectangle. This is the approach used in most online tutorials.
✅ Simple, fast, standard OpenCV.
❌ Same problems as B. Also, the crop is a plain rectangle, so bits of a neighbouring digit can sneak in.

### Technique D – Hybrid (SELECTED ✅)
Connected components (B) plus four repair steps:

| Step | What it fixes | How |
|---|---|---|
| 1. Remove dust | tiny specks | drop blobs with < 6 % of the biggest blob's ink |
| 2. Merge broken pieces | a digit drawn in 2 pieces, one above the other | merge blobs in the same column that sit above/below each other |
| 3. Attach fragments | short detached strokes (e.g. the top bar of a "5") | glue short pieces onto the nearest digit, or drop them if they're far away |
| 4. Split touching digits | two digits joined into one blob | a blob much wider than a digit is cut using a **shortest-path cut** (see below) |

**The shortest-path cut.** Imagine the blob is a field where walking through ink costs 50 points and walking through background costs 1 point. We find the cheapest way to walk from the top to the bottom (moving at most one step left or right per row). That path goes through the thinnest connection between the two digits, and it can **bend** around slanted strokes, which a straight cut can't. This is solved with **dynamic programming**, a classic algorithm idea you can mention in the report.

---

## Part 4 – What's in this folder

```
task2_segmentation/
├── segmentation.py      ★ the main code: 4 techniques + MNIST format
├── number_builder.py    builds number images from a folder of digit images
├── compare_methods.py   the experiment: scores all 4 techniques
├── demo_segment.py      try segmentation on one image and see it
├── mnist_loader.py      reads MNIST into NumPy arrays
├── README.md            this guide
├── data/                MNIST files (downloaded automatically)
├── digits/              folder of single-digit images (made by number_builder)
├── outputs/             results of demo_segment / number_builder
└── results/             results of the experiment (CSV + charts for the report)
```

---

## Part 5 – Running it, step by step

### Step 1 – Put the folder in your project
Copy the whole `task2_segmentation` folder into your project folder (the one with your `venv`), e.g.:
```
E:\SWIN\Interligent System\Project files\Task 2\NumberRecognitionSystem-main\task2_segmentation
```

### Step 2 – Open a terminal and switch the venv on
```
cd "E:\SWIN\Interligent System\Project files\Task 2\NumberRecognitionSystem-main"
venv\Scripts\activate
cd task2_segmentation
```
You must see `(venv)` at the start of the line.

### Step 3 – Check MNIST loads
```
python mnist_loader.py
```
Expected:
```
Images: (10000, 28, 28) uint8
Labels: (10000,) first ten = [7 2 1 0 4 1 4 9 5 9]
```
(The first time, it downloads MNIST into `data/`, which needs internet.)

### Step 4 – Make the folder of single digits
```
python number_builder.py --export
```
This creates `digits/0/`, `digits/1/` … `digits/9/`, each with 20 MNIST digit images. Open the folder and have a look. This is the "folder of images of individual digits" the brief talks about.

### Step 5 – Build a number image
```
python number_builder.py 4071
```
Open `outputs/number_4071.png`. Try also:
```
python number_builder.py 4071 --gap -3       (digits touching)
python number_builder.py 2026 --paper        (dark ink on paper, like a photo)
```

### Step 6 – Segment it and look at the result
```
python demo_segment.py --number 4071
```
Open `outputs/segmented_4071.png`. You'll see 4 rows: the input, the black-and-white version, green boxes around each digit (numbered left to right), and each digit in 28 × 28 MNIST format.
The 28 × 28 digits are also saved as separate files in `outputs/digits_4071/`. **These are what Task 3's model will read.**

Compare all four techniques on a hard (touching) example:
```
python demo_segment.py --number 4071 --gap -3 --all
```

Try your own handwriting: write a number on white paper, take a photo, save it as `my_number.jpg` in this folder, then:
```
python demo_segment.py --image my_number.jpg --threshold adaptive --show
```
(`--show` opens a window. Close it to finish.)

### Step 7 – Run the experiment (for your report)
```
python compare_methods.py
```
It takes a few seconds and creates:
- `results/segmentation_results.csv`: the table of numbers (open in Excel)
- `results/segmentation_comparison.png`: bar chart
- `results/segmentation_examples.png`: every technique on an example of every condition

### Step 8 – Save your work to GitHub
```
cd ..
git add task2_segmentation
git commit -m "Task 2: segmentation techniques, comparison experiment and selected hybrid method"
git push
```
Make sure your `.gitignore` contains `venv/`, so the big environment folder is not uploaded. It's fine to leave out `task2_segmentation/data/` too (MNIST downloads itself).

---

## Part 6 – How the experiment works and what it found

**Test data.** For each of 5 conditions, 300 numbers of 2 to 6 digits are built from real MNIST test digits (1,500 images in total, about 6,000 digits). Each digit gets a random size and a small up/down shift, like real handwriting.

| Condition | What it simulates |
|---|---|
| Well spaced | neat handwriting with clear gaps (4–10 px) |
| Close | small gaps (0–3 px) |
| Touching | digits overlapping by 1–5 px |
| Noisy | 25 random dots of dust |
| Paper photo | dark ink on unevenly lit, grainy paper |

**Scoring.** Because we built the images ourselves, we know exactly which pixel belongs to which digit. A digit counts as correctly segmented when one segment contains **at least 85 % of the digit's ink** and is **at least 90 % made of that digit** (no big bits of a neighbour). A whole number is correct only when the count is right, every digit is right, and they're in the right order.

**Results: % of numbers segmented perfectly** (run with seed 42, so you'll get the same numbers):

| Condition | Projection | Conn. components | Contours | **Hybrid** |
|---|---|---|---|---|
| Well spaced | **100.0** | 97.0 | 97.0 | 97.3 |
| Close | 77.0 | 82.7 | 82.7 | **93.0** |
| Touching | 0.7 | 5.3 | 3.7 | **15.3** |
| Noisy | 81.7 | 90.3 | 90.3 | **95.7** |
| Paper photo | **99.3** | 97.7 | 97.7 | 96.7 |
| **Average** | 71.7 | 74.6 | 74.3 | **79.6** |

Speed: every method takes about 0.1–1 ms per image, so speed isn't a deciding factor.

**What this tells us (use this in your report):**
1. **Projection** is the best when digits are clearly apart, but it collapses as soon as they get close or touch.
2. **Connected components** and **contours** give almost the same results, because both look for separate blobs. Contours' rectangular crops can include parts of neighbours.
3. **Hybrid** is the best overall and the most reliable. Its repair steps give the biggest gains on close (+10 points) and noisy (+5 points) numbers, and it's the only method that handles touching digits at all (3× better). On clean images it gives up about 2–3 points to projection, a small price.
4. **Touching digits remain the hardest problem.** Hybrid correctly separates about a third of the digits there, but only 15 % of whole numbers. When strokes really overlap, no cut can separate them cleanly without knowing what the digits are.

**Selected technique: Hybrid**, because it has the highest average accuracy and no condition where it fails badly (except touching, where all methods struggle).

---

## Part 7 – Limitations and ideas for the future (good for "critical analysis")

- **Touching digits.** A known better approach is **recognition-based segmentation**: try several possible cuts, ask the Task 3 model how confident it is about each piece, and keep the cut with the highest confidence. This needs the trained Task 3 model, so it's a good joint idea for later sprints or the research component.
- **Synthetic test data.** The tests use MNIST digits glued together, not real photos of written numbers. Collecting 30–50 real photos from the team and testing on them would make the evaluation stronger.
- **Fixed thresholds.** Settings such as `SPLIT_RATIO = 1.2` were tuned on the synthetic data. Very wide single digits (a wide "0") can occasionally be split wrongly. These settings are at the top of `segmentation.py`, so the GUI could let the user change them, which the brief says a good project does.
- **One line only.** The code assumes the number is written on one horizontal line. A slanted line or multiple lines would need extra steps (deskewing, line detection).
- **Shared preprocessing.** `prepare_binary()` does the minimum preprocessing segmentation needs. Agree with your Task 1 teammate whether to use their function instead, so the team doesn't have two versions.

---

## Part 8 – How your teammates use your code

For **Task 3** (the model) and **Task 4 / GUI** (integration), they only need one function:

```python
import cv2
from segmentation import segment_number

image = cv2.imread("some_number.png")
digits = segment_number(image, method="hybrid")   # list, left to right

for d in digits:
    print(d.box)        # (x, y, w, h) – where it is in the image
    pic = d.digit28     # 28x28 uint8 NumPy array, white digit on black
    # prediction = model.predict(pic / 255.0 ...)   <- Task 3's part
```

Options they can pass (useful for the GUI):
- `method="projection" | "connected_components" | "contours" | "hybrid"`
- `threshold="otsu" | "adaptive"` (use adaptive for phone photos)

For the GUI's image acquisition, `number_builder.build_number_from_folder("4071")` creates a number image from the digit folder, which covers the brief's "automatic creation of the image of a number from a folder of images" option.

---

## Part 9 – If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (or numpy, etc.) | The venv isn't on. Run `venv\Scripts\activate` and check for `(venv)`. |
| `ModuleNotFoundError: No module named 'segmentation'` | You're in the wrong folder. `cd` into `task2_segmentation` first. |
| `No digit images in 'digits'` | Run `python number_builder.py --export`. |
| MNIST download fails | Check your internet, then run `python mnist_loader.py` again. |
| Your own photo finds too many/too few digits | Try `--threshold adaptive`, crop the photo closer to the number, write with a darker pen, and leave small gaps between digits. |
| `--show` window doesn't appear | Leave out `--show` and open the PNG in `outputs/` instead. |
