"""
segmentation.py  -  TASK 2: IMAGE SEGMENTATION
==============================================
Takes ONE image of a handwritten number (e.g. "4071") and cuts it into
separate images of each digit ("4", "0", "7", "1"), in left-to-right order,
each resized to the 28 x 28 MNIST format so the Task 3 model can read it.

    full number image  -->  [ prepare ]  -->  [ find digits ]  -->  [ clean up ]
                                                                         |
          28x28 digit images (left to right)  <--  [ MNIST format ]  <---+

FOUR TECHNIQUES ARE IMPLEMENTED SO THEY CAN BE COMPARED (the brief asks for this):

  1. "projection"            Vertical projection profile. Count the ink in
                             every column; empty columns are gaps between digits.
  2. "connected_components"  Connected-component labelling. Every group of
                             touching ink pixels is one "blob" = one digit.
  3. "contours"              Contour detection. Trace the outline of each
                             shape and take its bounding rectangle.
  4. "hybrid"  (SELECTED)    Connected components + three repair steps:
                             - merge pieces of a broken digit (e.g. a "5" whose
                               top bar is not joined) and drop stray marks,
                             - remove noise (tiny dots/specks),
                             - split blobs that are too wide (digits that touch).

Run  compare_methods.py  to measure how well each one works.

QUICK USE
    import cv2
    from segmentation import segment_number
    img = cv2.imread("my_number.png")
    digits = segment_number(img, method="hybrid")
    for d in digits:
        print(d.box)           # where the digit is: (x, y, width, height)
        d.digit28              # 28x28 uint8 image, ready for the Task 3 model
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Settings (tuning knobs). Changing these changes how the "hybrid" method
# behaves. They are collected here so the GUI can later let the user edit them.
# --------------------------------------------------------------------------
MIN_INK_ABS = 6        # blobs with fewer ink pixels than this are noise
MIN_INK_REL = 0.10     # ... or with less than 10% of the biggest blob's ink
DUST_REL = 0.06        # first clean-up pass: specks under 6% of the biggest blob
MERGE_X_OVERLAP = 0.70 # merge two blobs if they overlap horizontally this much
MERGE_Y_OVERLAP = 0.50 # ... and do NOT overlap vertically more than this
SPLIT_RATIO = 1.20     # a blob wider than 1.2 x its height is probably 2+ digits
DIGIT_ASPECT = 0.70    # a typical digit is about 0.7 x as wide as it is tall
FRAGMENT_HEIGHT = 0.45 # a piece shorter than 45% of a digit's height is a fragment
FRAGMENT_REACH = 0.35  # glue a fragment on if it is this close (x digit height)
STROKE_TARGET = 0.20   # MNIST pen thickness (line width / digit size measure)
STROKE_MIN = 0.14      # lines thinner than this are thickened to the target
FRAGMENT_MIN_INK = 0.12 # ... and has at least 12% of a normal digit's ink


# ==========================================================================
# 0. DATA STRUCTURE FOR ONE FOUND DIGIT
# ==========================================================================
@dataclass
class Segment:
    """One piece cut out of the number image (hopefully exactly one digit).

    box  : (x, y, w, h) position in the full image. x,y = top-left corner.
    mask : the ink pixels inside that box (uint8 image, ink = 255, rest = 0).
    """

    box: tuple
    mask: np.ndarray
    _digit28: np.ndarray = field(default=None, repr=False)

    @property
    def ink(self):
        """How many ink pixels this segment has (a measure of its 'size')."""
        return int(np.count_nonzero(self.mask))

    @property
    def digit28(self):
        """The segment converted to MNIST format (28 x 28). Computed once."""
        if self._digit28 is None:
            self._digit28 = to_mnist_format(self.mask)
        return self._digit28


# ==========================================================================
# 1. PREPARE THE IMAGE  (turn any picture into a clean black/white image)
# ==========================================================================
def prepare_binary(image, threshold="otsu"):
    """Convert any input image into a BINARY image: ink = 255 (white),
    background = 0 (black) - the same style as MNIST.

    NOTE: this is the *minimum* preprocessing segmentation needs. If your
    Task 1 teammate has a better preprocessing function, you can call theirs
    first and pass the result in here.

    Args:
        image:     grayscale (H x W) or colour (H x W x 3, BGR) image.
        threshold: "otsu"     - one threshold for the whole image, chosen
                                automatically (good for clean scans/MNIST).
                   "adaptive" - a different threshold for each small area
                                (better for phone photos with shadows).
    """
    # (a) Colour -> grayscale. We only need brightness, not colour.
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # (b) Make sure the ink is WHITE on BLACK, like MNIST.
    #     Most of an image is background, so the median pixel tells us
    #     the background colour. Bright background = paper -> invert it.
    if np.median(gray) > 127:
        gray = 255 - gray

    # (c) Small blur to smooth away pixel noise before thresholding.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # (d) Thresholding: every pixel becomes either 0 or 255.
    if threshold == "adaptive":
        block = max(15, (min(gray.shape) // 4) | 1)  # odd window size
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -10
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ==========================================================================
# 2. TECHNIQUE A: VERTICAL PROJECTION PROFILE
# ==========================================================================
def segment_projection(binary):
    """Cut wherever a column of the image contains no ink at all.

    Idea: add up the ink in every column. Digits = runs of columns with ink,
    gaps = columns with zero ink.  Simple and fast, BUT it fails when digits
    touch or when a slanted digit "leans over" its neighbour (no empty column).
    """
    column_ink = np.count_nonzero(binary, axis=0)  # ink per column
    has_ink = column_ink > 0

    segments = []
    x = 0
    width = binary.shape[1]
    while x < width:
        if not has_ink[x]:
            x += 1
            continue
        # Found the start of a run of ink columns; walk to its end.
        start = x
        while x < width and has_ink[x]:
            x += 1
        strip = binary[:, start:x]
        # Find the top and bottom of the ink inside this strip.
        rows = np.where(np.count_nonzero(strip, axis=1) > 0)[0]
        y0, y1 = rows[0], rows[-1] + 1
        segments.append(Segment((start, y0, x - start, y1 - y0), strip[y0:y1].copy()))
    return segments


# ==========================================================================
# 3. TECHNIQUE B: CONNECTED COMPONENT LABELLING
# ==========================================================================
def segment_connected_components(binary):
    """Give every group of connected ink pixels its own label (blob).

    "Connected" means pixels touch, including diagonally (8-connectivity).
    One blob is usually one digit. Unlike projection, digits that overlap
    horizontally (but don't touch) are still separated, and each crop holds
    ONLY that blob's pixels - no bits of the neighbouring digit.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    segments = []
    for i in range(1, n):  # label 0 is the background, so skip it
        x, y, w, h, _area = stats[i]
        mask = np.where(labels[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
        segments.append(Segment((int(x), int(y), int(w), int(h)), mask))
    return segments


# ==========================================================================
# 4. TECHNIQUE C: CONTOUR DETECTION
# ==========================================================================
def segment_contours(binary):
    """Trace the outer outline (contour) of every shape, then crop its
    bounding rectangle - the classic OpenCV tutorial approach.

    The crop is a plain rectangle, so if a neighbouring digit leans into that
    rectangle, part of the neighbour is included too (a weakness we measure).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        segments.append(Segment((x, y, w, h), binary[y:y + h, x:x + w].copy()))
    return segments


# ==========================================================================
# 5. TECHNIQUE D (SELECTED): HYBRID = CONNECTED COMPONENTS + REPAIRS
# ==========================================================================
def _overlap(a0, a1, b0, b1):
    """Length of the overlap between ranges [a0,a1) and [b0,b1)."""
    return max(0, min(a1, b1) - max(a0, b0))


def _combine(seg_a, seg_b):
    """Join two segments into one (used when a digit was broken in pieces)."""
    ax, ay, aw, ah = seg_a.box
    bx, by, bw, bh = seg_b.box
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for s in (seg_a, seg_b):
        sx, sy, sw, sh = s.box
        region = mask[sy - y0:sy - y0 + sh, sx - x0:sx - x0 + sw]
        np.maximum(region, s.mask, out=region)
    return Segment((x0, y0, x1 - x0, y1 - y0), mask)


def remove_noise(segments, min_abs=MIN_INK_ABS, min_rel=MIN_INK_REL):
    """Throw away blobs that are far too small to be a digit (dust, specks)."""
    if not segments:
        return segments
    biggest = max(s.ink for s in segments)
    limit = max(min_abs, min_rel * biggest)
    return [s for s in segments if s.ink >= limit]


def _typical_height(segments):
    """Typical digit height = median height of the 'big' blobs (ignoring specks)."""
    biggest = max(s.ink for s in segments)
    return float(np.median([s.box[3] for s in segments if s.ink >= 0.3 * biggest]))


def merge_broken_pieces(segments, x_overlap=MERGE_X_OVERLAP, y_overlap=MERGE_Y_OVERLAP):
    """Join blobs that sit ABOVE/BELOW each other in the same column.

    Example: a "5" written with its top bar not touching the body gives two
    blobs, one on top of the other. Two different digits sit SIDE BY SIDE,
    so they overlap a lot vertically - they are not merged.
    Extra safety rules (so that specks of dust are not chained together):
      * the two pieces must be vertically close (gap <= 30% of a digit height)
      * at least one of them must be a big part of a digit (>= 45% of its height)
    """
    if len(segments) < 2:
        return list(segments)
    ref_h = _typical_height(segments)
    segs = list(segments)
    merged = True
    while merged:  # keep going until nothing more can be merged
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                ax, ay, aw, ah = segs[i].box
                bx, by, bw, bh = segs[j].box
                horiz = _overlap(ax, ax + aw, bx, bx + bw) / min(aw, bw)
                vert = _overlap(ay, ay + ah, by, by + bh) / min(ah, bh)
                gap = max(0, max(ay, by) - min(ay + ah, by + bh))
                big_enough = max(ah, bh) >= 0.45 * ref_h
                if (horiz >= x_overlap and vert <= y_overlap
                        and gap <= 0.3 * ref_h and big_enough):
                    segs[i] = _combine(segs[i], segs[j])
                    del segs[j]
                    merged = True
                    break
            if merged:
                break
    return segs


def attach_fragments(segments, small=FRAGMENT_HEIGHT, reach=FRAGMENT_REACH,
                     min_ink=FRAGMENT_MIN_INK):
    """Deal with small, short pieces (much shorter than a normal digit).

    A real digit is never only a few pixels tall, so a short piece is either
    a detached stroke of a nearby digit (e.g. the top bar of a "5" that sits
    slightly to the right) or a dot of noise.
      - if it is close to a digit AND has a reasonable amount of ink,
        glue it onto that digit;
      - if it is far from everything, throw it away as noise.
    """
    if len(segments) < 2:
        return segments
    ref_h = _typical_height(segments)
    median_ink = float(np.median([s.ink for s in segments if s.box[3] >= small * ref_h] or [1]))

    def is_fragment(s):
        return s.box[3] < small * ref_h

    def distance(a, b):
        ax, ay, aw, ah = a.box
        bx, by, bw, bh = b.box
        dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))  # horizontal gap
        dy = max(0, max(ay, by) - min(ay + ah, by + bh))  # vertical gap
        return (dx * dx + dy * dy) ** 0.5

    digits = [s for s in segments if not is_fragment(s)]
    if not digits:
        return segments
    for frag in [s for s in segments if is_fragment(s)]:
        nearest = min(range(len(digits)), key=lambda i: distance(frag, digits[i]))
        big_enough = frag.ink >= min_ink * median_ink  # a real stroke, not a speck
        if big_enough and distance(frag, digits[nearest]) <= reach * ref_h:
            digits[nearest] = _combine(digits[nearest], frag)
        # else: too far from any digit -> noise, simply not kept
    return digits


def _find_cut_path(mask, lo, hi):
    """Find the best top-to-bottom cutting path between columns lo..hi.

    Think of the blob as a field where walking through INK is very expensive
    and walking through background is cheap. We look for the cheapest walk
    from the top row to the bottom row, moving at most one column left or
    right per row. This "shortest path" cut can bend around slanted strokes,
    which a straight vertical cut cannot. (Solved with dynamic programming.)

    Returns: an array path[y] = the column where row y is cut.
    """
    h = mask.shape[0]
    cost = 1.0 + 50.0 * (mask[:, lo:hi + 1] > 0)      # ink costs 50x more
    total = cost.copy()
    step = np.zeros(cost.shape, np.int8)              # which way we came from
    for y in range(1, h):
        prev = total[y - 1]
        left = np.r_[np.inf, prev[:-1]] + 0.5            # small penalty for bending
        right = np.r_[prev[1:], np.inf] + 0.5
        options = np.vstack([left, prev, right])         # rows: from x-1, x, x+1
        best = options.argmin(axis=0)
        total[y] += options[best, np.arange(len(prev))]
        step[y] = best - 1
    # walk back up from the cheapest bottom cell
    path = np.zeros(h, int)
    x = int(total[-1].argmin())
    for y in range(h - 1, -1, -1):
        path[y] = x + lo
        if y > 0:
            x += int(step[y, x])  # move to the column we came from
    return path


def split_touching(segments, ratio=SPLIT_RATIO, aspect=DIGIT_ASPECT):
    """Cut blobs that are too wide to be one digit (two digits touching).

    1. A digit is normally taller than it is wide. If a blob is more than
       `ratio` x as wide as the typical digit height, it probably holds 2+ digits.
    2. Guess how many: n = width / (typical digit width).
    3. Near each expected cut position, find the cheapest cutting path through
       the blob (see _find_cut_path) and separate the pixels along it.
    """
    if not segments:
        return segments
    ref_h = float(np.median([s.box[3] for s in segments]))  # typical height

    result = []
    for s in segments:
        x, y, w, h = s.box
        height = max(h, ref_h)
        if w <= ratio * height:
            result.append(s)  # normal width - keep as is
            continue

        n = max(2, int(round(w / (aspect * height))))  # how many digits inside
        paths = [np.zeros(h, int)]                      # left edge of the blob
        for k in range(1, n):
            expected = k * w / n
            lo = int(max(1, expected - 0.3 * w / n))
            hi = int(min(w - 2, expected + 0.3 * w / n))
            if hi > lo:
                paths.append(_find_cut_path(s.mask, lo, hi))
        paths.append(np.full(h, w))                     # right edge of the blob

        cols = np.arange(w)[None, :]
        for left, right in zip(paths[:-1], paths[1:]):
            keep = (cols >= left[:, None]) & (cols < right[:, None])
            part = np.where(keep, s.mask, 0).astype(np.uint8)
            ys, xs = np.nonzero(part)
            if len(xs) == 0:
                continue
            part = part[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            result.append(Segment((x + int(xs.min()), y + int(ys.min()),
                                   part.shape[1], part.shape[0]), part.copy()))
    return result


def segment_hybrid(binary):
    """The SELECTED technique: connected components + repair steps."""
    segs = segment_connected_components(binary)
    segs = remove_noise(segs, min_rel=DUST_REL)       # drop tiny dust/specks first
    segs = merge_broken_pieces(segs)                  # fix digits broken in pieces
    segs = attach_fragments(segs)                     # glue on / drop short pieces
    segs = remove_noise(segs)                         # drop remaining small specks
    segs = split_touching(segs)                       # separate touching digits
    return segs


# ==========================================================================
# 6. CONVERT ONE DIGIT TO MNIST FORMAT (the hand-off to Task 3)
# ==========================================================================
def stroke_ratio(mask):
    """How thick the pen line is, compared with the size of the digit.

    The "distance transform" gives, for every ink pixel, its distance to the
    nearest background pixel. In the middle of a thick line that distance is
    large, in a thin line it is small - so the average tells us the thickness.
    MNIST digits measure about 0.20 with this formula.
    """
    ink = np.pad((mask > 127).astype(np.uint8), 1)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    if not dist.any():
        return 0.0
    return 4.0 * float(dist[dist > 0].mean()) / max(mask.shape)


def normalise_stroke(mask, target=STROKE_TARGET, minimum=STROKE_MIN):
    """Thicken a digit written with a THIN line so it looks like MNIST.

    MNIST digits were written with a thick pen. A digit drawn with a thin line
    (a fine pen, or a thin brush in Paint) almost disappears when it is shrunk
    to 28 x 28, and the Task 3 model will not recognise it. If the line is
    thinner than `minimum`, we "dilate" it (grow the ink outwards) until it is
    about as thick as an MNIST digit. Digits that are already thick enough are
    returned unchanged.
    """
    ratio = stroke_ratio(mask)
    if ratio == 0 or ratio >= minimum:
        return mask
    grow = int(round((target - ratio) * max(mask.shape)))  # pixels to add
    if grow < 1:
        return mask
    pad = grow  # room for the ink to grow into
    padded = np.pad(mask, pad)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow + 1, grow + 1))
    return cv2.dilate(padded, kernel)


def to_mnist_format(mask):
    """Make a cropped digit look exactly like an MNIST image.

    MNIST rules (from the original dataset description):
      * the pen line is thick (thin lines are thickened first),
      * the digit is resized to fit in a 20 x 20 box, KEEPING its shape
        (a tall "1" stays thin - we do not stretch it),
      * it is placed on a 28 x 28 black canvas,
      * it is shifted so its centre of mass sits in the middle of the canvas.
    A model trained on MNIST only works well if its input follows these rules.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = normalise_stroke(crop)  # thin pen lines are thickened to MNIST style

    # Resize so the LONGER side becomes 20 pixels.
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Put it in the centre of a 28 x 28 canvas.
    canvas = np.zeros((28, 28), np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    # Shift so the centre of mass is at (14, 14).
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28))
    return canvas


# ==========================================================================
# 7. MAIN ENTRY POINT - this is what other team members call
# ==========================================================================
def _network(name):
    """Techniques E-H: the TRAINED neural networks (see seg_models.py and
    nn_segment.py). Imported only when used, so the rule-based methods still
    work without PyTorch."""
    def run(binary):
        from nn_segment import segment_nn
        return segment_nn(binary, name)
    run.__name__ = f"segment_{name}"
    return run


NEURAL_NETWORKS = ("unet", "resnet", "mobilenet", "fcdensenet")


METHODS = {
    "projection": segment_projection,
    "connected_components": segment_connected_components,
    "contours": segment_contours,
    "hybrid": segment_hybrid,
    "unet": _network("unet"),
    "resnet": _network("resnet"),
    "mobilenet": _network("mobilenet"),
    "fcdensenet": _network("fcdensenet"),
}


def segment_number(image, method="hybrid", threshold="otsu", return_binary=False):
    """Split an image of a handwritten number into its digits.

    Args:
        image:   the number image (grayscale or colour, any size).
        method:  one of "projection", "connected_components", "contours",
                 "hybrid" (rule-based), or a trained neural network:
                 "unet", "resnet", "mobilenet", "fcdensenet"
                 (needs models/<name>_seg.pt from train_seg.py).
        threshold: "otsu" or "adaptive" (see prepare_binary).
        return_binary: also return the black/white image (for display).

    Returns:
        list of Segment objects sorted LEFT to RIGHT.
        Use  seg.digit28  to get the 28x28 image for the classifier.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")
    binary = prepare_binary(image, threshold)
    segments = METHODS[method](binary)
    if method != "hybrid" and method not in NEURAL_NETWORKS:
        # the basic methods get a simple noise filter so the comparison is fair
        segments = remove_noise(segments)
    segments.sort(key=lambda s: s.box[0])  # left -> right = reading order
    return (segments, binary) if return_binary else segments


def draw_boxes(image, segments):
    """Return a colour copy of the image with a numbered box around each digit."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    # Line and text size grow with the image, so they stay visible on big photos.
    size = max(vis.shape[:2])
    thick = max(1, round(size / 300))
    font = max(0.35, size / 700)
    for i, s in enumerate(segments):
        x, y, w, h = s.box
        cv2.rectangle(vis, (x - thick, y - thick), (x + w, y + h), (0, 200, 0), thick)
        cv2.putText(vis, str(i + 1), (x, max(int(30 * font), y - 3 * thick)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thick, cv2.LINE_AA)
    return vis
