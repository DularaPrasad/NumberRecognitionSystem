"""
check_task2.py  -  quick self-check that the Task 2 code works properly
=======================================================================
Run it inside Task2_Segmentation with the venv switched on:

    python check_task2.py

It runs 10 checks and prints PASS or FAIL for each one, then a summary.
It takes about 30-60 seconds. If everything says PASS, Task 2 is working.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, fn):
    """Run one check. fn() returns (ok, detail)."""
    start = time.time()
    try:
        ok, detail = fn()
    except Exception as e:                      # any crash counts as FAIL
        ok, detail = False, f"error: {type(e).__name__}: {e}"
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  -  {detail}  ({time.time() - start:.1f}s)")


# ---------------------------------------------------------------- the checks
def libraries():
    import cv2, numpy, scipy, torch  # noqa: F401
    return True, f"OpenCV {cv2.__version__}, NumPy {numpy.__version__}, PyTorch {torch.__version__}"


def mnist():
    from mnist_loader import load_mnist
    imgs, labels = load_mnist(train=False)
    return imgs.shape == (10000, 28, 28), f"{len(imgs)} test digits loaded"


def model_file():
    from nn_segment import load_model
    from seg_models import MODEL_NAMES
    sizes = []
    for name in MODEL_NAMES:
        net = load_model(name)
        sizes.append(f"{name} {sum(p.numel() for p in net.parameters()):,}")
    return True, "all 4 trained networks loaded (" + ", ".join(sizes) + " parameters)"


def _test_numbers(condition, n, seed):
    """Build n numbers from MNIST TEST digits; return list of (image, owner, digit count)."""
    import numpy as np
    from mnist_loader import load_mnist
    from number_builder import build_number
    from compare_methods import CONDITIONS
    imgs, _ = load_mnist(train=False)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(2, 7))
        img, owner = build_number([imgs[i] for i in rng.integers(0, len(imgs), k)], rng=rng, **CONDITIONS[condition])
        out.append((img, owner, k))
    return out


def _accuracy(method, tests):
    from compare_methods import score
    from segmentation import segment_number
    ok = 0
    for img, owner, k in tests:
        segs, binary = segment_number(img, method=method, return_binary=True)
        ok += score(segs, binary, owner, k)[0]
    return 100 * ok / len(tests)


def spaced_numbers():
    acc = _accuracy("unet", _test_numbers("Well spaced", 40, seed=1))
    return acc >= 90, f"U-Net segmented {acc:.0f}% of 40 well-spaced numbers perfectly (need >= 90%)"


def touching_numbers():
    tests = _test_numbers("Touching", 60, seed=2)
    unet, hybrid = _accuracy("unet", tests), _accuracy("hybrid", tests)
    return unet > hybrid, f"touching digits: U-Net {unet:.0f}% vs rule-based hybrid {hybrid:.0f}% (U-Net should be higher)"


def output_format():
    from segmentation import segment_number
    img, _, _ = _test_numbers("Well spaced", 1, seed=5)[0]
    segs = segment_number(img, method="unet")
    xs = [s.box[0] for s in segs]
    good = len(segs) > 0 and all(s.digit28.shape == (28, 28) and s.digit28.dtype.name == "uint8" for s in segs)
    ordered = xs == sorted(xs)
    return good and ordered, f"{len(segs)} digits returned, each 28x28 uint8, in left-to-right order"


def own_drawing():
    import cv2
    from segmentation import segment_number
    path = os.path.join(HERE, "my_number.png")
    if not os.path.exists(path):
        return True, "my_number.png not found - skipped"
    segs = segment_number(cv2.imread(path), method="unet")
    return len(segs) == 4, f"my_number.png (6969): found {len(segs)} digits (expected 4)"


def empty_image():
    import numpy as np
    from segmentation import segment_number
    segs = segment_number(np.full((60, 200), 255, np.uint8), method="unet")
    return segs == [], "blank white image returns no digits and does not crash"


def all_networks_segment():
    from seg_models import MODEL_NAMES
    tests = _test_numbers("Well spaced", 20, seed=6)
    accs = {name: _accuracy(name, tests) for name in MODEL_NAMES}
    ok = all(a >= 80 for a in accs.values())
    return ok, "well-spaced accuracy: " + ", ".join(f"{k} {v:.0f}%" for k, v in accs.items()) + " (each needs >= 80%)"


def all_methods_run():
    from segmentation import METHODS, segment_number
    img, _, _ = _test_numbers("Close", 1, seed=4)[0]
    for m in METHODS:
        segment_number(img, method=m)
    return True, f"all {len(METHODS)} methods run: {', '.join(METHODS)}"


if __name__ == "__main__":
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    print("Task 2 self-check\n" + "-" * 60)
    check("1. Libraries installed", libraries)
    check("2. MNIST loads", mnist)
    check("3. All 4 trained networks load", model_file)
    check("4. Well-spaced numbers", spaced_numbers)
    check("5. Touching numbers (U-Net vs rule-based)", touching_numbers)
    check("6. Output format for Task 3", output_format)
    check("7. Own hand-drawn number", own_drawing)
    check("8. Blank image handled", empty_image)
    check("9. All methods run", all_methods_run)
    check("10. All 4 networks segment correctly", all_networks_segment)
    print("-" * 60)
    passed = sum(results)
    print(f"{passed} of {len(results)} checks passed.", "Task 2 is working." if passed == len(results)
          else "Send a screenshot of the FAIL lines to get them fixed.")
