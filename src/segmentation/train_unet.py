"""
train_unet.py  -  kept for convenience: trains the U-Net.
All four networks are now trained with train_seg.py, for example:
    python train_seg.py --model unet
    python train_seg.py --model all
"""

import sys

from train_seg import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--model", "unet"] + sys.argv[1:]
    main()
