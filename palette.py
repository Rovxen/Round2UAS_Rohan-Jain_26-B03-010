"""
palette.py — turn a JPEG-noisy image into clean class labels.

The input images are drawings saved as JPEG. JPEG compression smears
colours, so a drawing that should contain ~10 flat colours actually
contains thousands of near-duplicates. Exact colour matching fails.

Instead we list the true colours, then relabel every pixel as whichever
listed colour it sits closest to. Edge and compression noise gets
absorbed into whichever region it is nearest.
"""

import numpy as np

# Each class maps to the raw RGB values that belong to it.
# A class can own several colours - light green appears both as plain
# green and as a paler variant near the large ellipses.
CLASS_COLOURS = {
    "light_green": [(84, 215, 0), (97, 194, 55)],
    "mid_green":   [(76, 137, 0)],
    "dark_green":  [(47, 82, 0)],
    "blue":        [(93, 23, 235)],
    "black":       [(0, 0, 0)],
    "orange":      [(255, 116, 31)],
    "purple":      [(203, 107, 230)],
    "red":         [(251, 2, 23)],
    "yellow":      [(255, 222, 89)],
    "white":       [(255, 255, 255)],
}

# Class names in a fixed order. A pixel's label is an index into this
# list, so the order must stay stable once you save any output.
CLASSES = list(CLASS_COLOURS.keys())
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Flatten the dictionary into two parallel arrays:
#   REF_COLOURS[i] is a raw RGB value
#   REF_LABELS[i]  is the class that colour belongs to
_refs, _labels = [], []
for name, colours in CLASS_COLOURS.items():
    for c in colours:
        _refs.append(c)
        _labels.append(CLASS_INDEX[name])

REF_COLOURS = np.array(_refs, dtype=np.int32)     # (R, 3)
REF_LABELS = np.array(_labels, dtype=np.uint8)    # (R,)

# Terrain properties, used later for path cost and elevation reporting.
SPEED = {"light_green": 20.0, "mid_green": 15.0, "dark_green": 10.0}
LEVEL = {"light_green": 0, "mid_green": 1, "dark_green": 2}

CASUALTY_COLOURS = ("red", "yellow", "white")
SEVERITY = {"red": 3, "yellow": 2, "white": 1}


def snap(img):
    """Relabel every pixel as its nearest reference colour's class.

    img     : (H, W, 3) uint8 array in RGB order
    returns : (H, W) uint8 array of class indices into CLASSES
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {img.shape}")

    h, w, _ = img.shape

    # Cast to int32 before subtracting. In uint8, 84 - 215 wraps around
    # to 125 instead of going negative, which would silently corrupt
    # every distance we compute.
    flat = img.reshape(-1, 1, 3).astype(np.int32)          # (N, 1, 3)

    # Squared distance from each pixel to each reference colour.
    # Broadcasting (N,1,3) against (1,R,3) gives (N,R,3), summed to (N,R).
    # We skip the square root: the nearest colour by squared distance is
    # the same one as by true distance, so it would be wasted work.
    dist = ((flat - REF_COLOURS.reshape(1, -1, 3)) ** 2).sum(axis=2)

    nearest = dist.argmin(axis=1)                           # (N,)
    return REF_LABELS[nearest].reshape(h, w)


def snap_in_chunks(img, rows=64):
    """Same as snap(), but processes the image a few rows at a time.

    snap() builds an (N, R) distance array. For a 1280x720 image with
    12 reference colours that is about 44 MB - fine here, but it grows
    with image size. Use this version if you hit memory trouble.
    """
    h, w, _ = img.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in range(0, h, rows):
        out[y:y + rows] = snap(img[y:y + rows])
    return out


def to_rgb(labels):
    """Rebuild a viewable image from a label array.

    Each class is drawn in its first listed colour. Use this to eyeball
    whether snapping produced something that looks like the original.
    """
    display = np.array([CLASS_COLOURS[name][0] for name in CLASSES],
                       dtype=np.uint8)
    return display[labels]


def summarise(labels):
    """Print how many pixels landed in each class.

    A class with zero pixels means either that feature is absent from
    this image, or its colour is missing from CLASS_COLOURS.
    """
    total = labels.size
    for i, name in enumerate(CLASSES):
        n = int((labels == i).sum())
        print(f"{name:12s} {n:8d}  {100 * n / total:6.2f}%")