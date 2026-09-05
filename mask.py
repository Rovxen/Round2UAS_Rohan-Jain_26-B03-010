"""
mask.py - build the traversable / non-traversable mask.

Traversable (white, 255): all green terrain, plus casualties, start
and destination markers - the rover has to be able to reach those.

Non-traversable (black, 0): black walls and blue regions.

On blue: the task text only names black as an obstacle, but the sample
mask on page 5 of the brief shows the blue ellipses as black blobs.
The figure and the text disagree; we follow the figure.
"""

import numpy as np
from palette import CLASSES, CLASS_INDEX

BLOCKED = ("black", "blue")

TRAVERSABLE = ("light_green", "mid_green", "dark_green",
               "orange", "purple", "red", "yellow", "white")


def build_mask(labels):
    """Return a uint8 mask: 255 where the rover may go, 0 where it may not."""
    blocked = np.zeros(labels.shape, dtype=bool)
    for name in BLOCKED:
        blocked |= (labels == CLASS_INDEX[name])

    # Anything not blocked is traversable. Written this way so a class
    # accidentally missing from both lists still gets a definite answer.
    return np.where(blocked, 0, 255).astype(np.uint8)


def check_coverage(labels):
    """Warn if any class was left out of both lists."""
    named = set(BLOCKED) | set(TRAVERSABLE)
    for name in CLASSES:
        if name not in named:
            n = int((labels == CLASS_INDEX[name]).sum())
            print(f"WARNING: '{name}' ({n} px) is in neither list")