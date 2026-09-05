"""
pipeline.py run every stage on a single image.

Knows nothing about folders or rankings. Takes one image path,
returns one dictionary of results.
"""

import cv2
import numpy as np

from palette import snap
from mask import build_mask
from casualties import find_casualties, find_marker
from pathfind import build_grid, to_small, nearest_free, all_pairs
from optimise import plan
from output import (build_path, snap_path_to_points, path_time,
                    score_breakdown, draw_path)


def load(image_path):
    """Read an image as RGB, failing loudly if the file is missing."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"could not read {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process(image_path, mask_out=None, path_out=None, verbose=True):
    """Run the whole pipeline on one image.

    mask_out / path_out : where to save the two output pictures.
                          Pass None to skip saving.
    """
    img = load(image_path)

    # 1. Clean the JPEG noise into flat class labels.
    labels = snap(img)

    # 2. Traversable / non-traversable mask.
    mask = build_mask(labels)
    if mask_out is not None:
        cv2.imwrite(str(mask_out), mask)

    # 3. Find the markers and the casualties.
    start = find_marker(labels, "orange")
    goal = find_marker(labels, "purple")
    people = find_casualties(labels)

    # 4. Distances between every pair of interesting points.
    #    Point 0 is the start, the last point is the goal.
    free, dist_cost, time_cost = build_grid(labels, mask)
    points_xy = [start] + [(c["x"], c["y"]) for c in people] + [goal]
    points_rc = [nearest_free(free, to_small(p)) for p in points_xy]
    D, T, routes = all_pairs(free, dist_cost, time_cost, points_rc)

    # 5. Decide the visiting order.
    order, score, straight, priority = plan(points_xy, people, D, verbose)

    # 6. Stitch the chosen legs into one pixel path.
    path = build_path(order, routes, len(points_xy))
    path = snap_path_to_points(path, order, points_xy, len(points_xy))

    # 7. Time it, and draw it.
    seconds, pixels = path_time(path, labels)
    if path_out is not None:
        draw_path(img, path, order, points_xy, people, path_out)

    return {
        "image": str(image_path),
        "size": {"width": img.shape[1], "height": img.shape[0]},
        "start": list(start),
        "goal": list(goal),
        "num_casualties": len(people),
        "casualties": people,
        "visit_order": [list(points_xy[c]) for c in order],
        "casualty_scores": score_breakdown(order, D, straight, priority, people),
        "total_path_score": round(float(score), 4),
        "total_distance_px": round(float(pixels), 2),
        "total_time_s": round(float(seconds), 2),
        "path": [[int(x), int(y)] for x, y in path],
        
    }
    seconds, pixels = path_time(path, labels)
    if path_out is not None:
        draw_path(img, path, order, points_xy, people, path_out)