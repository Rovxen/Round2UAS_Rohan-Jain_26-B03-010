"""
output.py - reconstruct the pixel path, time it, and draw it.
"""

import cv2
import numpy as np

from palette import CLASS_INDEX, SPEED
from pathfind import to_full

# Terrain label -> rover speed in pixels per second.
SPEED_BY_LABEL = {CLASS_INDEX[name]: s for name, s in SPEED.items()}


def build_path(order, routes, n_points):
    """Join the per-leg routes into one pixel path, start to goal."""
    legs = [0] + list(order) + [n_points - 1]
    path = []

    for a, b in zip(legs, legs[1:]):
        leg = routes[a][b]
        if leg is None:
            raise ValueError(f"no route between points {a} and {b}")
        # Skip each leg's first cell after the first leg - it repeats
        # the previous leg's last cell.
        cells = leg if not path else leg[1:]
        path.extend(to_full(rc) for rc in cells)

    return path


def snap_path_to_points(path, order, points_xy, n_points):
    """Replace nearby route points with the exact casualty coordinates.

    The route was planned on a shrunken grid, so its points land near
    the casualties rather than exactly on them. This nudges the path
    onto the real coordinates so the drawing lines up.
    """
    targets = [points_xy[0]] + [points_xy[c] for c in order] + [points_xy[n_points - 1]]
    path = list(path)

    for tx, ty in targets:
        d = [(px - tx) ** 2 + (py - ty) ** 2 for px, py in path]
        path[int(np.argmin(d))] = (tx, ty)

    return path


def path_time(path, labels):
    """Return (seconds, pixels) for walking a path, terrain by terrain."""
    h, w = labels.shape
    total_time = 0.0
    total_dist = 0.0

    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        d = float(np.hypot(x2 - x1, y2 - y1))
        total_dist += d

        # Speed of the terrain being moved into. Casualty and marker
        # pixels are not terrain, so fall back to light green.
        yy = min(max(int(y2), 0), h - 1)
        xx = min(max(int(x2), 0), w - 1)
        total_time += d / SPEED_BY_LABEL.get(int(labels[yy, xx]), 20.0)

    return total_time, total_dist


def score_breakdown(order, D, straight, priority, people):
    """Per-casualty score rows, in visiting sequence."""
    rows = []
    travelled = 0.0
    here = 0

    for c in order:
        travelled += D[here, c]
        ratio = straight[c] / travelled if travelled > 0 else 0.0
        p = people[c - 1]

        rows.append({
            "coords": [p["x"], p["y"]],
            "colour": p["colour"],
            "shape": p["shape"],
            "age_group": p["age_group"],
            "severity": p["colour"],
            "severity_score": p["severity_score"],
            "age_score": p["age_score"],
            "priority": p["priority"],
            "level": p["level"],
            "displacement": round(float(straight[c]), 2),
            "distance_travelled": round(float(travelled), 2),
            "score": round(float(ratio * priority[c]), 4),
        })
        here = c

    return rows


def draw_path(img, path, order, points_xy, people, out_path):
    """Save the original image with the route, stops, start and goal drawn."""
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

    pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (0, 0, 0), 3, cv2.LINE_AA)

    for step, c in enumerate(order, start=1):
        x, y = points_xy[c]
        label = f"{step}"
        cv2.circle(canvas, (x, y), 18, (255, 255, 255), 2)
        cv2.putText(canvas, label, (x - 10, y - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(canvas, label, (x - 10, y - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    for (px, py), text in ((points_xy[0], "START"), (points_xy[-1], "GOAL")):
        cv2.circle(canvas, (px, py), 20, (0, 0, 0), 3)
        cv2.putText(canvas, text, (px - 34, py + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4)
        cv2.putText(canvas, text, (px - 34, py + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imwrite(str(out_path), canvas)
    
    pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (0, 0, 0), 3, cv2.LINE_AA)
    ...
    cv2.imwrite(str(out_path), canvas)