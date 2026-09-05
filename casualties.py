
import cv2
import numpy as np
from palette import CLASS_INDEX, SEVERITY, LEVEL, CASUALTY_COLOURS


MIN_AREA = 80

AGE_SCORE = {"circle": 3, "square": 2, "star": 1}
AGE_GROUP = {"circle": "children", "square": "senior", "star": "adult"}

TERRAIN = ("light_green", "mid_green", "dark_green")


def classify_shape(contour, area):
    """Return 'circle', 'square' or 'star' for one blob outline."""
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return "circle"

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 1.0
    circularity = 4 * np.pi * area / (perimeter ** 2)

    
    if solidity < 0.75:
        return "star"

    
    return "circle" if circularity > 0.87 else "square"


def terrain_under(labels, blob_mask):
    """Which terrain a blob is standing on, judged by the ring around it."""
    kernel = np.ones((9, 9), np.uint8)
    grown = cv2.dilate(blob_mask, kernel, iterations=2)
    ring = (grown > 0) & (blob_mask == 0)

    best_name, best_count = "light_green", 0
    for name in TERRAIN:
        count = int(((labels == CLASS_INDEX[name]) & ring).sum())
        if count > best_count:
            best_name, best_count = name, count
    return best_name


def find_casualties(labels):
    """Return a list of dicts, one per casualty."""
    found = []

    for colour in CASUALTY_COLOURS:
        colour_mask = (labels == CLASS_INDEX[colour]).astype(np.uint8)

        colour_mask = cv2.morphologyEx(
            colour_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        n, comp, stats, centroids = cv2.connectedComponentsWithStats(
            colour_mask, connectivity=8)

        
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue

            blob = (comp == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            shape = classify_shape(max(contours, key=cv2.contourArea), area)
            terrain = terrain_under(labels, blob)

            severity = SEVERITY[colour]
            age = AGE_SCORE[shape]

            found.append({
                "colour": colour,
                "shape": shape,
                "age_group": AGE_GROUP[shape],
                "severity_score": severity,
                "age_score": age,
                "priority": severity * age,
                "x": int(round(centroids[i][0])),
                "y": int(round(centroids[i][1])),
                "level": LEVEL[terrain],
                "area": area,
            })

    return found


def find_marker(labels, colour):
    """Centre of the orange or purple triangle."""
    m = (labels == CLASS_INDEX[colour]).astype(np.uint8)
    n, comp, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        raise ValueError(f"no {colour} marker found")

    
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return int(round(centroids[best][0])), int(round(centroids[best][1]))