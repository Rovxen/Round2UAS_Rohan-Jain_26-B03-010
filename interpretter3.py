import cv2
import numpy as np
from palette import snap
from mask import build_mask
from casualties import find_marker
from pathfind import (build_grid, to_small, nearest_free,
                      dijkstra, trace, to_full, SCALE)

img = cv2.cvtColor(cv2.imread("Images/IMG-20260831-WA0029.jpg"),
                   cv2.COLOR_BGR2RGB)
labels = snap(img)
mask = build_mask(labels)

start = find_marker(labels, "orange")
goal = find_marker(labels, "purple")

free, dist_cost, time_cost = build_grid(labels, mask)
s = nearest_free(free, to_small(start))
g = nearest_free(free, to_small(goal))

dist, prev = dijkstra(free, dist_cost, s)

straight = np.hypot(goal[0] - start[0], goal[1] - start[1])
travelled = dist[g] * SCALE          # small-grid steps back to full pixels

print("start", start, " goal", goal)
print(f"straight line : {straight:8.1f} px")
print(f"actual route  : {travelled:8.1f} px")
print(f"ratio         : {straight / travelled:8.3f}")

route = trace(prev, g)
print("route points  :", len(route))

