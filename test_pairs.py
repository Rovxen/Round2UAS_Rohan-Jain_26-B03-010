import cv2
import numpy as np

from palette import snap
from mask import build_mask
from casualties import find_casualties, find_marker
from pathfind import build_grid, to_small, nearest_free, all_pairs

img = cv2.cvtColor(cv2.imread("Images/IMG-20260831-WA0029.jpg"),
                   cv2.COLOR_BGR2RGB)
labels = snap(img)
mask = build_mask(labels)

start = find_marker(labels, "orange")
goal = find_marker(labels, "purple")
people = find_casualties(labels)

free, dist_cost, time_cost = build_grid(labels, mask)


points_xy = [start] + [(c["x"], c["y"]) for c in people] + [goal]
points_rc = [nearest_free(free, to_small(p)) for p in points_xy]

print(f"{len(points_xy)} points: 1 start + {len(people)} casualties + 1 goal")
print("running Dijkstra, this takes a few seconds...")

D, T, routes = all_pairs(free, dist_cost, time_cost, points_rc)

print("\ndistance matrix shape:", D.shape)
print("unreachable pairs:", int(np.isinf(D).sum()))


print(f"\n{'point':22s} {'dist':>8s} {'time':>7s}")
for i, p in enumerate(points_xy):
    name = "START" if i == 0 else ("GOAL" if i == len(points_xy) - 1
                                   else f"{people[i-1]['colour']} {people[i-1]['shape']}")
    print(f"{name:22s} {D[0, i]:8.1f} {T[0, i]:7.2f}")


from optimise import plan, score_order

order, score, straight, priority = plan(points_xy, people, D)

print(f"\nvisiting {len(order)} of {len(people)} casualties")
print(f"{'stop':22s} {'pri':>4s} {'straight':>9s} {'travelled':>10s} {'ratio':>6s} {'score':>7s}")

travelled, here = 0.0, 0
for c in order:
    travelled += D[here, c]
    ratio = straight[c] / travelled
    print(f"{people[c-1]['colour'] + ' ' + people[c-1]['shape']:22s} "
          f"{priority[c]:4.0f} {straight[c]:9.1f} {travelled:10.1f} "
          f"{ratio:6.3f} {ratio * priority[c]:7.2f}")
    here = c

travelled += D[here, len(points_xy) - 1]
print(f"\nTOTAL SCORE {score:.2f}   total distance {travelled:.1f} px")


from pathlib import Path
from output import build_path, snap_path_to_points, path_time, draw_path

Path("outputs/paths").mkdir(parents=True, exist_ok=True)

n = len(points_xy)
path = build_path(order, routes, n)
path = snap_path_to_points(path, order, points_xy, n)

seconds, pixels = path_time(path, labels)
print(f"\npath points  : {len(path)}")
print(f"distance     : {pixels:.1f} px")
print(f"total time   : {seconds:.2f} s")

draw_path(img, path, order, points_xy, people, "outputs/paths/test_path.png")
print("saved outputs/paths/test_path.png")