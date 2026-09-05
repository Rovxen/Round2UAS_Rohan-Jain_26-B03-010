import cv2, json
from palette import snap
from casualties import find_casualties, find_marker

img = cv2.cvtColor(cv2.imread("Images/IMG-20260831-WA0025.jpg"), cv2.COLOR_BGR2RGB)
labels = snap(img)

people = find_casualties(labels)
start = find_marker(labels, "orange")
goal = find_marker(labels, "purple")

print(f"start {start}   goal {goal}   casualties {len(people)}")
print(f"{'CASUALTY':18s} {'PRI':>4s} {'COORDS':>12s} {'LVL':>4s} {'AREA':>6s}")
for c in sorted(people, key=lambda c: -c["priority"]):
    print(f"{c['colour'] + ' ' + c['shape']:18s} {c['priority']:4d} "
          f"{str((c['x'], c['y'])):>12s} {c['level']:4d} {c['area']:6d}")