import cv2
import numpy as np

img = cv2.imread("IMG-20260831-WA0025.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

flat = img.reshape(-1, 3)
colours, counts = np.unique(flat, axis=0, return_counts=True)

total = flat.shape[0]
print("image size:", img.shape)
print("total pixels:", total)
print("distinct colours:", len(colours))
print()

order = np.argsort(-counts)

threshold = 300
keep = counts >= threshold
print("colours above threshold:", keep.sum())

for i in np.argsort(-counts):
    if counts[i] < threshold:
        break
    r, g, b = colours[i]
    print(f"RGB({r:3d},{g:3d},{b:3d})  {counts[i]:7d}")
