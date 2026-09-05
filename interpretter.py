import cv2
from palette import snap
from mask import build_mask, check_coverage

bgr = cv2.imread("Images\IMG-20260831-WA0029.jpg")
if bgr is None:
    raise FileNotFoundError("could not read images/image1.jpg")

labels = snap(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
check_coverage(labels)

mask = build_mask(labels)
cv2.imwrite("output/mask_image5.png", mask)

print("traversable:", round(100 * (mask == 255).mean(), 2), "%")