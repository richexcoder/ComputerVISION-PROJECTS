import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("stop.png")
if img is None:
    print("Image not found!")
    exit()

# Resize
img = cv2.resize(img, (640, 480))

# CONVERT TO HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# RED COLOR
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = mask1 + mask2

# REMOVE NOISE
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# CONTOURS
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detecting People
people = cv2.d

# DETECTING OCTAGON
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 2000:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) == 8:  # octagon
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                "STOP SIGN",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

# RESULT
cv2.imshow("Stop Sign Detection", img)
cv2.imshow("Red Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
