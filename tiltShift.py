import cv2
import numpy as np

# Load image
image = cv2.imread("me.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Resize image
image = cv2.resize(image, (800, 600))

# Create blurred version
blurred = cv2.GaussianBlur(image, (25, 25), 0)

height, width = image.shape[:2]

# Focus region height
focus_height = int(height * 0.4)

# Offset for moving focus area
offset = 0

print("Controls:")
print("W - Move focus up")
print("S - Move focus down")
print("ESC - Exit")

while True:
    # Create mask
    mask = np.zeros((height, width), dtype=np.float32)

    start = height // 2 - focus_height // 2 + offset
    end = height // 2 + focus_height // 2 - offset

    # Clamp values
    start = max(0, start)
    end = min(height, end)

    mask[start:end, :] = 1.0

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (55, 55), 0)

    # Convert to 3 channels
    mask = cv2.merge([mask, mask, mask])

    # Blend images
    tilt_shift = (image * mask + blurred * (1 - mask)).astype(np.uint8)

    # Show result
    cv2.imshow("Tilt Shift Effect", tilt_shift)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('w'):
        offset -= 10
    elif key == ord('s'):
        offset += 10
    elif key == 27:  # ESC key
        break

cv2.destroyAllWindows()