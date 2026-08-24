import cv2
import random
from ultralytics import YOLO

# Loading YOLO model
yolo = YOLO("yolov8x.pt")


def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))


# Open webcam
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()

    if not ret:
        break

    # Run YOLO tracking
    results = yolo.track(frame, stream=True)

    for result in results:
        class_names = result.names

        for box in result.boxes:

            # Only detect objects with confidence above 0.4
            if box.conf[0] > 0.4:

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class number
                cls = int(box.cls[0])

                # Get class name
                class_name = class_names[cls]

                # Get confidence
                conf = float(box.conf[0])

                # Generate colour based on class
                colour = getColours(cls)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    colour,
                    2
                )

                # Draw label
                cv2.putText(
                    frame,
                    f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2
                )

    cv2.imshow("Object Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCap.release()
cv2.destroyAllWindows()