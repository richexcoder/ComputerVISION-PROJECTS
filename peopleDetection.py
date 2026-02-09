import cv2
import imutils

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=640)

    # Detect people in the image
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Draw bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    # Display number of people detected
    cv2.putText(frame, f"People Detected: {len(boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("People Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()