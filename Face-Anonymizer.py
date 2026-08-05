import cv2
import os

webcam = cv2.VideoCapture(0)
# Loading the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Visualize webcam
while True:
    ret, frame = webcam.read()

    if not ret:
        break
    # Converting frame to grayscale for better detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Drawing rectangles around detected faces
    for (x, y, w, h) in faces:
        # Extracting the face ROI
        face_roi = frame[y:y + h, x:x + w]

        # Applying Gaussian blur to the face ROI
        blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)

        # Replacing the original face region with blurred version
        frame[y:y + h, x:x + w] = blurred_face

     
    cv2.imshow("Face anonymizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()