import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Load ImageNet class names
labels_path = tf.keras.utils.get_file(
    "imagenet_class_index.json",
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)

import json

with open(labels_path) as f:
    class_index = json.load(f)

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting image classifier...")
print("Press Q to quit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Resize image for MobileNetV2
    image = cv2.resize(frame, (224, 224))

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to numpy array
    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Preprocess image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Make prediction
    predictions = model.predict(image, verbose=0)

    # Get top prediction
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=1
    )[0][0]

    # Extract prediction information
    label = decoded[1]
    confidence = decoded[2] * 100

    # Create text
    text = f"{label}: {confidence:.1f}%"

    # Display prediction on webcam
    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("AI Image Classifier", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()