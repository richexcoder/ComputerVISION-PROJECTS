import cv2
import numpy as np
import tensorflow as tf

# Loading the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Loading the ImageNet class names
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
    # Reading the frame from my webcam
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

    # Adding the batch dimension
    image = np.expand_dims(image, axis=0)

    # Preprocess our image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Make our image prediction
    predictions = model.predict(image, verbose=0)
