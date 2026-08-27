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

