import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64  # image size

# Use exact folder paths
drowsy_folder = "dataset\Drowsy"
alert_folder  = "dataset\Non Drowsy"

def load_images(folder_path, label):
    images = []
    labels = []
    for i, img_name in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
        # Print progress every 1000 images
        if (i+1) % 1000 == 0:
            print(f"Loaded {i+1} images from {folder_path}")
    return images, labels

# -----------------------------
# Load images
# -----------------------------
print("Loading Drowsy images...")
drowsy_images, drowsy_labels = load_images(drowsy_folder, 1)

print("Loading Alert (Non Drowsy) images...")
alert_images, alert_labels = load_images(alert_folder, 0)

# -----------------------------
# Combine and normalize data
# -----------------------------
X = np.array(drowsy_images + alert_images)
y = np.array(drowsy_labels + alert_labels)

X = X / 255.0  # normalize
y = to_categorical(y, 2)  # one-hot encoding

print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

# -----------------------------
# Save preprocessed arrays (compressed)
# -----------------------------
np.savez_compressed("dataset.npz", X=X, y=y)
print("Preprocessed data saved as dataset.npz")
