import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET_DIR = "data/GTSRB/Final_Training/Images"
IMG_HEIGHT, IMG_WIDTH = 30, 30
NUM_CLASSES = 43

def load_data(dataset_path):
    images = []
    labels = []
    for class_id in range(NUM_CLASSES):
        class_folder = os.path.join(dataset_path, format(class_id, '05d'))
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Missing folder: {class_folder}")
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(class_id)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

print("[INFO] Loading dataset...")
X, y = load_data(DATASET_DIR)
print(f"[INFO] Dataset loaded: {X.shape[0]} images.")

X = X / 255.0
y = to_categorical(y, NUM_CLASSES)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training model...")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=64)

model.save("traffic_sign_model.h5")
print("[INFO] Model saved as traffic_sign_model.h5")
