# File: src/load_data.py

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define paths
data_dir = 'data/images'  # Path to the directory containing labeled folders
labels_file = 'data/labels.csv'  # Path to the labels CSV file (if needed)

# Image processing parameters
IMG_SIZE = 48  # Image size (48x48 pixels)
num_classes = 8  # Number of emotion categories (can adjust based on your dataset)

# Define the mapping between folder names and emotion labels
emotion_mapping = {
    'Neutral': 0,
    'Happy': 1,
    'Sad': 2,
    'Surprise': 3,
    'Fear': 4,
    'Disgust': 5,
    'Anger': 6,
    'Contempt': 7
}

# Load and preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 48x48
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def load_data():
    X = []
    y = []

    for emotion, label in emotion_mapping.items():
        emotion_dir = os.path.join(data_dir, emotion)
        for image_name in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, image_name)
            try:
                img = preprocess_image(image_path)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
