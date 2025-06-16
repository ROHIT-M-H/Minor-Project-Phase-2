import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Data Loading and Preprocessing ---
def load_and_preprocess_images(data_dir, target_class):
    images = []
    labels = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                # Resize images to a consistent size (adjust as needed)
                img_resized = cv2.resize(img, (128, 128))
                # Flatten the image into a 1D array of pixel values
                img_flattened = img_resized.flatten()
                images.append(img_flattened)
                labels.append(target_class)
    return np.array(images), np.array(labels)

# Define your data directories
parkinsons_spiral_dir = r'/Users/preetikapanalkar/Desktop/Parkinson/pd/spiral'
parkinsons_wave_dir = r'/Users/preetikapanalkar/Desktop/Parkinson/pd/wave'
healthy_spiral_dir = r'/Users/preetikapanalkar/Desktop/Parkinson/Healthy/spiral'
healthy_wave_dir = r'/Users/preetikapanalkar/Desktop/Parkinson/Healthy/wave'

# Load images and create labels
X_parkinsons_spiral, y_parkinsons_spiral = load_and_preprocess_images(parkinsons_spiral_dir, 1) # 1 for Parkinson's
X_parkinsons_wave, y_parkinsons_wave = load_and_preprocess_images(parkinsons_wave_dir, 1)
X_healthy_spiral, y_healthy_spiral = load_and_preprocess_images(healthy_spiral_dir, 0)     # 0 for Healthy
X_healthy_wave, y_healthy_wave = load_and_preprocess_images(healthy_wave_dir, 0)

# Combine all data
X = np.concatenate((X_parkinsons_spiral, X_parkinsons_wave, X_healthy_spiral, X_healthy_wave))
y = np.concatenate((y_parkinsons_spiral, y_parkinsons_wave, y_healthy_spiral, y_healthy_wave))

# --- Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is suitable for small datasets
model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Parkinson\'s'], yticklabels=['Healthy', 'Parkinson\'s'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

import joblib

# Save the trained model
model_filename = 'parkinson_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved as {model_filename}")

# Save the scaler
scaler_filename = 'feature_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Feature scaler saved as {scaler_filename}")

