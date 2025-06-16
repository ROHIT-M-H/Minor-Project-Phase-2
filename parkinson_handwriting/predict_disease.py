import numpy as np
import cv2
import joblib
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for loading
import matplotlib.pyplot as plt

# --- Load the Trained Model and Scaler ---
try:
    loaded_model = joblib.load('parkinson_model.joblib')
    loaded_scaler = joblib.load('feature_scaler.joblib')
    print("Trained model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Trained model or scaler file not found. Make sure 'parkinson_model.joblib' and 'feature_scaler.joblib' are in the same directory.")
    exit()

# --- Function to Preprocess a Single Image ---
def preprocess_single_image(image_path, target_size=(128, 128)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        img_resized = cv2.resize(img, target_size)
        img_flattened = img_resized.flatten().reshape(1, -1) # Reshape for scaling
        img_scaled = loaded_scaler.transform(img_flattened)
        return img_scaled
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Prediction Function ---
def predict_parkinson(processed_image):
    if processed_image is not None:
        prediction_probability = loaded_model.predict_proba(processed_image)[0][1] # Probability of Parkinson's (class 1)
        predicted_class = loaded_model.predict(processed_image)[0]
        return predicted_class, prediction_probability
    else:
        return None, None

# --- Get Image Path from User ---
image_path = input("Enter the path to the handwriting image (spiral or wave): ")

# --- Preprocess the Image ---
processed_image = preprocess_single_image(image_path)

# --- Make Prediction ---
if processed_image is not None:
    predicted_class, prediction_probability = predict_parkinson(processed_image)

    if predicted_class == 1:
        prediction_label = "Parkinson's Disease"
    else:
        prediction_label = "Healthy"

    print(f"Prediction: {prediction_label} (Probability of Parkinson's: {prediction_probability:.4f})")