#!/usr/bin/env python3
"""
Face Verification Service
Uses the pre-trained face_classifier_model.pkl to verify if a captured face matches a registration number.
"""

import pickle
import pandas as pd
import numpy as np
import cv2
import base64
import sys
import json
import os
from pathlib import Path

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model and data once at startup
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_classifier_model.pkl")
LABELS_PATH = os.path.join(SCRIPT_DIR, "data", "labels.xlsx")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "data", "images")

# Global variables to store loaded model and data
model = None
label_map = None

def load_model_and_data():
    """Load the face classifier model and label mapping."""
    global model, label_map
    
    if model is None:
        print("Loading face classifier model...", file=sys.stderr)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.", file=sys.stderr)
    
    if label_map is None:
        print("Loading labels...", file=sys.stderr)
        df = pd.read_excel(LABELS_PATH)
        # Create mapping from registration number to image filenames
        label_map = {}
        for _, row in df.iterrows():
            regno = str(row['regno']).strip()
            filename = str(row['filename']).strip()
            if regno not in label_map:
                label_map[regno] = []
            label_map[regno].append(filename)
        print(f"Loaded {len(label_map)} registration numbers.", file=sys.stderr)
    
    return model, label_map

def extract_face_features(image_path):
    """Extract face features from an image using the model's preprocessing."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size (adjust based on your model's expected input)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Flatten or reshape based on model input requirements
        # This is a generic approach - adjust based on your actual model
        return img_normalized.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}", file=sys.stderr)
        return None

def extract_face_from_base64(base64_string):
    """Extract face features from a base64 encoded image."""
    try:
        # Decode base64
        image_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Flatten or reshape based on model input requirements
        return img_normalized.flatten()
    except Exception as e:
        print(f"Error processing base64 image: {e}", file=sys.stderr)
        return None

def predict_face(image_features):
    """Predict registration number from face features."""
    global model
    
    if model is None:
        load_model_and_data()
    
    if image_features is None:
        return None
    
    try:
        # Reshape for model input (adjust based on your model)
        if len(image_features.shape) == 1:
            image_features = image_features.reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(image_features)
        prediction_proba = model.predict_proba(image_features) if hasattr(model, 'predict_proba') else None
        
        return {
            'predicted_regno': str(prediction[0]) if len(prediction) > 0 else None,
            'confidence': float(np.max(prediction_proba[0])) if prediction_proba is not None else None
        }
    except Exception as e:
        print(f"Error in prediction: {e}", file=sys.stderr)
        # Try alternative approach - compare with database images
        return compare_with_database(image_features)

def compare_with_database(image_features):
    """Compare face features with all images in database."""
    global label_map
    
    if label_map is None:
        load_model_and_data()
    
    best_match = None
    best_score = 0
    
    # Compare with all images in database
    for regno, filenames in label_map.items():
        for filename in filenames:
            image_path = os.path.join(IMAGES_DIR, filename)
            if os.path.exists(image_path):
                db_features = extract_face_features(image_path)
                if db_features is not None:
                    # Calculate similarity (cosine similarity or euclidean distance)
                    similarity = np.dot(image_features.flatten(), db_features.flatten()) / (
                        np.linalg.norm(image_features.flatten()) * np.linalg.norm(db_features.flatten())
                    )
                    if similarity > best_score:
                        best_score = similarity
                        best_match = regno
    
    return {
        'predicted_regno': best_match,
        'confidence': float(best_score)
    }

def verify_face(register_number, base64_image):
    """Verify if the captured face matches the given registration number."""
    global model, label_map
    
    # Load model and data if not already loaded
    if model is None or label_map is None:
        load_model_and_data()
    
    # Check if registration number exists in database
    register_number = str(register_number).strip()
    if register_number not in label_map:
        return {
            'verified': False,
            'message': f'Registration number {register_number} not found in database',
            'confidence': 0.0
        }
    
    # Extract features from captured image
    captured_features = extract_face_from_base64(base64_image)
    if captured_features is None:
        return {
            'verified': False,
            'message': 'Failed to process captured image',
            'confidence': 0.0
        }
    
    # Get reference images for this registration number
    reference_filenames = label_map[register_number]
    
    # Compare with reference images
    best_similarity = 0
    for filename in reference_filenames:
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            ref_features = extract_face_features(image_path)
            if ref_features is not None:
                # Calculate cosine similarity
                similarity = np.dot(captured_features.flatten(), ref_features.flatten()) / (
                    np.linalg.norm(captured_features.flatten()) * np.linalg.norm(ref_features.flatten())
                )
                best_similarity = max(best_similarity, similarity)
    
    # Threshold for verification (adjust as needed)
    threshold = 0.6
    verified = best_similarity >= threshold
    
    return {
        'verified': verified,
        'message': 'Face matches registration number' if verified else 'Face does not match registration number',
        'confidence': float(best_similarity),
        'threshold': threshold
    }

if __name__ == "__main__":
    # Command line interface for testing
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python face_verification.py <register_number> <base64_image>'}))
        sys.exit(1)
    
    register_number = sys.argv[1]
    base64_image = sys.argv[2]
    
    result = verify_face(register_number, base64_image)
    print(json.dumps(result))
