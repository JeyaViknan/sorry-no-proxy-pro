#!/usr/bin/env python3
"""
Face Verification Service
Uses face detection and improved feature extraction to verify if a captured face matches a registration number.
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
face_detector = None
face_embeddings_cache = {}

def load_face_detector():
    """Load OpenCV DNN face detector."""
    global face_detector
    
    if face_detector is None:
        try:
            # Try to load OpenCV DNN face detector
            # Using OpenCV's built-in DNN face detector (more reliable than Haar cascades)
            prototxt_path = os.path.join(SCRIPT_DIR, "deploy.prototxt")
            model_path = os.path.join(SCRIPT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("Loaded OpenCV DNN face detector.", file=sys.stderr)
            else:
                # Fallback to Haar Cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_detector = cv2.CascadeClassifier(cascade_path)
                if face_detector.empty():
                    print("Warning: Could not load face detector.", file=sys.stderr)
                    face_detector = None
                else:
                    print("Loaded Haar Cascade face detector.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading face detector: {e}", file=sys.stderr)
            # Fallback to Haar Cascade
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_detector = cv2.CascadeClassifier(cascade_path)
                if not face_detector.empty():
                    print("Loaded Haar Cascade face detector (fallback).", file=sys.stderr)
                else:
                    face_detector = None
            except:
                face_detector = None
    
    return face_detector

def detect_face_opencv_dnn(image):
    """Detect face using OpenCV DNN."""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123])
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype(int)
            faces.append((x, y, x2 - x, y2 - y))
    
    return faces

def detect_face_haar(image):
    """Detect face using Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return [(x, y, w, h) for (x, y, w, h) in faces]

def detect_and_extract_face(image):
    """Detect face in image and extract face region."""
    global face_detector
    
    if face_detector is None:
        load_face_detector()
    
    if face_detector is None:
        # If no detector available, return center crop
        h, w = image.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        face_roi = image[y:y+size, x:x+size]
        return cv2.resize(face_roi, (224, 224))
    
    # Try DNN detector first
    if hasattr(face_detector, 'setInput'):
        faces = detect_face_opencv_dnn(image)
    else:
        faces = detect_face_haar(image)
    
    if len(faces) == 0:
        # No face detected, return center crop
        h, w = image.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        face_roi = image[y:y+size, x:x+size]
        return cv2.resize(face_roi, (224, 224))
    
    # Use the largest face detected
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Extract face region with some padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    face_roi = image[y:y+h, x:x+w]
    
    # Resize to standard size
    face_resized = cv2.resize(face_roi, (224, 224))
    
    return face_resized

def extract_face_features_improved(image):
    """Extract improved face features using multiple techniques."""
    # Detect and extract face
    face_roi = detect_and_extract_face(image)
    
    # Convert to grayscale for some features
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    
    features = []
    
    # 1. Histogram of Oriented Gradients (HOG) features
    try:
        # Calculate HOG features
        win_size = (224, 224)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray)
        features.extend(hog_features.flatten())
    except:
        pass
    
    # 2. Local Binary Pattern (LBP) features
    try:
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        features.extend(hist)
    except:
        pass
    
    # 3. Color histogram features (for RGB images)
    if len(face_roi.shape) == 3:
        for channel in range(3):
            hist = cv2.calcHist([face_roi], [channel], None, [32], [0, 256])
            features.extend(hist.flatten())
    
    # 4. Normalized pixel values (smaller, focused region)
    face_normalized = face_roi.astype(np.float32) / 255.0
    # Use a smaller representation
    face_small = cv2.resize(face_normalized, (64, 64))
    features.extend(face_small.flatten())
    
    return np.array(features, dtype=np.float32)

def extract_face_features(image_path):
    """Extract face features from an image file."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return extract_face_features_improved(img)
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}", file=sys.stderr)
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
        
        return extract_face_features_improved(img)
    except Exception as e:
        print(f"Error processing base64 image: {e}", file=sys.stderr)
        return None

def calculate_similarity(features1, features2):
    """Calculate similarity between two feature vectors."""
    if features1 is None or features2 is None:
        return 0.0
    
    # Normalize features
    features1_norm = features1 / (np.linalg.norm(features1) + 1e-10)
    features2_norm = features2 / (np.linalg.norm(features2) + 1e-10)
    
    # Cosine similarity
    cosine_sim = np.dot(features1_norm, features2_norm)
    
    # Also calculate Euclidean distance (inverted and normalized)
    euclidean_dist = np.linalg.norm(features1 - features2)
    # Normalize by max possible distance (approximate)
    max_dist = np.linalg.norm(features1) + np.linalg.norm(features2)
    euclidean_sim = 1.0 - min(euclidean_dist / (max_dist + 1e-10), 1.0)
    
    # Combine both similarities (weighted average)
    combined_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
    
    return float(combined_similarity)

def load_model_and_data():
    """Load the face classifier model and label mapping."""
    global model, label_map
    
    if model is None:
        try:
            if os.path.exists(MODEL_PATH):
                print("Loading face classifier model...", file=sys.stderr)
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Could not load model file: {e}", file=sys.stderr)
            model = None
    
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
    
    # Load face detector
    load_face_detector()
    
    return model, label_map

def get_cached_embeddings(regno):
    """Get cached face embeddings for a registration number."""
    global face_embeddings_cache
    
    if regno in face_embeddings_cache:
        return face_embeddings_cache[regno]
    
    return None

def cache_embeddings(regno, embeddings):
    """Cache face embeddings for a registration number."""
    global face_embeddings_cache
    face_embeddings_cache[regno] = embeddings

def verify_face(register_number, base64_image):
    """Verify if the captured face matches the given registration number."""
    global model, label_map
    
    # Load model and data if not already loaded
    if label_map is None:
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
            'message': 'Failed to process captured image or no face detected',
            'confidence': 0.0
        }
    
    # Get reference images for this registration number
    reference_filenames = label_map[register_number]
    
    # Check cache first
    cached_embeddings = get_cached_embeddings(register_number)
    if cached_embeddings is not None:
        best_similarity = calculate_similarity(captured_features, cached_embeddings)
    else:
        # Compare with reference images
        best_similarity = 0
        all_ref_features = []
        
        for filename in reference_filenames:
            image_path = os.path.join(IMAGES_DIR, filename)
            if os.path.exists(image_path):
                ref_features = extract_face_features(image_path)
                if ref_features is not None:
                    all_ref_features.append(ref_features)
                    similarity = calculate_similarity(captured_features, ref_features)
                    best_similarity = max(best_similarity, similarity)
        
        # Cache average embeddings for future use
        if all_ref_features:
            avg_features = np.mean(all_ref_features, axis=0)
            cache_embeddings(register_number, avg_features)
    
    # Adaptive threshold based on feature quality
    # Lower threshold for better face recognition
    threshold = 0.5
    
    verified = best_similarity >= threshold
    
    return {
        'verified': verified,
        'message': 'Face matches registration number' if verified else f'Face does not match registration number (similarity: {best_similarity:.3f}, threshold: {threshold})',
        'confidence': float(best_similarity),
        'threshold': threshold
    }

if __name__ == "__main__":
    # Command line interface for testing
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: python face_verification.py <register_number> <base64_image>'}))
        sys.exit(1)
    
    register_number = sys.argv[1]
    base64_image = sys.argv[2]
    
    result = verify_face(register_number, base64_image)
    print(json.dumps(result))
