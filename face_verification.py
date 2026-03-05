#!/usr/bin/env python3
"""
Face Verification Service (DeepFace + FaceNet)
Loads dataset embeddings at startup and verifies a captured face against
a registration number using cosine similarity.
"""

import base64
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
from deepface import DeepFace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "retinaface"
SIMILARITY_THRESHOLD = 0.6

# Prefer requested dataset layout, keep fallbacks for compatibility.
DATASET_DIRS = [
    os.path.join(SCRIPT_DIR, "dataset"),
    os.path.join(SCRIPT_DIR, "data", "images"),
    os.path.join(SCRIPT_DIR, "data"),
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# In-memory store: {registration_number: embedding_vector}
embedding_store: Dict[str, np.ndarray] = {}


def normalize_regno(value: str) -> str:
    return str(value).strip().upper()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    return float(np.dot(vec1, vec2) / denom)


def represent_face(img_path: str) -> Optional[np.ndarray]:
    """Generate FaceNet embedding for a single image path."""
    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
        if not reps:
            return None
        return np.array(reps[0]["embedding"], dtype=np.float32)
    except Exception as exc:
        print(f"Embedding failed for {img_path}: {exc}", file=sys.stderr)
        return None


def extract_embedding_from_base64(base64_image: str) -> Optional[np.ndarray]:
    """Decode base64 image and compute FaceNet embedding."""
    temp_path = None
    try:
        payload = base64_image.split(",", 1)[1] if "," in base64_image else base64_image
        image_bytes = base64.b64decode(payload)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        return represent_face(temp_path)
    except Exception as exc:
        print(f"Failed to process captured image: {exc}", file=sys.stderr)
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def discover_dataset_images() -> List[str]:
    images: List[str] = []
    seen = set()

    for dataset_dir in DATASET_DIRS:
        if not os.path.isdir(dataset_dir):
            continue
        for root, _, files in os.walk(dataset_dir):
            for fname in files:
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                full_path = os.path.join(root, fname)
                if full_path in seen:
                    continue
                seen.add(full_path)
                images.append(full_path)

    return images


def initialize_embeddings() -> Dict[str, np.ndarray]:
    """Load all dataset images and build in-memory registration->embedding map."""
    global embedding_store

    if embedding_store:
        return embedding_store

    dataset_images = discover_dataset_images()
    if not dataset_images:
        print(
            f"No dataset images found in: {', '.join(DATASET_DIRS)}",
            file=sys.stderr,
        )
        embedding_store = {}
        return embedding_store

    grouped: Dict[str, List[np.ndarray]] = {}
    print(f"Initializing embeddings from {len(dataset_images)} images...", file=sys.stderr)

    for img_path in dataset_images:
        regno = normalize_regno(os.path.splitext(os.path.basename(img_path))[0])
        if not regno:
            continue

        emb = represent_face(img_path)
        if emb is None:
            continue

        grouped.setdefault(regno, []).append(emb)

    # If multiple images exist for same registration number, average their embeddings.
    embedding_store = {
        regno: np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        for regno, embs in grouped.items()
        if embs
    }

    print(f"Initialized embeddings for {len(embedding_store)} registration numbers.", file=sys.stderr)
    return embedding_store


def verify_face(register_number: str, base64_image: str) -> dict:
    store = initialize_embeddings()
    regno = normalize_regno(register_number)

    if regno not in store:
        return {
            "verified": False,
            "message": f"Registration number {regno} not found in database",
            "confidence": 0.0,
        }

    captured_embedding = extract_embedding_from_base64(base64_image)
    if captured_embedding is None:
        return {
            "verified": False,
            "message": "Failed to process captured image or no face detected",
            "confidence": 0.0,
        }

    similarity = cosine_similarity(captured_embedding, store[regno])
    verified = similarity >= SIMILARITY_THRESHOLD

    return {
        "verified": verified,
        "message": (
            "Face matches registration number"
            if verified
            else "Face does not match registration number"
        ),
        "confidence": float(similarity),
        "threshold": SIMILARITY_THRESHOLD,
    }


def serve_forever() -> None:
    """Persistent worker mode for fast repeated verification requests."""
    initialize_embeddings()
    print("Face verifier ready", file=sys.stderr)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            reg = payload.get("registerNumber")
            img = payload.get("faceImage")

            if not reg or not img:
                response = {
                    "id": req_id,
                    "verified": False,
                    "message": "Missing register number or face image",
                    "confidence": 0.0,
                }
            else:
                result = verify_face(reg, img)
                response = {"id": req_id, **result}
        except Exception as exc:
            response = {
                "id": req_id,
                "verified": False,
                "message": f"Internal face verification error: {exc}",
                "confidence": 0.0,
            }

        print(json.dumps(response), flush=True)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve_forever()
        return

    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python face_verification.py <register_number> <base64_image>"}))
        sys.exit(1)

    register_number = sys.argv[1]
    base64_image = sys.argv[2]

    result = verify_face(register_number, base64_image)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
