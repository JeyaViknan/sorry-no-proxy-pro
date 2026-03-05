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

# Prefer requested dataset layout.
PRIMARY_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
FALLBACK_DATASET_DIRS = [
    os.path.join(SCRIPT_DIR, "dataset"),
    os.path.join(SCRIPT_DIR, "data", "images"),
    os.path.join(SCRIPT_DIR, "data"),
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# In-memory store: {registration_number: [normalized embedding vectors]}
embedding_store: Dict[str, List[np.ndarray]] = {}


def normalize_regno(value: str) -> str:
    return str(value).strip().upper()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    return float(np.dot(vec1, vec2) / denom)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-10
    return (vec / norm).astype(np.float32)


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
        return l2_normalize(np.array(reps[0]["embedding"], dtype=np.float32))
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


def get_active_dataset_dirs() -> List[str]:
    """Use only dataset/ when present; otherwise fallback to legacy folders."""
    if os.path.isdir(PRIMARY_DATASET_DIR):
        return [PRIMARY_DATASET_DIR]
    return [d for d in FALLBACK_DATASET_DIRS if os.path.isdir(d)]


def discover_dataset_images() -> List[str]:
    images: List[str] = []
    seen = set()

    for dataset_dir in get_active_dataset_dirs():
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


def initialize_embeddings() -> Dict[str, List[np.ndarray]]:
    """Load all dataset images and build in-memory embedding store."""
    global embedding_store

    if embedding_store:
        return embedding_store

    active_dirs = get_active_dataset_dirs()
    dataset_images = discover_dataset_images()
    if not dataset_images:
        print(
            f"No dataset images found in: {', '.join(active_dirs) if active_dirs else '[]'}",
            file=sys.stderr,
        )
        embedding_store = {}
        return embedding_store

    grouped: Dict[str, List[np.ndarray]] = {}
    print(
        f"Initializing embeddings from {len(dataset_images)} images in {active_dirs}...",
        file=sys.stderr
    )

    for img_path in dataset_images:
        regno = normalize_regno(os.path.splitext(os.path.basename(img_path))[0])
        if not regno:
            continue

        emb = represent_face(img_path)
        if emb is None:
            continue

        grouped.setdefault(regno, []).append(emb)

    embedding_store = {regno: embs for regno, embs in grouped.items() if embs}

    total_embeddings = sum(len(v) for v in embedding_store.values())
    print(
        f"Initialized {total_embeddings} embeddings for {len(embedding_store)} registration numbers.",
        file=sys.stderr
    )
    return embedding_store


def best_similarity_to_regno(captured_embedding: np.ndarray, reg_embeddings: List[np.ndarray]) -> float:
    """Best similarity between captured embedding and any sample of a regno."""
    if not reg_embeddings:
        return 0.0
    return max(cosine_similarity(captured_embedding, emb) for emb in reg_embeddings)


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

    # Identification logic:
    # 1) Claimed registration must be above threshold.
    # 2) Claimed registration must be the best overall match.
    # 3) Best match should be separated from second-best by a margin.
    claimed_similarity = best_similarity_to_regno(captured_embedding, store[regno])

    reg_scores: Dict[str, float] = {}
    for candidate_regno, candidate_embeddings in store.items():
        reg_scores[candidate_regno] = best_similarity_to_regno(
            captured_embedding,
            candidate_embeddings
        )

    if not reg_scores:
        return {
            "verified": False,
            "message": "No valid face embeddings available in database",
            "confidence": 0.0,
        }

    ranked = sorted(reg_scores.items(), key=lambda item: item[1], reverse=True)
    best_regno, best_score = ranked[0]
    second_best_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_best_score

    margin_threshold = 0.08
    verified = (
        claimed_similarity >= SIMILARITY_THRESHOLD
        and best_regno == regno
        and margin >= margin_threshold
    )

    return {
        "verified": verified,
        "message": (
            "Face matches registration number"
            if verified
            else f"Face does not match registration number (best match: {best_regno})"
        ),
        "confidence": float(claimed_similarity),
        "threshold": SIMILARITY_THRESHOLD,
        "best_match_regno": best_regno,
        "best_match_confidence": float(best_score),
        "second_best_confidence": float(second_best_score),
        "margin": float(margin),
        "margin_threshold": margin_threshold,
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
