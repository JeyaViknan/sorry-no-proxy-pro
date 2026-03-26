#!/usr/bin/env python3
"""
Face Verification Service (InsightFace)
Loads embeddings at startup and verifies a captured face against
a registration number using cosine similarity.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import base64
import json
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_PATH = os.path.join(SCRIPT_DIR, "face_db.pkl")
# NOTE: `face_db.pkl` must contain InsightFace embeddings produced by FaceAnalysis().
SIMILARITY_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.12

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

_face_app: Optional[FaceAnalysis] = None


def normalize_regno(value: str) -> str:
    return str(value).strip().upper()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    return float(np.dot(vec1, vec2) / denom)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-10
    return (vec / norm).astype(np.float32)


def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is not None:
        return _face_app

    model_root = os.environ.get("INSIGHTFACE_HOME")
    kwargs = {
        "name": "buffalo_s",
        "allowed_modules": ["detection", "recognition"],
        "providers": ["CPUExecutionProvider"],
    }
    if model_root:
        kwargs["root"] = model_root
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=-1)
    _face_app = app
    return _face_app


def embedding_from_bgr_image(img_bgr: np.ndarray, debug_tag: str = "") -> Optional[np.ndarray]:
    """Extract a single face embedding from a BGR image using InsightFace.

    - Requires at least 1 detected face.
    - Uses the largest face if multiple are found.
    """
    try:
        faces = get_face_app().get(img_bgr)
        if not faces:
            if debug_tag:
                print(f"Embedding skipped (no face detected): {debug_tag}", file=sys.stderr)
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = getattr(face, "embedding", None)
        if emb is None:
            if debug_tag:
                print(f"Embedding skipped (no embedding field): {debug_tag}", file=sys.stderr)
            return None

        return l2_normalize(np.array(emb, dtype=np.float32).reshape(-1))
    except Exception as exc:
        if debug_tag:
            print(f"Embedding failed for {debug_tag}: {exc}", file=sys.stderr)
        return None


def extract_embedding_from_base64(base64_image: str) -> Optional[np.ndarray]:
    """Decode base64 image and compute InsightFace embedding."""
    try:
        payload = base64_image.split(",", 1)[1] if "," in base64_image else base64_image
        image_bytes = base64.b64decode(payload)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return embedding_from_bgr_image(img, debug_tag="captured_frame")
    except Exception as exc:
        print(f"Failed to process captured image: {exc}", file=sys.stderr)
        return None


def get_active_dataset_dirs() -> List[str]:
    """Use only dataset/ when present; otherwise fallback to legacy folders."""
    if os.path.isdir(PRIMARY_DATASET_DIR):
        return [PRIMARY_DATASET_DIR]
    return [d for d in FALLBACK_DATASET_DIRS if os.path.isdir(d)]


def extract_regno_from_filename(path: str) -> str:
    """Extract registration number from filename (prefix before underscore)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return normalize_regno(stem.split("_")[0])


def discover_dataset_images_by_regno() -> Dict[str, List[str]]:
    grouped_paths: Dict[str, List[str]] = {}
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

                regno = extract_regno_from_filename(full_path)
                if not regno:
                    continue
                grouped_paths.setdefault(regno, []).append(full_path)

    return grouped_paths


def load_embeddings_from_face_db() -> Dict[str, List[np.ndarray]]:
    """Load embeddings from face_db.pkl.

    Expected format: {regno: np.ndarray(shape=(512,))} or {regno: [np.ndarray, ...]}.
    """
    if not os.path.exists(FACE_DB_PATH):
        return {}

    import pickle  # local import to keep startup lightweight

    with open(FACE_DB_PATH, "rb") as f:
        db = pickle.load(f)

    if not isinstance(db, dict):
        raise ValueError("face_db.pkl must contain a dict mapping regno -> embedding(s)")

    store: Dict[str, List[np.ndarray]] = {}
    for regno_raw, value in db.items():
        regno = normalize_regno(regno_raw)
        if value is None:
            continue

        embs: List[np.ndarray] = []
        if isinstance(value, np.ndarray):
            embs = [value]
        elif isinstance(value, (list, tuple)):
            embs = [v for v in value if isinstance(v, np.ndarray)]
        else:
            continue

        normalized = [l2_normalize(np.array(e, dtype=np.float32).reshape(-1)) for e in embs if e.size]
        if normalized:
            store[regno] = normalized

    return store


def initialize_embeddings() -> Dict[str, List[np.ndarray]]:
    """Initialize embedding store.

    Preference order:
    1) `face_db.pkl` (precomputed InsightFace embeddings)
    2) Scan dataset images and compute embeddings on the fly
    """
    global embedding_store

    if embedding_store:
        return embedding_store

    # Prefer precomputed embeddings (fast, consistent).
    try:
        face_db_store = load_embeddings_from_face_db()
        if face_db_store:
            embedding_store = face_db_store
            total_embeddings = sum(len(v) for v in embedding_store.values())
            print(
                f"Loaded {total_embeddings} embeddings for {len(embedding_store)} registration numbers from face_db.pkl.",
                file=sys.stderr,
            )
            return embedding_store
    except Exception as exc:
        print(f"Failed to load face_db.pkl ({FACE_DB_PATH}): {exc}", file=sys.stderr)

    active_dirs = get_active_dataset_dirs()
    dataset_images_by_regno = discover_dataset_images_by_regno()
    total_images = sum(len(paths) for paths in dataset_images_by_regno.values())
    if total_images == 0:
        print(
            f"No dataset images found in: {', '.join(active_dirs) if active_dirs else '[]'}",
            file=sys.stderr,
        )
        embedding_store = {}
        return embedding_store

    grouped: Dict[str, List[np.ndarray]] = {}
    embedding_counts_all: Dict[str, int] = {}
    print(
        f"Initializing embeddings from {total_images} images in {active_dirs}...",
        file=sys.stderr
    )

    for regno, image_paths in dataset_images_by_regno.items():
        embeddings_for_regno: List[np.ndarray] = []
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
            except Exception:
                img = None
            if img is None:
                continue
            emb = embedding_from_bgr_image(img, debug_tag=img_path)
            if emb is None:
                # Invalid images (no face or multiple faces) are skipped by design.
                continue
            embeddings_for_regno.append(emb)

        if not embeddings_for_regno:
            embedding_counts_all[regno] = 0
            print(f"Warning: no valid embeddings for registration {regno}; skipping.", file=sys.stderr)
            continue

        grouped[regno] = embeddings_for_regno
        embedding_counts_all[regno] = len(embeddings_for_regno)

    embedding_store = {regno: embs for regno, embs in grouped.items() if embs}
    print(
        f"Embedding counts by registration: {json.dumps(embedding_counts_all, sort_keys=True)}",
        file=sys.stderr
    )

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

    claimed_similarity = best_similarity_to_regno(captured_embedding, store[regno])

    best_regno: Optional[str] = None
    best_score = -1.0
    second_best_score = -1.0

    for candidate_regno, candidate_embeddings in store.items():
        score = best_similarity_to_regno(captured_embedding, candidate_embeddings)
        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_regno = candidate_regno

    margin = best_score - second_best_score

    # Strict identity verification: the entered registration number must be
    # the best global match and sufficiently separated from the runner-up.
    verified = (
        claimed_similarity >= SIMILARITY_THRESHOLD
        and best_regno == regno
        and margin >= MARGIN_THRESHOLD
    )

    print(
        "Verification diagnostics: "
        + json.dumps(
            {
                "entered_registration_number": regno,
                "claimed_similarity": float(claimed_similarity),
                "best_match_regno": best_regno,
                "best_match_confidence": float(best_score),
                "second_best_confidence": float(second_best_score),
                "margin": float(margin),
            }
        ),
        file=sys.stderr,
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
        "margin_threshold": MARGIN_THRESHOLD,
    }


def serve_forever() -> None:
    """Persistent worker mode for fast repeated verification requests."""
    # Warm the InsightFace model before declaring readiness so the first
    # verification request does not pay model download/init cost.
    get_face_app()
    initialize_embeddings()
    print(json.dumps({"type": "ready"}), flush=True)
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
