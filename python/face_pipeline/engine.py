"""InsightFace wrapper — the only module that touches the model.

Isolating it here means quality.py, matching.py and gallery.py stay pure
NumPy and can be unit tested without a 600 MB model download and an ONNX
runtime. It also means swapping the recognition backend later touches one
file rather than the whole pipeline.
"""

from __future__ import annotations

import base64
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import quality
from .matching import l2_normalize


@dataclass
class DetectedFace:
    embedding: np.ndarray
    report: quality.QualityReport


class FaceEngine:
    """Detection + recognition, with the model held warm for the process life."""

    def __init__(self, config):
        self.config = config
        self._app = None

    def load(self) -> None:
        """Load the model. Called once at worker startup, never per request.

        This is the cost the old architecture paid on *every* request by
        spawning a fresh Python process: 2-4s of imports plus 1-3s of model
        initialisation, serialised behind a global lock.
        """
        # Imported lazily so that importing this module (for tests, tooling,
        # or --selftest) does not require the heavy dependencies.
        import os

        import cv2  # noqa: F401  (ensures the OpenCV runtime is present early)
        from insightface.app import FaceAnalysis

        kwargs = {
            "name": self.config.model_name,
            # Only what verification needs. The default set also loads
            # genderage and landmark_3d_68, which cost memory and time we
            # never use.
            "allowed_modules": ["detection", "recognition"],
            "providers": ["CPUExecutionProvider"],
        }

        model_root = os.environ.get("INSIGHTFACE_HOME")
        if model_root:
            kwargs["root"] = model_root

        app = FaceAnalysis(**kwargs)
        # Detector input size. 640 is InsightFace's default; 480 uses
        # noticeably less memory and is ample for a face that fills a phone
        # selfie frame. Configurable so a 512MB host can be made to fit.
        det = int(os.environ.get("FACE_DET_SIZE", "640"))
        app.prepare(ctx_id=-1, det_size=(det, det))
        self._app = app

    @property
    def ready(self) -> bool:
        return self._app is not None

    def decode_image(self, base64_data: str) -> Optional[np.ndarray]:
        """base64 (with or without data-URI prefix) -> BGR ndarray."""
        import cv2

        try:
            payload = base64_data.split(",", 1)[1] if "," in base64_data else base64_data
            raw = base64.b64decode(payload, validate=False)
            buffer = np.frombuffer(raw, np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return image
        except Exception as exc:  # noqa: BLE001 — never let a bad frame kill the worker
            print(f"[engine] frame decode failed: {exc}", file=sys.stderr)
            return None

    def _aligned_crop(self, image: np.ndarray, face) -> np.ndarray:
        """The model's normalised 112x112 input, for quality measurement.

        Measuring blur and exposure on this rather than the raw frame is what
        makes the thresholds comparable across devices: it is already scale-
        and rotation-normalised by the same alignment the recogniser uses.
        """
        try:
            from insightface.utils import face_align

            return face_align.norm_crop(image, landmark=face.kps, image_size=112)
        except Exception:  # noqa: BLE001 — fall back to a bbox crop
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            height, width = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                return image
            return image[y1:y2, x1:x2]

    def analyse(self, image: np.ndarray, thresholds) -> tuple[Optional[DetectedFace], Optional[str]]:
        """Detect the primary face and score its quality.

        Returns (face, error_code). Exactly one is non-None.
        """
        if self._app is None:
            return None, "ENGINE_NOT_READY"

        try:
            faces = self._app.get(image)
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] detection failed: {exc}", file=sys.stderr)
            return None, "DETECTION_FAILED"

        if not faces:
            return None, "NO_FACE"

        # Largest face wins. build_gallery.py uses the identical rule — the
        # old code did NOT: generate_embeddings.py took faces[0] (arbitrary
        # detector order) while verification took the largest, so any
        # enrollment photo containing a bystander could have stored the wrong
        # person's embedding, permanently breaking that student.
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float((x2 - x1) * (y2 - y1))

        faces = sorted(faces, key=area, reverse=True)
        primary = faces[0]

        # A second comparably-sized face means we cannot be sure who is being
        # verified. Looser than the enrollment rule, which rejects outright.
        if len(faces) > 1 and area(faces[1]) > 0.55 * area(primary):
            return None, "MULTIPLE_FACES"

        embedding = getattr(primary, "embedding", None)
        if embedding is None:
            return None, "NO_EMBEDDING"

        report = quality.assess(
            aligned_crop=self._aligned_crop(image, primary),
            keypoints=primary.kps,
            det_score=float(getattr(primary, "det_score", 1.0)),
            thresholds=thresholds,
        )

        return DetectedFace(embedding=l2_normalize(embedding), report=report), None
