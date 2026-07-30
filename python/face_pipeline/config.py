"""Pipeline configuration.

Thresholds arrive from the Node process via environment variables so that
server/config.js stays the single source of truth. Nothing here invents a
default that contradicts it; the fallbacks exist only so the module is
importable standalone (tests, build scripts, a REPL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class QualityThresholds:
    """Gates applied to a *probe* frame before it is trusted.

    Deliberately looser than the enrollment gates in build_gallery.py. The
    asymmetry is the point: an enrollment error is permanent and silent, so
    it should be caught hard at capture time; a verification error is
    retryable and visible, so being too strict here just rejects legitimate
    students who could have succeeded on the next frame.

    Blur is measured on the aligned 112x112 crop rather than the raw frame,
    because variance-of-Laplacian scales with resolution and face size. A
    threshold on the raw image would mean something different on every phone.
    """

    # Detector confidence.
    min_det_score: float = 0.60
    # Inter-ocular distance in pixels, measured on the original frame.
    # Below ~55px the model is upsampling invented detail into its 112x112
    # input, which is what the old 480px capture path was doing.
    min_interocular_px: float = 55.0
    # Variance of Laplacian on the aligned crop.
    # TODO(enrollment): recalibrate against the new dataset — set to the 2nd
    # percentile of frames judged acceptable. See docs/ENROLLMENT.md.
    min_blur_variance: float = 45.0
    # Mean luminance of the aligned crop, 0-255.
    min_brightness: float = 55.0
    max_brightness: float = 205.0
    # Flat / washed-out detection.
    min_contrast_std: float = 22.0
    # Fraction of pixels pinned at 0 or 255.
    max_clipped_fraction: float = 0.08
    # Pose, degrees.
    max_yaw_deg: float = 28.0
    max_pitch_deg: float = 25.0
    max_roll_deg: float = 20.0

    @classmethod
    def from_env(cls) -> "QualityThresholds":
        return cls(
            min_det_score=_float("FACE_MIN_DET_SCORE", cls.min_det_score),
            min_interocular_px=_float("FACE_MIN_IOD_PX", cls.min_interocular_px),
            min_blur_variance=_float("FACE_MIN_BLUR_VAR", cls.min_blur_variance),
            min_brightness=_float("FACE_MIN_BRIGHTNESS", cls.min_brightness),
            max_brightness=_float("FACE_MAX_BRIGHTNESS", cls.max_brightness),
            min_contrast_std=_float("FACE_MIN_CONTRAST_STD", cls.min_contrast_std),
            max_clipped_fraction=_float("FACE_MAX_CLIPPED", cls.max_clipped_fraction),
            max_yaw_deg=_float("FACE_MAX_YAW_DEG", cls.max_yaw_deg),
            max_pitch_deg=_float("FACE_MAX_PITCH_DEG", cls.max_pitch_deg),
            max_roll_deg=_float("FACE_MAX_ROLL_DEG", cls.max_roll_deg),
        )


@dataclass(frozen=True)
class PipelineConfig:
    gallery_dir: Path = field(default_factory=lambda: REPO_ROOT / "gallery")
    model_name: str = "buffalo_l"
    # Must match what build_gallery.py used. Embeddings are NOT comparable
    # across models — a silent mismatch rejects every student with no error
    # anywhere, so gallery.py refuses to load on a mismatch.
    embedding_dim: int = 512

    # 0.55 / 0.45, not 0.50 / 0.42. In the real 67-student gallery,
    # 25BRS1169 and 25BRS1286 score 0.5016 against each other — at 0.50 they
    # can verify as one another. Defaults must be safe when unset.
    # Node always passes explicit values; these matter for build_gallery.py,
    # verify_gallery.py and anyone running the worker standalone.
    threshold_accept: float = 0.55
    threshold_review: float = 0.45

    max_frames: int = 3

    quality: QualityThresholds = field(default_factory=QualityThresholds)

    @property
    def face_db_path(self) -> Path:
        return self.gallery_dir / "face_db.pkl"

    @property
    def images_dir(self) -> Path:
        return self.gallery_dir / "images"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        gallery = os.environ.get("GALLERY_DIR", "").strip()
        return cls(
            gallery_dir=Path(gallery).resolve() if gallery else REPO_ROOT / "gallery",
            model_name=os.environ.get("FACE_MODEL_NAME", "").strip() or cls.model_name,
            threshold_accept=_float("FACE_THRESHOLD_ACCEPT", cls.threshold_accept),
            threshold_review=_float("FACE_THRESHOLD_REVIEW", cls.threshold_review),
            max_frames=_int("FACE_MAX_FRAMES", cls.max_frames),
            quality=QualityThresholds.from_env(),
        )
