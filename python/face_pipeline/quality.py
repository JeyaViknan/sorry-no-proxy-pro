"""Frame quality assessment — pure NumPy, no model dependency.

WHY THIS EXISTS
---------------
The old pipeline compared whatever frame arrived, however bad. A student who
pressed Submit mid-motion produced a blurred frame, a garbage embedding, and
a rejection they could not act on ("Please speak to the professor"). There was
no way to distinguish "this is not you" from "this photo is unusable", so both
looked identical to the student and to the faculty member.

Separating the two matters because they need opposite responses: a quality
failure should say "hold still, move into the light, try again", while an
identity failure should not invite endless retries.

DESIGN NOTE
-----------
Everything here is deliberately free of cv2/insightface so it can be unit
tested without a 600MB model and an ONNX runtime. The scoring functions are
also reused by build_gallery.py to gate *enrollment* images, at stricter
thresholds — one implementation, two operating points.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

# InsightFace keypoint order.
LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH = range(5)


@dataclass
class QualityReport:
    det_score: float
    interocular_px: float
    blur_variance: float
    brightness: float
    contrast_std: float
    clipped_fraction: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    passed: bool = True
    failure: Optional[str] = None
    message: Optional[str] = None

    def as_dict(self) -> dict:
        data = asdict(self)
        # Keep the wire payload small and JSON-safe.
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 4)
        return data

    @property
    def sharpness_score(self) -> float:
        """Single scalar for ranking frames against each other.

        Blur dominates (it is the failure that actually happens), scaled by
        face size because a large sharp face carries far more usable detail
        than a small sharp one. Detector confidence breaks ties.
        """
        return (
            math.log1p(max(self.blur_variance, 0.0))
            * max(self.interocular_px, 1.0)
            * max(self.det_score, 0.01)
        )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV order) or already-gray -> float32 luminance."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        blue, green, red = array[:, :, 0], array[:, :, 1], array[:, :, 2]
        return (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.float32)
    raise ValueError(f"unsupported image shape {array.shape}")


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian — the standard blur proxy.

    Implemented with array slicing rather than cv2.Laplacian so this module
    stays dependency-light and testable. Equivalent to convolving with
    [[0,1,0],[1,-4,1],[0,1,0]] and discarding the border.
    """
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    centre = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * centre
    )
    return float(laplacian.var())


def exposure_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """(mean luminance, std dev, fraction of clipped pixels)."""
    if gray.size == 0:
        return 0.0, 0.0, 1.0

    clipped = np.count_nonzero((gray <= 1.0) | (gray >= 254.0))
    return float(gray.mean()), float(gray.std()), float(clipped / gray.size)


def interocular_distance(keypoints: Sequence[Sequence[float]]) -> float:
    """Pixel distance between eye centres — the ISO/IEC 19794-5 size metric.

    Preferred over bounding-box width because it is invariant to how much
    hair, chin or neck the detector happened to include.
    """
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[RIGHT_EYE] - points[LEFT_EYE]))


def head_pose(keypoints: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Approximate (yaw, pitch, roll) in degrees from the 5 landmarks.

    A full 3D pose solve needs the 68-point model, which would mean loading an
    extra InsightFace module and paying for it on every frame. These geometric
    proxies are accurate enough for a gate whose job is only to reject
    "clearly turned away" — and they are free, since the 5 keypoints come back
    from detection regardless.

    Calibration constants are empirical; the proxies are monotonic in the true
    angle, which is all a threshold needs.
    """
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape[0] < 5:
        return 0.0, 0.0, 0.0

    left_eye, right_eye = points[LEFT_EYE], points[RIGHT_EYE]
    nose = points[NOSE]
    mouth_mid = (points[LEFT_MOUTH] + points[RIGHT_MOUTH]) / 2.0

    eye_span = float(np.linalg.norm(right_eye - left_eye))
    if eye_span < 1e-3:
        return 0.0, 0.0, 0.0

    eye_mid = (left_eye + right_eye) / 2.0

    # Roll is exact: the tilt of the inter-eye line.
    roll = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Yaw: the nose drifts horizontally off the eye midpoint as the head turns.
    # Ratio is ~0 frontal and ~0.35 at 20 degrees.
    yaw = math.degrees(math.atan2(float(nose[0] - eye_mid[0]), eye_span * 1.6))

    # Pitch: the nose sits between eyes and mouth; nodding shifts it along
    # that axis. Normalised by eye-to-mouth distance so it is scale-free.
    eye_to_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    if eye_to_mouth < 1e-3:
        return yaw, 0.0, roll
    nose_offset = float(nose[1] - eye_mid[1]) / eye_to_mouth
    pitch = math.degrees(math.atan2(nose_offset - 0.5, 1.2))

    return yaw, pitch, roll


def assess(
    *,
    aligned_crop: np.ndarray,
    keypoints: Sequence[Sequence[float]],
    det_score: float,
    thresholds,
) -> QualityReport:
    """Score one detected face and decide whether it is usable.

    `aligned_crop` must be the model's normalised 112x112 input, not the raw
    frame — that is what makes blur and exposure comparable across devices.
    """
    gray = to_grayscale(aligned_crop)
    brightness, contrast, clipped = exposure_stats(gray)
    yaw, pitch, roll = head_pose(keypoints)

    report = QualityReport(
        det_score=float(det_score),
        interocular_px=interocular_distance(keypoints),
        blur_variance=laplacian_variance(gray),
        brightness=brightness,
        contrast_std=contrast,
        clipped_fraction=clipped,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )

    # Ordered so the student gets the most actionable instruction first:
    # framing, then steadiness, then lighting, then pose.
    checks = (
        (
            report.det_score < thresholds.min_det_score,
            "LOW_DETECTION",
            "Face not clearly visible. Move into better light and look at the camera.",
        ),
        (
            report.interocular_px < thresholds.min_interocular_px,
            "FACE_TOO_SMALL",
            "Move a little closer to the camera.",
        ),
        (
            report.blur_variance < thresholds.min_blur_variance,
            "TOO_BLURRY",
            "Hold the phone steady and try again.",
        ),
        (
            report.brightness < thresholds.min_brightness,
            "TOO_DARK",
            "Too dark. Face a window or a light.",
        ),
        (
            report.brightness > thresholds.max_brightness,
            "TOO_BRIGHT",
            "Too bright. Move away from direct light behind you.",
        ),
        (
            report.clipped_fraction > thresholds.max_clipped_fraction,
            "BAD_EXPOSURE",
            "Strong glare or shadow. Turn away from the bright light behind you.",
        ),
        (
            report.contrast_std < thresholds.min_contrast_std,
            "LOW_CONTRAST",
            "The picture looks washed out. Try somewhere with more even light.",
        ),
        (
            abs(report.yaw_deg) > thresholds.max_yaw_deg,
            "HEAD_TURNED",
            "Look straight at the camera.",
        ),
        (
            abs(report.pitch_deg) > thresholds.max_pitch_deg,
            "HEAD_TILTED",
            "Hold the phone at eye level.",
        ),
        (
            abs(report.roll_deg) > thresholds.max_roll_deg,
            "HEAD_ROTATED",
            "Keep your head upright.",
        ),
    )

    for failed, code, message in checks:
        if failed:
            report.passed = False
            report.failure = code
            report.message = message
            break

    return report
