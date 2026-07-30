#!/usr/bin/env python3
"""Persistent face-verification worker.

Speaks newline-delimited JSON on stdin/stdout, correlated by request id.
Started once by server/services/faceWorker.js and kept warm for the life of
the process, so the model load is paid at boot instead of per request.

    stdin  <- {"id": "...", "registerNumber": "25BCE1276", "frames": ["<b64>", ...]}
    stdout -> {"id": "...", "result": {...}}
    stdout -> {"type": "ready", "identities": 67, "model": "buffalo_l"}
    stderr -> human-readable logs

stdout carries ONLY protocol JSON. The old implementation printed diagnostics
to stdout too, which forced server.js to scan the output backwards looking for
a line that happened to start with '{'. Everything informational goes to
stderr here, so a stray print can never corrupt a response.

Usage:
    python3 verifier.py --serve       run as a worker (normal operation)
    python3 verifier.py --selftest    validate imports/gallery and exit
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Keep the numeric libraries single-threaded. Each worker gets one core and
# parallelism comes from running several workers; without this, ONNX Runtime
# and BLAS each try to grab every core and oversubscribe the CPU, making the
# whole pool slower under load.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_pipeline import gallery as gallery_module  # noqa: E402
from face_pipeline.config import PipelineConfig  # noqa: E402
from face_pipeline.matching import MatchResult, best_of_frames, match_probe  # noqa: E402


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def emit(payload: dict) -> None:
    """Write one protocol message to stdout."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class Verifier:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.gallery = None
        self.engine = None

    def start(self) -> None:
        from face_pipeline.engine import FaceEngine

        started = time.time()

        self.gallery = gallery_module.load(self.config)
        log(f"[verifier] gallery loaded: {json.dumps(self.gallery.summary())}")

        self.engine = FaceEngine(self.config)
        self.engine.load()
        log(f"[verifier] model warm in {time.time() - started:.1f}s")

    def verify(self, register_number: str, frames: list[str], thresholds: dict) -> dict:
        """Verify a burst of frames against one registration number."""
        accept = float(thresholds.get("accept", self.config.threshold_accept))
        review = float(thresholds.get("review", self.config.threshold_review))
        register_number = gallery_module.normalize_regno(register_number)

        # Unknown registration number: answer identically to a mismatch, and
        # do not reveal which case it was. Distinguishing them would let an
        # unauthenticated caller enumerate which students are enrolled.
        if not self.gallery.has(register_number):
            return {
                "ok": False,
                "reason": "NO_MATCH",
                "message": "We could not match your face to that registration number.",
                "similarity": 0.0,
            }

        results: list[MatchResult] = []
        quality_failures: list[dict] = []
        best_quality = None

        for position, frame in enumerate(frames[: self.config.max_frames]):
            image = self.engine.decode_image(frame)
            if image is None:
                quality_failures.append({"frame": position, "code": "UNREADABLE"})
                continue

            face, error = self.engine.analyse(image, self.config.quality)
            if error is not None:
                quality_failures.append({"frame": position, "code": error})
                continue

            if best_quality is None or face.report.sharpness_score > best_quality.sharpness_score:
                best_quality = face.report

            if not face.report.passed:
                quality_failures.append(
                    {
                        "frame": position,
                        "code": face.report.failure,
                        "message": face.report.message,
                    }
                )
                continue

            result = match_probe(
                face.embedding,
                register_number=register_number,
                embeddings=self.gallery.embeddings,
                owners=self.gallery.owners,
                index=self.gallery.index,
                threshold_accept=accept,
                threshold_review=review,
            )
            results.append(result)

            # Latency optimisation: the client sends frames best-first, so an
            # early accept means the remaining frames cannot change the
            # outcome. Typical case becomes one frame (~200ms) instead of
            # three (~600ms), while a marginal case still gets the full burst.
            if result.status == "accepted":
                break

        if not results:
            return self._quality_failure_response(quality_failures, best_quality)

        best = best_of_frames(results)

        # Diagnostics stay on stderr. `best_other_regno` must never reach a
        # client — returning it turned the old endpoint into a biometric
        # identification oracle.
        log(
            "[verifier] "
            + json.dumps(
                {
                    "regno": register_number,
                    "status": best.status,
                    "similarity": round(best.similarity, 4),
                    "nearest_other": best.best_other_regno,
                    "nearest_other_similarity": round(best.best_other_similarity, 4),
                    "margin": round(best.margin, 4),
                    "frames_used": len(results),
                }
            )
        )

        if not best.verified:
            return {
                "ok": False,
                "reason": "NO_MATCH",
                "message": "We could not match your face to that registration number.",
                "similarity": best.similarity,
                "quality": best_quality.as_dict() if best_quality else None,
            }

        return {
            "ok": True,
            "status": best.status,
            "similarity": best.similarity,
            "quality": best_quality.as_dict() if best_quality else None,
        }

    @staticmethod
    def _quality_failure_response(failures: list[dict], best_quality) -> dict:
        """Turn quality problems into an instruction the student can act on.

        The distinction the old pipeline could not make: "the photo is
        unusable" is recoverable in two seconds by holding still or moving
        into the light, whereas "this is not you" is not. Collapsing both into
        one rejection is why legitimate students ended up being told to go
        speak to the professor.
        """
        if not failures:
            return {
                "ok": False,
                "reason": "NO_FACE",
                "message": "No face detected. Centre your face in the frame and try again.",
                "similarity": 0.0,
            }

        # Report the most common failure across the burst — one odd frame
        # should not decide the advice.
        counts: dict = {}
        for failure in failures:
            counts[failure["code"]] = counts.get(failure["code"], 0) + 1
        code = max(counts, key=counts.get)

        message = next(
            (f.get("message") for f in failures if f["code"] == code and f.get("message")),
            None,
        )
        fallbacks = {
            "NO_FACE": "No face detected. Centre your face in the frame and try again.",
            "MULTIPLE_FACES": "More than one face is visible. Make sure only you are in frame.",
            "UNREADABLE": "The photo could not be read. Please try again.",
            "DETECTION_FAILED": "Could not process the photo. Please try again.",
            "NO_EMBEDDING": "Could not process the photo. Please try again.",
            "ENGINE_NOT_READY": "Verification is starting up. Please try again in a moment.",
        }

        return {
            "ok": False,
            "reason": "QUALITY",
            "qualityCode": code,
            "message": message or fallbacks.get(code, "Photo quality too low. Please try again."),
            "similarity": 0.0,
            "quality": best_quality.as_dict() if best_quality else None,
        }


def serve(config: PipelineConfig) -> None:
    verifier = Verifier(config)

    try:
        verifier.start()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fatal", "error": str(exc)})
        log(f"[verifier] FATAL during startup: {exc}")
        sys.exit(1)

    emit(
        {
            "type": "ready",
            "identities": verifier.gallery.identity_count,
            "embeddings": verifier.gallery.embedding_count,
            "model": verifier.gallery.model_name,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")

            register_number = request.get("registerNumber")
            frames = request.get("frames") or []

            if not register_number or not frames:
                emit({"id": request_id, "error": "missing registerNumber or frames"})
                continue

            started = time.perf_counter()
            result = verifier.verify(
                register_number,
                frames,
                {
                    "accept": request.get("thresholdAccept", config.threshold_accept),
                    "review": request.get("thresholdReview", config.threshold_review),
                },
            )
            result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 1)

            emit({"id": request_id, "result": result})

        except Exception as exc:  # noqa: BLE001 — one bad request must not kill the worker
            log(f"[verifier] request failed: {exc}")
            emit({"id": request_id, "error": f"verification error: {exc}"})


def selftest(config: PipelineConfig) -> int:
    """Validate the environment without needing a camera or a live request."""
    print("Sorry No Proxy — verifier self-test")
    print(f"  gallery dir : {config.gallery_dir}")
    print(f"  model       : {config.model_name}")
    print(f"  thresholds  : accept={config.threshold_accept} review={config.threshold_review}")

    try:
        loaded = gallery_module.load(config)
        print(f"  gallery     : OK {json.dumps(loaded.summary())}")
    except gallery_module.GalleryError as exc:
        print(f"  gallery     : FAIL {exc}")
        return 1

    try:
        from face_pipeline.engine import FaceEngine

        engine = FaceEngine(config)
        engine.load()
        print("  model       : OK (loaded)")
    except Exception as exc:  # noqa: BLE001
        print(f"  model       : FAIL {exc}")
        return 1

    print("All checks passed.")
    return 0


def main() -> None:
    config = PipelineConfig.from_env()

    if "--selftest" in sys.argv:
        sys.exit(selftest(config))

    if "--serve" in sys.argv:
        serve(config)
        return

    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
