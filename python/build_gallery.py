#!/usr/bin/env python3
"""Build the enrollment gallery from student photographs.

    npm run build:gallery
    python3 python/build_gallery.py --input gallery/images --output gallery/face_db.npz

═══════════════════════════════════════════════════════════════════════════
  TODO(enrollment): THIS SCRIPT IS READY BUT HAS NOT BEEN RUN ON REAL DATA.
  The new enrollment images have not been collected yet. When they arrive:

    1. Drop them into  gallery/images/
       Naming:  {REGNO}_{NN}_{variant}.jpg     e.g. 25BCE1276_01_frontal.jpg
       Anything before the first underscore is taken as the registration
       number, so 25BCE1276.jpg also works for a single image per student.

    2. Run:  npm run build:gallery
       Every image is quality-gated; failures are reported per student and
       written to gallery/rejected/ so you can see exactly what to retake.

    3. Run:  npm run verify:gallery
       Self-consistency and lookalike checks, plus threshold calibration.

    4. Set FACE_THRESHOLD_ACCEPT / FACE_THRESHOLD_REVIEW in .env from the
       calibration output. Do NOT keep the current 0.50 placeholder — it was
       never calibrated on this data.

  No other part of the system needs to change. The verifier picks up the new
  face_db.npz on its next restart.
═══════════════════════════════════════════════════════════════════════════

WHY THE ENROLLMENT GATES ARE STRICTER THAN THE VERIFICATION GATES
----------------------------------------------------------------
An enrollment error is permanent, silent, and affects one student for the
whole semester. A verification error is visible and retryable in seconds. So
enrollment rejects aggressively — a retake costs 20 seconds now, versus a
student being unable to mark attendance for months.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from face_pipeline import gallery as gallery_module  # noqa: E402
from face_pipeline.config import PipelineConfig, QualityThresholds  # noqa: E402
from face_pipeline.matching import cosine_similarity  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Stricter than the probe thresholds in config.py. See module docstring.
ENROLLMENT_THRESHOLDS = QualityThresholds(
    min_det_score=0.85,
    min_interocular_px=90.0,   # ISO/IEC 19794-5 "high quality"
    min_blur_variance=90.0,
    min_brightness=80.0,
    max_brightness=180.0,
    min_contrast_std=32.0,
    max_clipped_fraction=0.03,
    max_yaw_deg=30.0,          # pose shots are expected; see --allow-pose
    max_pitch_deg=15.0,
    max_roll_deg=12.0,
)

# An image whose embedding disagrees this much with the student's own others
# is almost certainly a different person, a mislabelled file, or a bad crop.
SELF_CONSISTENCY_FLOOR = 0.55
# Similarity to a *different* student's images that warrants a human look:
# duplicate enrollment, siblings, or twins.
CROSS_IDENTITY_CEILING = 0.45


def extract_regno(path: Path) -> str:
    return gallery_module.normalize_regno(path.stem.split("_")[0])


def discover(input_dir: Path) -> dict:
    grouped: dict = defaultdict(list)
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "rejected" in path.parts:
            continue
        regno = extract_regno(path)
        if regno:
            grouped[regno].append(path)
    return dict(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the face enrollment gallery")
    parser.add_argument("--input", type=Path, default=None, help="directory of enrollment images")
    parser.add_argument("--output", type=Path, default=None, help="output .npz path")
    parser.add_argument(
        "--allow-pose",
        action="store_true",
        help="permit deliberate left/right turn shots (relaxes the yaw gate to 35 degrees)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=1,
        help="warn when a student has fewer than this many accepted images (recommended: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, do not write the gallery"
    )
    args = parser.parse_args()

    config = PipelineConfig.from_env()
    input_dir = args.input or config.images_dir
    output_path = args.output or (config.gallery_dir / "face_db.npz")
    rejected_dir = config.gallery_dir / "rejected"

    if not input_dir.is_dir():
        print(f"ERROR: no such directory: {input_dir}", file=sys.stderr)
        print("       Place enrollment images there. See docs/ENROLLMENT.md.", file=sys.stderr)
        return 1

    grouped = discover(input_dir)
    if not grouped:
        print(f"ERROR: no images found in {input_dir}", file=sys.stderr)
        return 1

    total_images = sum(len(paths) for paths in grouped.values())
    print(f"Found {total_images} images across {len(grouped)} registration numbers.")
    print(f"Model: {config.model_name}\n")

    thresholds = ENROLLMENT_THRESHOLDS
    if args.allow_pose:
        thresholds = replace(thresholds, max_yaw_deg=35.0)

    # Heavy imports happen only once we know there is work to do.
    import cv2

    from face_pipeline.engine import FaceEngine

    engine = FaceEngine(config)
    print("Loading model (first run downloads it; this can take a few minutes)...")
    engine.load()
    print("Model ready.\n")

    accepted: dict = {}
    rejections: list[dict] = []
    per_image_quality: dict = {}

    for regno in sorted(grouped):
        vectors = []
        for path in grouped[regno]:
            image = cv2.imread(str(path))
            if image is None:
                rejections.append({"regno": regno, "file": path.name, "reason": "UNREADABLE"})
                continue

            face, error = engine.analyse(image, thresholds)
            if error is not None:
                rejections.append({"regno": regno, "file": path.name, "reason": error})
                continue

            if not face.report.passed:
                rejections.append(
                    {
                        "regno": regno,
                        "file": path.name,
                        "reason": face.report.failure,
                        "detail": face.report.message,
                    }
                )
                continue

            vectors.append(face.embedding)
            per_image_quality[path.name] = face.report.as_dict()

        if vectors:
            accepted[regno] = vectors

    # ── Self-consistency: the check that catches mis-enrollment ──────
    # Compare each of a student's embeddings against the mean of their others.
    # A low score means the images are not all the same person — the exact
    # failure that makes one student permanently unable to verify, and one the
    # old pipeline had no way to notice.
    inconsistent: list[dict] = []
    for regno, vectors in accepted.items():
        if len(vectors) < 2:
            continue
        stacked = np.vstack(vectors)
        for position in range(len(vectors)):
            others = np.delete(stacked, position, axis=0).mean(axis=0)
            score = cosine_similarity(vectors[position], others)
            if score < SELF_CONSISTENCY_FLOOR:
                inconsistent.append(
                    {"regno": regno, "image": position, "similarity": round(score, 4)}
                )

    # ── Cross-identity: duplicates, siblings, twins ──────────────────
    collisions: list[dict] = []
    regnos = sorted(accepted)
    if len(regnos) > 1:
        centroids = np.vstack(
            [gallery_module.l2_normalize(np.mean(np.vstack(accepted[r]), axis=0)) for r in regnos]
        )
        similarity_matrix = centroids @ centroids.T
        np.fill_diagonal(similarity_matrix, -1.0)
        for i, regno in enumerate(regnos):
            j = int(np.argmax(similarity_matrix[i]))
            score = float(similarity_matrix[i][j])
            if score > CROSS_IDENTITY_CEILING and i < j:
                collisions.append({"a": regno, "b": regnos[j], "similarity": round(score, 4)})

    # ── Report ───────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Accepted : {sum(len(v) for v in accepted.values())} images / {len(accepted)} students")
    print(f"Rejected : {len(rejections)} images")

    thin = {r: len(v) for r, v in accepted.items() if len(v) < args.min_images}
    missing = sorted(set(grouped) - set(accepted))

    if rejections:
        print("\n--- Rejected images (retake these) ---")
        by_reason: dict = defaultdict(list)
        for rejection in rejections:
            by_reason[rejection["reason"]].append(rejection)
        for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {reason} ({len(items)}):")
            for item in items[:10]:
                print(f"    - {item['file']}{'  ' + item.get('detail', '') if item.get('detail') else ''}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

    if missing:
        print(f"\n!!! {len(missing)} students have NO usable image — they CANNOT be verified:")
        for regno in missing[:20]:
            print(f"    - {regno}")

    if thin:
        print(f"\n--- {len(thin)} students below --min-images={args.min_images} ---")
        for regno, count in sorted(thin.items())[:20]:
            print(f"    - {regno}: {count}")

    if inconsistent:
        print("\n!!! SELF-CONSISTENCY FAILURES — likely wrong person or bad crop:")
        for item in inconsistent:
            print(f"    - {item['regno']} image #{item['image']} scored {item['similarity']}")

    if collisions:
        print("\n--- Lookalike pairs (review manually: duplicates, siblings, twins) ---")
        for item in collisions:
            print(f"    - {item['a']} vs {item['b']}: {item['similarity']}")

    if not accepted:
        print("\nERROR: no usable embeddings — nothing to write.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--dry-run: gallery not written.")
        return 0

    # Copy rejects somewhere the operator can actually look at them.
    if rejections:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for rejection in rejections:
            for path in grouped.get(rejection["regno"], []):
                if path.name == rejection["file"]:
                    shutil.copy2(path, rejected_dir / f"{rejection['reason']}__{path.name}")
                    break

    meta = gallery_module.save(
        output_path,
        accepted,
        model_name=config.model_name,
        extra_meta={
            "source_dir": str(input_dir),
            "rejected_images": len(rejections),
            "self_consistency_failures": len(inconsistent),
            "lookalike_pairs": len(collisions),
            "quality": per_image_quality,
        },
    )

    print(f"\nWrote {output_path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "quality"}, indent=2))
    print("\nNext: npm run verify:gallery   (self-check + threshold calibration)")

    return 2 if (missing or inconsistent) else 0


if __name__ == "__main__":
    sys.exit(main())
