#!/usr/bin/env python3
"""Audit the gallery and calibrate the similarity threshold empirically.

    npm run verify:gallery
    python3 python/verify_gallery.py --target-far 0.001

WHY THIS EXISTS
---------------
The threshold has always been a guessed constant: SIMILARITY_THRESHOLD = 0.50,
never measured against this cohort. 0.50 is a reasonable published operating
point for ArcFace on clean, same-domain pairs — which these are not. The
result was elevated false rejections AND elevated false acceptances at the
same time.

This script measures the actual score distributions and recommends a
threshold at a chosen false-acceptance rate, rather than inheriting a number
from a paper about a different dataset.

TARGET FAR: default 0.001 (0.1%).
A false acceptance is a successful proxy — the exact thing the system exists
to prevent — and it is undetectable after the fact. A false rejection is
visible, recoverable in seconds by retrying, and caught by the review band.
The costs are not symmetric, so the operating point should not be either.

═══════════════════════════════════════════════════════════════════════════
  TODO(enrollment): impostor statistics work on any gallery, but GENUINE
  statistics need multiple images per student. With one image each there are
  no genuine pairs and only half the picture is available. Once the new
  multi-image dataset is in place this reports both, and the recommended
  threshold becomes trustworthy.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from face_pipeline import gallery as gallery_module  # noqa: E402
from face_pipeline.config import PipelineConfig  # noqa: E402


def percentile_table(name: str, scores: np.ndarray) -> None:
    if scores.size == 0:
        print(f"  {name}: (none)")
        return
    points = [1, 5, 25, 50, 75, 95, 99]
    values = np.percentile(scores, points)
    print(f"  {name} (n={scores.size})")
    print("    " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(points, values)))
    print(f"    min={scores.min():.3f}  mean={scores.mean():.3f}  max={scores.max():.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and calibrate the face gallery")
    parser.add_argument("--target-far", type=float, default=0.001, help="target false-accept rate")
    parser.add_argument("--verbose", action="store_true", help="list per-student detail")
    args = parser.parse_args()

    config = PipelineConfig.from_env()

    try:
        gallery = gallery_module.load(config)
    except gallery_module.GalleryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = gallery.summary()
    print("=" * 70)
    print("GALLERY")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    embeddings = gallery.embeddings
    owners = gallery.owners

    if embeddings.shape[0] < 2:
        print("\nNot enough embeddings to analyse.")
        return 0

    # Full pairwise similarity. Cheap: 5000 identities is a 5000x5000 matmul.
    similarity = embeddings @ embeddings.T
    same_identity = owners[:, None] == owners[None, :]
    upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)

    genuine = similarity[same_identity & upper]
    impostor = similarity[(~same_identity) & upper]

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS")
    percentile_table("genuine  (same student, different image)", genuine)
    percentile_table("impostor (different students)", impostor)

    if genuine.size == 0:
        print(
            "\n  NOTE: no genuine pairs — every student has exactly one image.\n"
            "        False-rejection behaviour cannot be estimated from this gallery.\n"
            "        Collect 5-7 images per student (see docs/ENROLLMENT.md) and rerun."
        )

    # ── Operating point analysis ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION")

    if impostor.size == 0:
        print("  Not enough identities to estimate a false-accept rate.")
        return 0

    # Always show the FAR curve. This is measurable from any gallery and is
    # the half of the picture that does not need multiple images per student.
    print("  False-accept rate across candidate thresholds:")
    for candidate in (0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60):
        colliding = int((impostor >= candidate).sum())
        marker = "  <-- configured" if abs(candidate - config.threshold_accept) < 1e-9 else ""
        far_text = f"FAR={colliding / impostor.size:.4%}"
        print(f"    {candidate:.2f}: {colliding:4d} colliding pairs  {far_text}{marker}")

    # Pairs that collide at the CURRENTLY configured threshold are not a
    # statistic — each one is two real students who can mark each other
    # present, today.
    collide_mask = ((~same_identity) & upper) & (similarity >= config.threshold_accept)
    live_collisions = np.argwhere(collide_mask)
    if live_collisions.size > 0:
        print(f"\n  ** {len(live_collisions)} PAIR(S) COLLIDE AT THE CONFIGURED "
              f"{config.threshold_accept} THRESHOLD **")
        for i, j in live_collisions:
            print(f"       {owners[i]} <-> {owners[j]} : {similarity[i, j]:.4f}")
        print("     Each pair can verify as the other. Raise the threshold, or")
        print("     re-enroll these students with more images to separate them.")

    if genuine.size == 0:
        # Deliberately refuse to recommend. Deriving a threshold from impostor
        # data alone points in one direction only — lower it until the target
        # FAR is hit — which trades away security with no visibility into what
        # it buys in false rejections. That is precisely the mistake that
        # produced the uncalibrated 0.50 in the first place.
        safe_floor = float(impostor.max())
        print("\n  NO THRESHOLD RECOMMENDATION — genuine pairs are unavailable.")
        print("    Impostor data alone can only tell you how LOW you must not go.")
        print("    It says nothing about how many legitimate students a given")
        print("    threshold would reject, so any 'recommendation' from it would")
        print("    be one-sided. Collect 5-7 images per student, then rerun.")
        print(f"\n    Hard floor from this gallery: {safe_floor:.3f}")
        print(f"    (the highest impostor score seen — any threshold at or below")
        print(f"     this value admits a known collision)")
        print(f"\n    Currently configured: accept={config.threshold_accept} "
              f"review={config.threshold_review}")
        if config.threshold_accept <= safe_floor:
            print(f"    ** The configured threshold is AT OR BELOW that floor. **")
        return 0

    # With genuine data, a real operating point can be chosen.
    recommended = float(np.quantile(impostor, 1.0 - args.target_far))
    # Never recommend below the observed impostor ceiling.
    recommended = max(recommended, float(impostor.max()) + 0.01)
    review = min(float(np.quantile(genuine, 0.02)), recommended)

    print(f"\n  Target FAR         : {args.target_far:.4%}")
    print(f"  Recommended accept : {recommended:.3f}")
    print(f"    -> FAR {(impostor >= recommended).mean():.4%}")
    print(f"    -> FRR {(genuine < recommended).mean():.2%} of genuine pairs rejected")
    print(f"  Recommended review : {review:.3f}")
    print("       (scores in [review, accept) are recorded but flagged)")

    print("\n  Set these in .env:")
    print(f"    FACE_THRESHOLD_ACCEPT={recommended:.2f}")
    print(f"    FACE_THRESHOLD_REVIEW={review:.2f}")

    print(f"\n  Currently configured: accept={config.threshold_accept} review={config.threshold_review}")
    if abs(config.threshold_accept - recommended) > 0.05:
        print("  ** Configured threshold differs materially from the recommendation. **")

    # ── Weak links ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RISK REVIEW")

    risky = []
    for regno, rows in gallery.index.items():
        other_mask = owners != regno
        if not np.any(other_mask):
            continue
        best_other = float(similarity[rows][:, other_mask].max())
        if len(rows) > 1:
            worst_self = float(similarity[np.ix_(rows, rows)][np.triu(
                np.ones((len(rows), len(rows)), dtype=bool), k=1)].min())
        else:
            worst_self = None
        if best_other > 0.45 or (worst_self is not None and worst_self < 0.55):
            risky.append((regno, best_other, worst_self, len(rows)))

    if not risky:
        print("  No students flagged.")
    else:
        print(f"  {len(risky)} students need a look:")
        for regno, best_other, worst_self, count in sorted(risky, key=lambda r: -r[1])[:25]:
            self_text = f"{worst_self:.3f}" if worst_self is not None else "n/a"
            print(
                f"    {regno}  images={count}  nearest_other={best_other:.3f}  weakest_self={self_text}"
            )
        print("\n    nearest_other > 0.45 -> lookalike / possible duplicate enrollment")
        print("    weakest_self  < 0.55 -> images may not all be the same person; retake")

    single_image = [r for r, rows in gallery.index.items() if len(rows) == 1]
    if single_image:
        print(f"\n  {len(single_image)} students have only ONE enrollment image.")
        print("    Single-image enrollment is the largest single cause of false")
        print("    rejections (glasses, facial hair, lighting). Target 5-7 images.")
        if args.verbose:
            for regno in sorted(single_image):
                print(f"      - {regno}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
