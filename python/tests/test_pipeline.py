#!/usr/bin/env python3
"""Tests for the pure-NumPy pipeline logic.

Runs without insightface, opencv or a model — that separation is the reason
these can run in milliseconds instead of requiring a 600MB download.

    python3 python/tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_pipeline import gallery as gallery_module
from face_pipeline.config import PipelineConfig, QualityThresholds
from face_pipeline.matching import (
    MatchResult,
    best_of_frames,
    cosine_similarity,
    l2_normalize,
    match_probe,
)
from face_pipeline.quality import (
    QualityReport,
    assess,
    exposure_stats,
    head_pose,
    interocular_distance,
    laplacian_variance,
    to_grayscale,
)

RNG = np.random.default_rng(20260730)


def unit_vector(dim: int = 512) -> np.ndarray:
    return l2_normalize(RNG.normal(size=dim).astype(np.float32))


def near(vector: np.ndarray, similarity: float) -> np.ndarray:
    """A vector at approximately the requested cosine similarity to `vector`."""
    orthogonal = RNG.normal(size=vector.shape).astype(np.float32)
    orthogonal -= np.dot(orthogonal, vector) * vector
    orthogonal = l2_normalize(orthogonal)
    return l2_normalize(similarity * vector + np.sqrt(max(1 - similarity**2, 0)) * orthogonal)


class TestQualityMetrics(unittest.TestCase):
    def test_grayscale_from_bgr_and_gray(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # red channel in BGR order
        gray = to_grayscale(bgr)
        self.assertAlmostEqual(float(gray[0, 0]), 0.299 * 255, places=3)

        already_gray = np.full((4, 4), 128, dtype=np.uint8)
        self.assertAlmostEqual(float(to_grayscale(already_gray)[0, 0]), 128.0)

    def test_sharp_image_scores_far_above_blurred(self):
        # A checkerboard is maximally high-frequency.
        sharp = np.indices((64, 64)).sum(axis=0) % 2 * 255.0
        # A smooth ramp has almost no second derivative.
        blurred = np.tile(np.linspace(0, 255, 64), (64, 1))

        self.assertGreater(laplacian_variance(sharp), laplacian_variance(blurred) * 100)

    def test_flat_image_has_zero_laplacian_variance(self):
        self.assertAlmostEqual(laplacian_variance(np.full((32, 32), 100.0)), 0.0, places=6)

    def test_laplacian_handles_degenerate_input(self):
        self.assertEqual(laplacian_variance(np.zeros((2, 2))), 0.0)
        self.assertEqual(laplacian_variance(np.zeros((0, 0))), 0.0)

    def test_exposure_detects_clipping(self):
        image = np.full((10, 10), 128.0)
        mean, std, clipped = exposure_stats(image)
        self.assertAlmostEqual(mean, 128.0)
        self.assertAlmostEqual(std, 0.0)
        self.assertAlmostEqual(clipped, 0.0)

        blown = np.full((10, 10), 255.0)
        _, _, clipped_fraction = exposure_stats(blown)
        self.assertAlmostEqual(clipped_fraction, 1.0)

    def test_interocular_distance(self):
        keypoints = [[10, 50], [90, 50], [50, 70], [30, 90], [70, 90]]
        self.assertAlmostEqual(interocular_distance(keypoints), 80.0, places=3)

    def test_frontal_face_has_near_zero_pose(self):
        keypoints = [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]]
        yaw, pitch, roll = head_pose(keypoints)
        self.assertLess(abs(yaw), 5.0)
        self.assertLess(abs(roll), 2.0)
        self.assertLess(abs(pitch), 12.0)

    def test_turned_head_produces_signed_yaw(self):
        left = [[40, 50], [80, 50], [46, 70], [45, 90], [75, 90]]
        right = [[40, 50], [80, 50], [74, 70], [45, 90], [75, 90]]
        self.assertLess(head_pose(left)[0], -5.0)
        self.assertGreater(head_pose(right)[0], 5.0)

    def test_tilted_head_produces_roll(self):
        tilted = [[40, 40], [80, 60], [60, 70], [45, 90], [75, 90]]
        self.assertGreater(abs(head_pose(tilted)[2]), 15.0)

    def test_degenerate_keypoints_do_not_raise(self):
        self.assertEqual(head_pose([]), (0.0, 0.0, 0.0))
        self.assertEqual(head_pose([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]), (0.0, 0.0, 0.0))


class TestQualityGate(unittest.TestCase):
    def setUp(self):
        self.thresholds = QualityThresholds()
        # Keypoints are in ORIGINAL FRAME coordinates (that is what
        # InsightFace returns), not the 112x112 aligned crop. These describe a
        # face in a 720p portrait frame with ~120px inter-ocular distance —
        # roughly what a phone front camera produces at arm's length.
        self.good_keypoints = [[300, 420], [420, 420], [360, 500], [318, 570], [402, 570]]
        # Textured crop that passes blur and contrast.
        self.good_crop = (RNG.integers(60, 190, size=(112, 112, 3))).astype(np.uint8)

    def test_good_frame_passes(self):
        report = assess(
            aligned_crop=self.good_crop,
            keypoints=self.good_keypoints,
            det_score=0.95,
            thresholds=self.thresholds,
        )
        self.assertTrue(report.passed, f"unexpected failure: {report.failure}")

    def test_blurred_frame_is_rejected_with_actionable_message(self):
        smooth = np.tile(np.linspace(80, 160, 112), (112, 1))
        smooth = np.stack([smooth] * 3, axis=-1).astype(np.uint8)
        report = assess(
            aligned_crop=smooth,
            keypoints=self.good_keypoints,
            det_score=0.95,
            thresholds=self.thresholds,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failure, "TOO_BLURRY")
        self.assertIn("steady", report.message.lower())

    def test_dark_frame_is_rejected(self):
        dark = (RNG.integers(0, 25, size=(112, 112, 3))).astype(np.uint8)
        report = assess(
            aligned_crop=dark,
            keypoints=self.good_keypoints,
            det_score=0.95,
            thresholds=self.thresholds,
        )
        self.assertFalse(report.passed)
        self.assertIn(report.failure, ("TOO_DARK", "BAD_EXPOSURE", "LOW_CONTRAST"))

    def test_small_face_is_rejected_before_blur(self):
        # ~30px inter-ocular: a face too far from the camera. This is roughly
        # what the old 480px capture path produced, which is why it fed the
        # recogniser upsampled detail that was never in the image.
        tiny_keypoints = [[300, 420], [330, 420], [315, 440], [305, 462], [325, 462]]
        report = assess(
            aligned_crop=self.good_crop,
            keypoints=tiny_keypoints,
            det_score=0.95,
            thresholds=self.thresholds,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failure, "FACE_TOO_SMALL")

    def test_low_detection_confidence_is_rejected_first(self):
        report = assess(
            aligned_crop=self.good_crop,
            keypoints=self.good_keypoints,
            det_score=0.2,
            thresholds=self.thresholds,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failure, "LOW_DETECTION")

    def test_sharpness_score_ranks_frames_for_burst_selection(self):
        sharp = QualityReport(0.95, 120, 300, 130, 45, 0.0, 0, 0, 0)
        blurry = QualityReport(0.95, 120, 20, 130, 45, 0.0, 0, 0, 0)
        small = QualityReport(0.95, 60, 300, 130, 45, 0.0, 0, 0, 0)

        self.assertGreater(sharp.sharpness_score, blurry.sharpness_score)
        self.assertGreater(sharp.sharpness_score, small.sharpness_score)

    def test_enrollment_thresholds_are_stricter_than_probe(self):
        from build_gallery import ENROLLMENT_THRESHOLDS

        probe = QualityThresholds()
        self.assertGreater(ENROLLMENT_THRESHOLDS.min_det_score, probe.min_det_score)
        self.assertGreater(ENROLLMENT_THRESHOLDS.min_interocular_px, probe.min_interocular_px)
        self.assertGreater(ENROLLMENT_THRESHOLDS.min_blur_variance, probe.min_blur_variance)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.identities = {f"25BCE{1000 + i}": [unit_vector()] for i in range(20)}
        self.target = "25BCE1000"

    def _build(self, identities):
        embeddings = []
        owners = []
        for regno in sorted(identities):
            for vector in identities[regno]:
                embeddings.append(l2_normalize(vector))
                owners.append(regno)
        matrix = np.vstack(embeddings).astype(np.float32)
        owners_array = np.array(owners, dtype=np.str_)
        index = {}
        for position, regno in enumerate(owners):
            index.setdefault(regno, []).append(position)
        return matrix, owners_array, {k: np.array(v) for k, v in index.items()}

    def test_identical_embedding_scores_one(self):
        self.assertAlmostEqual(
            cosine_similarity(self.identities[self.target][0], self.identities[self.target][0]),
            1.0,
            places=5,
        )

    def test_near_helper_produces_requested_similarity(self):
        base = unit_vector()
        for wanted in (0.3, 0.55, 0.8):
            self.assertAlmostEqual(cosine_similarity(base, near(base, wanted)), wanted, places=3)

    def test_matching_probe_accepts_the_right_student(self):
        matrix, owners, index = self._build(self.identities)
        probe = near(self.identities[self.target][0], 0.85)

        result = match_probe(
            probe,
            register_number=self.target,
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertEqual(result.status, "accepted")
        self.assertTrue(result.verified)

    def test_matching_rejects_an_impostor(self):
        matrix, owners, index = self._build(self.identities)
        result = match_probe(
            unit_vector(),  # unrelated face
            register_number=self.target,
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.verified)

    def test_review_band_is_recorded_not_rejected(self):
        matrix, owners, index = self._build(self.identities)
        probe = near(self.identities[self.target][0], 0.46)

        result = match_probe(
            probe,
            register_number=self.target,
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertEqual(result.status, "review")
        self.assertTrue(result.verified, "review-band students are recorded, then flagged")

    def test_unknown_registration_number_is_rejected(self):
        matrix, owners, index = self._build(self.identities)
        result = match_probe(
            unit_vector(),
            register_number="99XXX9999",
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.similarity, 0.0)

    def test_multiple_images_use_the_best_match(self):
        """The core reason to enroll 5-7 images instead of 1."""
        base = self.identities[self.target][0]
        # Simulate glasses-on / glasses-off references.
        self.identities[self.target] = [base, near(base, 0.6)]
        matrix, owners, index = self._build(self.identities)

        # A probe close to the SECOND reference but far from the first.
        probe = near(self.identities[self.target][1], 0.9)

        result = match_probe(
            probe,
            register_number=self.target,
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertEqual(result.status, "accepted")

    def test_best_of_frames_takes_the_highest(self):
        results = [
            MatchResult("rejected", 0.30, 0.5, 0.42),
            MatchResult("accepted", 0.71, 0.5, 0.42),
            MatchResult("review", 0.45, 0.5, 0.42),
        ]
        self.assertAlmostEqual(best_of_frames(results).similarity, 0.71)

    def test_best_of_frames_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            best_of_frames([])

    def test_diagnostics_include_nearest_other_but_caller_must_not_leak_it(self):
        matrix, owners, index = self._build(self.identities)
        result = match_probe(
            near(self.identities[self.target][0], 0.9),
            register_number=self.target,
            embeddings=matrix,
            owners=owners,
            index=index,
            threshold_accept=0.5,
            threshold_review=0.42,
        )
        self.assertIsNotNone(result.best_other_regno)
        self.assertNotEqual(result.best_other_regno, self.target)


class TestGalleryRoundTrip(unittest.TestCase):
    def test_save_and_load_preserves_identities(self):
        identities = {f"25BCE{1000 + i}": [unit_vector() for _ in range(3)] for i in range(5)}

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            config = PipelineConfig(gallery_dir=gallery_dir, model_name="buffalo_l")

            meta = gallery_module.save(
                gallery_dir / "face_db.npz", identities, model_name="buffalo_l"
            )
            self.assertEqual(meta["identities"], 5)
            self.assertEqual(meta["embeddings"], 15)

            loaded = gallery_module.load(config)
            self.assertEqual(loaded.identity_count, 5)
            self.assertEqual(loaded.embedding_count, 15)
            for regno in identities:
                self.assertTrue(loaded.has(regno))
                self.assertEqual(len(loaded.rows_for(regno)), 3)

    def test_stored_embeddings_are_normalised(self):
        identities = {"25BCE1000": [RNG.normal(size=512).astype(np.float32) * 47.0]}

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            config = PipelineConfig(gallery_dir=gallery_dir, model_name="buffalo_l")
            gallery_module.save(gallery_dir / "face_db.npz", identities, model_name="buffalo_l")

            loaded = gallery_module.load(config)
            norms = np.linalg.norm(loaded.embeddings, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_model_mismatch_is_refused_loudly(self):
        """The failure mode that would silently reject every student."""
        identities = {"25BCE1000": [unit_vector()]}

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            gallery_module.save(gallery_dir / "face_db.npz", identities, model_name="buffalo_s")

            config = PipelineConfig(gallery_dir=gallery_dir, model_name="buffalo_l")
            with self.assertRaises(gallery_module.GalleryError) as context:
                gallery_module.load(config)
            self.assertIn("not comparable", str(context.exception))

    def test_missing_gallery_raises_instead_of_degrading(self):
        """The old code silently re-embedded the whole dataset per request."""
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(gallery_dir=Path(tmp), model_name="buffalo_l")
            with self.assertRaises(gallery_module.GalleryError) as context:
                gallery_module.load(config)
            self.assertIn("build:gallery", str(context.exception))

    def test_empty_gallery_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(gallery_module.GalleryError):
                gallery_module.save(Path(tmp) / "face_db.npz", {}, model_name="buffalo_l")

    def test_legacy_pickle_still_loads(self):
        """Bridge for the currently-deployed face_db.pkl."""
        import pickle

        identities = {"25BCE1000": unit_vector(), "25BCE1001": unit_vector()}

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            with (gallery_dir / "face_db.pkl").open("wb") as handle:
                pickle.dump(identities, handle)

            config = PipelineConfig(gallery_dir=gallery_dir, model_name="buffalo_l")
            loaded = gallery_module.load(config)
            self.assertEqual(loaded.identity_count, 2)
            self.assertTrue(loaded.meta.get("legacy"))

    def test_npz_is_preferred_over_legacy_pickle(self):
        import pickle

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            with (gallery_dir / "face_db.pkl").open("wb") as handle:
                pickle.dump({"OLD00001": unit_vector()}, handle)
            gallery_module.save(
                gallery_dir / "face_db.npz",
                {"NEW00001": [unit_vector()]},
                model_name="buffalo_l",
            )

            config = PipelineConfig(gallery_dir=gallery_dir, model_name="buffalo_l")
            loaded = gallery_module.load(config)
            self.assertTrue(loaded.has("NEW00001"))
            self.assertFalse(loaded.has("OLD00001"))


class TestRealGalleryIfPresent(unittest.TestCase):
    """Runs against the actual deployed gallery when it exists."""

    def test_existing_gallery_loads_and_is_reported(self):
        config = PipelineConfig.from_env()
        if not (config.gallery_dir / "face_db.npz").exists() and not (
            config.gallery_dir / "face_db.pkl"
        ).exists():
            self.skipTest("no gallery present")

        loaded = gallery_module.load(config)
        self.assertGreater(loaded.identity_count, 0)
        self.assertEqual(loaded.embeddings.shape[1], config.embedding_dim)
        print(f"\n    [real gallery] {loaded.summary()}")

    def test_default_threshold_admits_no_known_impostor_pair(self):
        """The default must be SAFE, not merely conventional.

        This is the check that would have caught the shipped 0.50: in this
        gallery 25BRS1169 and 25BRS1286 score 0.5016 against each other, so
        0.50 let each of them mark the other present — a live false accept,
        the exact failure the system exists to prevent.

        Asserted against the REAL gallery rather than synthetic vectors,
        because the whole point is that a plausible-looking constant was
        wrong for this specific cohort. If a future re-enrollment introduces
        a closer pair, this fails and forces a recalibration instead of
        quietly degrading.
        """
        config = PipelineConfig.from_env()
        if not (config.gallery_dir / "face_db.npz").exists() and not (
            config.gallery_dir / "face_db.pkl"
        ).exists():
            self.skipTest("no gallery present")

        loaded = gallery_module.load(config)
        if loaded.identity_count < 2:
            self.skipTest("need at least two identities to have impostor pairs")

        similarity = loaded.embeddings @ loaded.embeddings.T
        different_identity = loaded.owners[:, None] != loaded.owners[None, :]
        upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)
        impostor = similarity[different_identity & upper]

        worst = float(impostor.max())
        colliding = int((impostor >= config.threshold_accept).sum())

        print(
            f"\n    [threshold] accept={config.threshold_accept} "
            f"worst_impostor={worst:.4f} colliding_pairs={colliding}"
        )

        self.assertEqual(
            colliding,
            0,
            f"{colliding} pair(s) of DIFFERENT students score >= the accept "
            f"threshold {config.threshold_accept} (worst {worst:.4f}). Each such "
            f"pair can verify as the other. Raise FACE_THRESHOLD_ACCEPT above "
            f"{worst:.4f}, or re-enroll those students with more images. "
            f"Run `npm run verify:gallery` to see which pairs.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
