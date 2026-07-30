"""Embedding comparison and the accept / review / reject decision.

Pure NumPy — no model dependency, so the decision logic is unit testable.

TWO CHANGES FROM THE OLD IMPLEMENTATION
---------------------------------------
1. VECTORISED MATCHING. The old code looped over the gallery dict in Python,
   calling max() per student. Here every embedding lives in one contiguous
   (N, 512) matrix, so scoring the whole cohort is a single matmul: ~1 ms for
   thousands of identities instead of a loop that grew linearly in Python.

2. THE IDENTITY ORACLE IS GONE. The old response included
   `"best match: {best_regno}"` — an unauthenticated endpoint that told the
   caller which *other* student a photo resembled. Upload arbitrary photos,
   have the server identify students by registration number. Cross-gallery
   scores are still computed (they are nearly free once vectorised, and they
   are genuinely useful for spotting lookalikes and mis-enrollments) but they
   are returned only in the diagnostics channel, which the Node layer logs
   and never forwards to a client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPSILON = 1e-10


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return (array / (np.linalg.norm(array) + EPSILON)).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for arbitrary (not necessarily normalised) vectors."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = (np.linalg.norm(a) * np.linalg.norm(b)) + EPSILON
    return float(np.dot(a, b) / denominator)


@dataclass
class MatchResult:
    """Outcome of comparing one probe against the gallery.

    `status` is one of: accepted | review | rejected
    """

    status: str
    similarity: float
    threshold_accept: float
    threshold_review: float

    # Diagnostics — logged server-side, never returned to a client.
    best_other_regno: Optional[str] = None
    best_other_similarity: float = 0.0
    margin: float = 0.0

    @property
    def verified(self) -> bool:
        return self.status in ("accepted", "review")


def match_probe(
    probe: np.ndarray,
    *,
    register_number: str,
    embeddings: np.ndarray,
    owners: np.ndarray,
    index: dict,
    threshold_accept: float,
    threshold_review: float,
) -> MatchResult:
    """Score one probe embedding against the gallery.

    This is 1:1 verification — the entered registration number must clear the
    threshold on its own. Identification (who does this look like most?) is
    computed alongside purely as a diagnostic.
    """
    probe = l2_normalize(probe)

    rows = index.get(register_number)
    if rows is None or len(rows) == 0:
        return MatchResult(
            status="rejected",
            similarity=0.0,
            threshold_accept=threshold_accept,
            threshold_review=threshold_review,
        )

    # One matmul over the whole gallery.
    all_scores = embeddings @ probe

    claimed_scores = all_scores[rows]
    claimed = float(claimed_scores.max())

    # Best score belonging to a *different* registration number.
    mask = owners != register_number
    if np.any(mask):
        other_scores = all_scores[mask]
        best_other_position = int(np.argmax(other_scores))
        best_other_similarity = float(other_scores[best_other_position])
        best_other_regno = str(owners[mask][best_other_position])
    else:
        best_other_similarity = 0.0
        best_other_regno = None

    if claimed >= threshold_accept:
        status = "accepted"
    elif claimed >= threshold_review:
        status = "review"
    else:
        status = "rejected"

    return MatchResult(
        status=status,
        similarity=claimed,
        threshold_accept=threshold_accept,
        threshold_review=threshold_review,
        best_other_regno=best_other_regno,
        best_other_similarity=best_other_similarity,
        margin=claimed - best_other_similarity,
    )


def best_of_frames(results: list[MatchResult]) -> MatchResult:
    """Pick the strongest result from a burst.

    Taking the max over frames reduces false rejections substantially — a
    single blurred or blinking frame no longer decides the outcome — while
    barely moving the false-acceptance rate, because five frames of an
    impostor are still five frames of an impostor. This is the highest
    accuracy-per-line change available in the pipeline.
    """
    if not results:
        raise ValueError("no frame results to choose from")
    return max(results, key=lambda r: r.similarity)
