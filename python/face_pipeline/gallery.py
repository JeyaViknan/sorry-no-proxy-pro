"""Loading, saving and validating the enrollment gallery.

STORAGE FORMAT — .npz, NOT .pkl
-------------------------------
The gallery moved from pickle to NumPy's .npz. `pickle.load` executes
arbitrary code during deserialisation, so any path where the file could be
substituted (object storage, a CI artifact, a mounted volume) turns into
remote code execution. The gallery holds float matrices and strings; none of
that needs pickle's generality. .npz gives the same thing with no execution
surface, loads faster, and is roughly half the size.

The legacy .pkl is still readable so the currently-deployed database keeps
working until the new enrollment images arrive. That path logs a warning and
is documented as temporary.

VERSION GUARD
-------------
Embeddings from different models are NOT comparable. Swapping buffalo_l for
buffalo_s does not degrade accuracy gracefully — similarity collapses toward
random and *every* student is rejected, with no error raised anywhere. The
Dockerfile used to pre-warm buffalo_s while the code requested buffalo_l,
which is exactly the setup where someone "fixes" the inconsistency in the
wrong direction. The model name is now recorded inside the gallery and
checked on load, so that mistake fails loudly at startup instead of silently
at 9am in a lecture hall.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .matching import l2_normalize

FORMAT_VERSION = 2


class GalleryError(RuntimeError):
    """Raised when the gallery is missing, corrupt or model-mismatched."""


@dataclass
class Gallery:
    """An immutable, match-ready view of the enrollment data.

    embeddings : (N, D) float32, L2-normalised, one row per enrollment image
    owners     : (N,) unicode, the registration number owning each row
    index      : registration number -> row indices
    """

    embeddings: np.ndarray
    owners: np.ndarray
    index: dict
    meta: dict

    @property
    def identity_count(self) -> int:
        return len(self.index)

    @property
    def embedding_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def model_name(self) -> str:
        return str(self.meta.get("model", "unknown"))

    def has(self, register_number: str) -> bool:
        return register_number in self.index

    def rows_for(self, register_number: str) -> np.ndarray:
        return self.index.get(register_number, np.empty(0, dtype=np.int64))

    def summary(self) -> dict:
        per_identity = [len(rows) for rows in self.index.values()]
        return {
            "identities": self.identity_count,
            "embeddings": self.embedding_count,
            "model": self.model_name,
            "created_at": self.meta.get("created_at"),
            "images_per_identity": {
                "min": int(min(per_identity)) if per_identity else 0,
                "max": int(max(per_identity)) if per_identity else 0,
                "mean": round(float(np.mean(per_identity)), 2) if per_identity else 0.0,
            },
        }


def normalize_regno(value) -> str:
    return str(value).strip().upper()


def _build(identities: dict, meta: dict, expected_dim: int) -> Gallery:
    """Flatten {regno: [vec, ...]} into the matrix form used for matching."""
    rows: list[np.ndarray] = []
    owners: list[str] = []

    for regno in sorted(identities):
        for vector in identities[regno]:
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            if array.size != expected_dim:
                raise GalleryError(
                    f"{regno}: embedding has {array.size} dimensions, expected {expected_dim}. "
                    f"This usually means the gallery was built with a different model."
                )
            rows.append(l2_normalize(array))
            owners.append(regno)

    if not rows:
        raise GalleryError("gallery contains no embeddings")

    embeddings = np.vstack(rows).astype(np.float32)
    owners_array = np.array(owners, dtype=np.str_)

    index: dict = {}
    for position, regno in enumerate(owners):
        index.setdefault(regno, []).append(position)
    index = {regno: np.array(positions, dtype=np.int64) for regno, positions in index.items()}

    return Gallery(embeddings=embeddings, owners=owners_array, index=index, meta=meta)


def save(path: Path, identities: dict, *, model_name: str, extra_meta: Optional[dict] = None) -> dict:
    """Write the gallery in the .npz format. Returns the metadata written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []
    owners: list[str] = []
    for regno in sorted(identities):
        for vector in identities[regno]:
            rows.append(l2_normalize(np.asarray(vector, dtype=np.float32)))
            owners.append(regno)

    if not rows:
        raise GalleryError("refusing to save an empty gallery")

    embeddings = np.vstack(rows).astype(np.float32)
    meta = {
        "format_version": FORMAT_VERSION,
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "identities": len(identities),
        "embeddings": int(embeddings.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra_meta or {}),
    }

    np.savez_compressed(
        path,
        embeddings=embeddings,
        owners=np.array(owners, dtype=np.str_),
        meta=np.array(json.dumps(meta), dtype=np.str_),
    )
    return meta


def _load_npz(path: Path, expected_model: str, expected_dim: int) -> Gallery:
    with np.load(path, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        owners = np.asarray(data["owners"], dtype=np.str_)
        meta = json.loads(str(data["meta"]))

    if meta.get("model") != expected_model:
        raise GalleryError(
            f"gallery was built with model '{meta.get('model')}' but the verifier is "
            f"configured for '{expected_model}'. Embeddings are not comparable across "
            f"models — every student would be rejected. Rebuild with: "
            f"npm run build:gallery"
        )
    if embeddings.shape[1] != expected_dim:
        raise GalleryError(
            f"gallery embeddings are {embeddings.shape[1]}-dimensional, expected {expected_dim}"
        )
    if embeddings.shape[0] != owners.shape[0]:
        raise GalleryError("gallery is corrupt: embedding and owner counts differ")

    # Re-normalise defensively; costs microseconds, removes a class of bugs.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / np.maximum(norms, 1e-10)).astype(np.float32)

    index: dict = {}
    for position, regno in enumerate(owners.tolist()):
        index.setdefault(regno, []).append(position)
    index = {regno: np.array(positions, dtype=np.int64) for regno, positions in index.items()}

    return Gallery(embeddings=embeddings, owners=owners, index=index, meta=meta)


def _load_legacy_pickle(path: Path, expected_model: str, expected_dim: int) -> Gallery:
    """Read the pre-v2 pickle: {regno: ndarray} or {regno: [ndarray, ...]}.

    TEMPORARY. Kept only so the currently-deployed gallery keeps working until
    the new enrollment dataset is collected. Remove once build_gallery.py has
    produced a .npz. See docs/ENROLLMENT.md.
    """
    import pickle  # noqa: S403 — legacy path only, file is a local build artifact

    print(
        f"[gallery] WARNING: loading legacy pickle {path.name}. "
        f"Rebuild as .npz with `npm run build:gallery` — pickle deserialisation "
        f"executes arbitrary code and should not survive into production.",
        file=sys.stderr,
    )

    with path.open("rb") as handle:
        raw = pickle.load(handle)

    if not isinstance(raw, dict):
        raise GalleryError("legacy face_db.pkl must contain a dict of regno -> embedding(s)")

    identities: dict = {}
    for regno_raw, value in raw.items():
        regno = normalize_regno(regno_raw)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            vectors = [value] if value.ndim == 1 else list(value)
        elif isinstance(value, (list, tuple)):
            vectors = [np.asarray(v) for v in value if v is not None]
        else:
            continue
        vectors = [v for v in vectors if getattr(v, "size", 0) == expected_dim]
        if vectors:
            identities[regno] = vectors

    if not identities:
        raise GalleryError("legacy face_db.pkl produced no usable embeddings")

    meta = {
        "format_version": 1,
        "model": expected_model,  # legacy files carry no model tag; assume configured
        "embedding_dim": expected_dim,
        "created_at": None,
        "legacy": True,
    }
    return _build(identities, meta, expected_dim)


def load(config) -> Gallery:
    """Load the gallery, preferring .npz and falling back to the legacy pickle.

    Raises GalleryError with an actionable message rather than degrading
    silently. The old code fell back to re-embedding every image in data/ on
    *every request* when face_db.pkl was missing — a 20x slowdown that logged
    only to stderr and alarmed nothing.
    """
    npz_path = config.gallery_dir / "face_db.npz"
    pkl_path = config.gallery_dir / "face_db.pkl"

    if npz_path.exists():
        return _load_npz(npz_path, config.model_name, config.embedding_dim)

    if pkl_path.exists():
        return _load_legacy_pickle(pkl_path, config.model_name, config.embedding_dim)

    raise GalleryError(
        f"no gallery found in {config.gallery_dir}. Expected face_db.npz. "
        f"Build it with: npm run build:gallery"
    )
