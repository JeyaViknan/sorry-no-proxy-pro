"""Face verification pipeline.

Module layout is deliberate — the pure-NumPy parts carry no model dependency
so they can be tested in milliseconds without downloading buffalo_l:

    config    thresholds and paths, driven by the Node process's environment
    quality   frame quality assessment          (pure NumPy, tested)
    matching  similarity + accept/review/reject (pure NumPy, tested)
    gallery   enrollment data load/save/verify  (pure NumPy, tested)
    engine    InsightFace wrapper               (the only heavy import)
"""

from .config import PipelineConfig, QualityThresholds
from .gallery import Gallery, GalleryError

__all__ = ["PipelineConfig", "QualityThresholds", "Gallery", "GalleryError"]
