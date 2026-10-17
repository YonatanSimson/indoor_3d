"""Cache for precomputed Depth Anything 3 monocular predictions.

DA3 needs a torch cu128 environment (`depth`), incompatible with `indoor3d`'s
own conda env. Splitting inference (write side, `scripts/precompute_da3_depth.py`,
run once under `depth`) from anchoring (read side, `FusionConfig.depth_prior_provider`,
used from `indoor3d.fusion.fuse`) means the fusion pipeline never needs torch
at all -- it just reads cached predictions off disk, the same way it already
reads the dataset's own depth PNGs.

Stored as uint16 PNGs at the dataset's own TUM scale (value / 5000 = metres)
-- mainstream format, lossless to 0.2mm (well under any noise level this
project has measured), and consistent with how `ICLNUIMSequence.load_depth`
already reads ground truth, rather than inventing a new depth encoding for
one extra source.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Matches indoor3d.data_iclnuim._DEPTH_SCALE -- same TUM convention, so a
# cached prediction round-trips through the identical decode path as any
# other depth PNG in this dataset.
_DEPTH_SCALE = 5000.0


def cache_path(cache_dir: Path, frame_id: int) -> Path:
    return cache_dir / f"{frame_id:06d}.png"


def write_depth(path: Path, depth: np.ndarray) -> None:
    """Encode metres as uint16 at TUM scale and write a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(depth * _DEPTH_SCALE, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"failed to write depth cache {path}")


def read_depth(path: Path) -> np.ndarray:
    """Inverse of `write_depth`: float32 metres, 0 marking "no prediction"."""
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(f"no cached depth at {path}")
    return (encoded.astype(np.float32)) / _DEPTH_SCALE


class CachedDepthProvider:
    """Reads precomputed depth off disk, keyed by `frame.frame_id`.

    Matches the `FusionConfig.depth_prior_provider` signature: pass an
    instance of this directly as that field. Built for `ICLNUIMFrame` (the
    only loader with cached DA3 predictions so far), but any frame record
    with a `.frame_id` attribute works.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"no depth prediction cache at {self.cache_dir}")

    def __call__(self, frame: object) -> np.ndarray:
        frame_id = getattr(frame, "frame_id", None)
        if frame_id is None:
            raise AttributeError(
                f"{type(frame).__name__} has no .frame_id; CachedDepthProvider "
                "needs a frame record with a frame_id attribute"
            )
        return read_depth(cache_path(self.cache_dir, frame_id))
