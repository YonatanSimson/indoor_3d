"""Degrade clean ground-truth depth into something that looks like our LiDAR.

Phase V's simulated-sensor step: ICL-NUIM ships clean 640x480 depth, but the
pipeline this repo built was tuned against ARKit's 256x192 LiDAR, whose two
defining traits -- coarser sampling and per-pixel confidence gating -- have no
counterpart in the raw ICL-NUIM data. Fusing the clean depth directly would
validate a pipeline the real capture never runs: confidence gating in
particular "does real work" (see `hand_off_original_plan.md`) only if some
pixels actually fail it.

Neither the range-dependent noise curve nor the confidence heuristic below is
calibrated against real ARKit output -- no ground truth for that exists in
this offline setting. They are documented approximations whose only job is to
exercise the same code paths (`depth_filters.clean_depth`'s confidence
gating, `depth_upsampling`'s use of a confidence map) that the real capture
exercises, not to reproduce ARKit's noise to the millimetre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .data import PosedSequence
from .depth_filters import flying_pixel_mask

# `DegradedSequence` runs cv2 (dilate/erode/resize) hundreds of times in a
# tight loop, which leaves OpenCV's own thread pool alive afterward. On this
# Windows/conda stack that pool contends with Open3D's threading for any
# `open3d` call made later in the same process -- reproduced twice: a
# `load_ground_truth_tree` call that normally takes ~30s (confirmed by
# timing it in isolation) instead spun at high CPU for 15+ minutes with no
# progress when it ran shortly after a `DegradedSequence` construction.
# Per-frame image ops here are cheap regardless of thread count, so
# single-threading cv2 costs little and removes the interaction entirely.
cv2.setNumThreads(1)


@dataclass
class SensorSimConfig:
    """Degradation parameters. Defaults describe our real capture's grid."""

    target_width: int = 256
    target_height: int = 192

    # Range-dependent noise: sigma(z) = noise_base + noise_growth * z^2. The
    # z^2 growth is the qualitative shape reported for structured-light/ToF
    # depth sensors (e.g. Nguyen et al. 2012 for Kinect v1); no ARKit-specific
    # curve is available to calibrate against, so this is deliberately only a
    # shape match, not a measured one. Magnitude is set so noise stays in the
    # single-digit-mm range at this room's typical 1-2m depths and only grows
    # large at the far (~4m) end -- an initial 0.004 growth coefficient put
    # sigma at 17mm by 2m, which pushed nearly every pixel to low confidence
    # (see the confidence thresholds below) and would have made this sensor
    # model itself the noise floor, not the pose-vs-depth question the
    # pose-ceiling experiment exists to answer.
    noise_base: float = 0.0015  # metres, at z -> 0
    noise_growth: float = 0.0007  # metres, coefficient on z^2

    # Synthetic confidence. A pixel is graded level 2 (high) unless it sits on
    # a sharp local depth discontinuity (oblique / edge geometry) or a dark
    # RGB region (low return signal) -- the same two failure modes
    # `README.md` documents for real ARKit confidence -- in which case it
    # drops to level 1 or 0. "Sharp" is decided by reusing
    # `depth_filters.flying_pixel_mask` itself at two thresholds, rather than
    # inventing a second, parallel notion of what counts as an edge: the
    # tight one (its own clean_depth defaults) marks level 0, since a pixel
    # that bad would be rejected outright anyway; a looser one at double the
    # threshold catches the surrounding, merely-oblique ring as level 1.
    edge_max_step: float = 0.05
    edge_relative_step: float = 0.02
    darkness_low: int = 40  # mean RGB below this -> level 1


def _downsample_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Subsample to a coarser grid, one source pixel per output cell.

    An earlier version used `cv2.INTER_AREA` (a weighted average of every
    source pixel under each output cell), which is wrong for what this is
    simulating: a real depth sensor with fewer, discrete sampling points
    does not optically blend two surfaces together at an edge, it reads one
    point per detector. Averaging invents exactly the "flying pixel" defect
    `depth_filters.py` exists to filter out -- confirmed empirically at real
    depth discontinuities in this dataset: `INTER_AREA` fabricated a depth
    value belonging to neither surface at 6 of 8 sampled edges, against 1 of
    8 (a boundary rounding case, not a real blend) for nearest-neighbour
    subsampling used here. `INTER_NEAREST` never averages, so holes (0)
    stay holes with no separate masking needed.
    """
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def _synthesize_confidence(
    depth: np.ndarray, rgb: np.ndarray, config: SensorSimConfig
) -> np.ndarray:
    """Grade each pixel 0/1/2, matching ARKit's confidence scale."""
    valid = depth > 0
    # `reject` uses clean_depth's own thresholds: a pixel that bad would be
    # dropped by flying-pixel rejection regardless, so level 0 is consistent
    # with what the rest of the pipeline already does with it. `caution`
    # halves both thresholds -- more sensitive, so it additionally catches
    # the milder, merely-oblique ring around those edges as level 1.
    reject = flying_pixel_mask(
        depth, max_step=config.edge_max_step, relative_step=config.edge_relative_step
    )
    caution = flying_pixel_mask(
        depth,
        max_step=config.edge_max_step / 2,
        relative_step=config.edge_relative_step / 2,
    )

    gray = rgb.astype(np.float32).mean(axis=2)
    if gray.shape != depth.shape:
        gray = cv2.resize(
            gray, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
        )

    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    confidence[caution | (gray < config.darkness_low + 20)] = 1
    confidence[reject | (gray < config.darkness_low)] = 0
    confidence[~valid] = 0
    return confidence


def degrade_frame(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    config: SensorSimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn clean 640x480 GT depth into simulated-LiDAR depth + confidence.

    Args:
        depth: clean depth, metres, native resolution.
        rgb: colour image at the same resolution as `depth`.
        intrinsics: 3x3 matrix describing `depth`'s grid.
        rng: shared across frames by the caller for a reproducible run.

    Returns:
        (degraded_depth, confidence, degraded_intrinsics) on the target grid.
    """
    config = config or SensorSimConfig()
    rng = rng or np.random.default_rng(0)

    scale_x = config.target_width / depth.shape[1]
    scale_y = config.target_height / depth.shape[0]

    coarse = _downsample_depth(depth, config.target_width, config.target_height)

    valid = coarse > 0
    sigma = config.noise_base + config.noise_growth * coarse**2
    noisy = coarse + np.where(valid, rng.normal(0.0, sigma), 0.0).astype(np.float32)
    noisy = np.where(valid & (noisy > 0), noisy, 0.0).astype(np.float32)

    guide = cv2.resize(
        rgb, (config.target_width, config.target_height), interpolation=cv2.INTER_AREA
    )
    confidence = _synthesize_confidence(noisy, guide, config)

    degraded_intrinsics = intrinsics.copy()
    degraded_intrinsics[0, :] *= scale_x
    degraded_intrinsics[1, :] *= scale_y

    return noisy, confidence, degraded_intrinsics


class DegradedSequence:
    """Wrap a clean sequence so `fusion.fuse` sees simulated-LiDAR depth.

    Precomputes `degrade_frame` once per frame at construction time and
    caches the result, keyed by `frame.frame_id` -- so two `fuse()` runs
    against the same `DegradedSequence` (e.g. one per pose hypothesis in
    `scripts/eval_iclnuim_pose_ceiling.py`) see byte-identical degraded depth,
    which is the whole point of that experiment ("fuse identical depth at
    5 mm twice"). Only depth/confidence/intrinsics are intercepted; RGB and
    pose lookups pass straight through to the wrapped sequence, so an
    override such as `ICLNUIMSequence.use_perturbed_poses` still takes effect
    on every subsequent `fuse()` call without needing to rebuild this wrapper.

    Requires the wrapped sequence's frame type to expose `.frame_id` --
    true of `ICLNUIMSequence`, not assumed for `Sequence` or
    `RawRecordingSequence`, so this is Phase-V-specific rather than a
    general-purpose decorator for all three loaders.
    """

    def __init__(
        self,
        sequence: PosedSequence,
        config: SensorSimConfig | None = None,
        seed: int = 0,
        progress_every: int = 100,
    ):
        self._sequence = sequence
        self.config = config or SensorSimConfig()
        rng = np.random.default_rng(seed)

        self._depth: dict[int, np.ndarray] = {}
        self._confidence: dict[int, np.ndarray] = {}
        self._intrinsics: dict[int, tuple[np.ndarray, int, int]] = {}

        for count, index in enumerate(range(len(sequence))):
            frame = sequence[index]
            frame_id = frame.frame_id
            depth = sequence.load_depth(frame)
            rgb = sequence.load_rgb(frame)
            intrinsics, _, _ = sequence.intrinsics(frame)

            degraded_depth, confidence, degraded_intrinsics = degrade_frame(
                depth, rgb, intrinsics, self.config, rng
            )
            self._depth[frame_id] = degraded_depth
            self._confidence[frame_id] = confidence
            self._intrinsics[frame_id] = (
                degraded_intrinsics,
                self.config.target_width,
                self.config.target_height,
            )

            if progress_every and count % progress_every == 0:
                print(f"  degraded {count + 1}/{len(sequence)} frames", flush=True)

    # -- PosedSequence interface -------------------------------------------

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index: int) -> Any:
        return self._sequence[index]

    @property
    def has_confidence(self) -> bool:
        return True

    def load_rgb(self, frame: Any) -> np.ndarray:
        return self._sequence.load_rgb(frame)

    def load_depth(self, frame: Any) -> np.ndarray:
        return self._depth[frame.frame_id]

    def load_confidence(self, frame: Any) -> np.ndarray | None:
        return self._confidence[frame.frame_id]

    def intrinsics(self, frame: Any) -> tuple[np.ndarray, int, int]:
        return self._intrinsics[frame.frame_id]

    def world_to_camera(self, frame: Any, traj_convention: str = "rdf") -> np.ndarray:
        return self._sequence.world_to_camera(frame, traj_convention)
