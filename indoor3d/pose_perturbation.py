"""Perturb ground-truth poses to stand in for real pose uncertainty.

Phase V's decisive experiment (`hand_off_original_plan.md`) fuses identical
depth twice -- once with ground-truth poses, once with poses perturbed to
match how far off our real (COLMAP-refined) poses actually are, which
Phase 2 measured at a 7.5 mm median correction against ARKit. That figure is
what this module reproduces, not an arbitrary noise level.
"""

from __future__ import annotations

import numpy as np


def perturb_poses(
    poses: dict[int, np.ndarray],
    translation_std: float = 0.0075,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Add isotropic Gaussian translation noise to camera-to-world poses.

    Translation-only, matching the hand-off doc's literal spec ("poses
    perturbed by ~7.5 mm to match our real pose uncertainty") and the 7.5 mm
    median correction Phase 2 measured between COLMAP and ARKit. Orientation
    is left untouched: the doc's ask is specifically about position
    uncertainty, and rotation error's much larger effect on fused geometry
    (`hand_off_original_plan.md`'s Gotchas: 1 degree at 2 m displaces
    geometry 35 mm) is a deliberately separate question this experiment does
    not conflate with.

    Args:
        poses: frame id -> 4x4 camera-to-world.
        translation_std: standard deviation of the isotropic noise, metres.
        seed: makes the perturbation reproducible across runs that must
            compare against each other.

    Returns:
        A new dict, frame id -> perturbed 4x4 camera-to-world. `poses` is
        not modified.
    """
    rng = np.random.default_rng(seed)
    perturbed: dict[int, np.ndarray] = {}
    for frame_id in sorted(poses):
        pose = poses[frame_id].copy()
        pose[:3, 3] += rng.normal(0.0, translation_std, size=3)
        perturbed[frame_id] = pose
    return perturbed
