"""Depth cleanup applied before TSDF integration.

LiDAR depth arrives low-resolution and upsampled, which introduces two errors
that fusion is bad at recovering from:

- sensor noise, which thickens surfaces and rounds off detail;
- "flying pixels" at depth discontinuities, where a pixel straddling a
  foreground/background edge is interpolated to a depth in between and lands in
  empty space, smearing object silhouettes into the background.

Both are cheaper to remove per frame than to clean up in the fused mesh.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _windows(image: np.ndarray, radius: int):
    """Yield (dy, dx, shifted view) for every offset in a square window.

    Edges are padded with NaN so out-of-bounds neighbours can be rejected by the
    caller rather than wrapping around or biasing the result toward the border.
    """
    padded = np.pad(image, radius, mode="constant", constant_values=np.nan)
    height, width = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            top = radius + dy
            left = radius + dx
            yield dy, dx, padded[top : top + height, left : left + width]


def bilateral_filter_depth(
    depth: np.ndarray,
    sigma_spatial: float = 2.0,
    sigma_depth: float = 0.02,
    radius: int = 2,
) -> np.ndarray:
    """Edge-preserving smoothing of a metric depth image.

    This is a masked normalised convolution rather than `cv2.bilateralFilter`,
    which has no concept of invalid pixels: passing it a depth map with zero
    holes lets those zeros bleed into neighbouring surfaces and drags real
    depths toward the camera. Here invalid pixels carry no weight, so holes stay
    holes and surfaces next to them keep their measured depth.

    Args:
        depth: metres, with 0 marking no measurement.
        sigma_spatial: falloff in pixels.
        sigma_depth: falloff in metres; sets how large a depth step is treated
            as an edge to preserve rather than noise to smooth.
        radius: half-window in pixels.

    Returns:
        Filtered depth in metres. Invalid pixels stay 0.
    """
    depth = depth.astype(np.float32)
    valid = depth > 0
    masked = np.where(valid, depth, np.nan)

    numerator = np.zeros_like(depth, dtype=np.float64)
    denominator = np.zeros_like(depth, dtype=np.float64)

    for dy, dx, neighbour in _windows(masked, radius):
        spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial**2))
        difference = neighbour - masked
        weight = spatial * np.exp(-(difference**2) / (2.0 * sigma_depth**2))

        # NaN marks either an invalid neighbour or an invalid centre pixel;
        # both should contribute nothing at all.
        weight = np.where(np.isfinite(weight), weight, 0.0)
        numerator += weight * np.nan_to_num(neighbour)
        denominator += weight

    filtered = np.zeros_like(depth)
    usable = valid & (denominator > 0)
    filtered[usable] = (numerator[usable] / denominator[usable]).astype(np.float32)
    return filtered


def flying_pixel_mask(
    depth: np.ndarray,
    max_step: float = 0.05,
    relative_step: float = 0.02,
    radius: int = 1,
) -> np.ndarray:
    """Mark pixels sitting on a depth discontinuity.

    A pixel is rejected when the depth range within its neighbourhood exceeds a
    threshold that grows with distance -- LiDAR noise scales with range, so a
    fixed step would over-reject far walls while under-rejecting near edges.

    Args:
        depth: metres, 0 for invalid.
        max_step: constant part of the threshold, in metres.
        relative_step: part proportional to depth (0.02 = 2 cm per metre).
        radius: half-window in pixels.

    Returns:
        Boolean mask, True where the pixel should be discarded.
    """
    valid = depth > 0
    window = 2 * radius + 1

    # Push invalid pixels to infinities so they never define the local range.
    high = np.where(valid, depth, -np.inf)
    low = np.where(valid, depth, np.inf)
    local_max = ndimage.maximum_filter(high, size=window, mode="nearest")
    local_min = ndimage.minimum_filter(low, size=window, mode="nearest")

    spread = local_max - local_min
    threshold = max_step + relative_step * depth
    return valid & np.isfinite(spread) & (spread > threshold)


def confidence_mask(confidence: np.ndarray, min_level: int = 1) -> np.ndarray:
    """Mark pixels whose ARKit confidence level is below `min_level`.

    ARKit grades each depth sample 0 (low), 1 (medium) or 2 (high). Integrating
    low-confidence samples at full weight is what produces billowing artefacts
    on windows, mirrors and dark surfaces.

    Returns:
        Boolean mask, True where the pixel should be discarded.
    """
    return confidence < min_level


def clean_depth(
    depth: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    bilateral: bool = True,
    drop_flying_pixels: bool = True,
    min_confidence: int = 1,
    max_depth: float = 5.0,
    sigma_spatial: float = 2.0,
    sigma_depth: float = 0.02,
    max_step: float = 0.05,
    relative_step: float = 0.02,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run the full per-frame cleanup, returning depth plus a rejection report.

    Order matters: discontinuities are found on the raw depth, because
    smoothing first would soften the very steps used to detect them.
    """
    depth = depth.astype(np.float32)
    initial_valid = int((depth > 0).sum())
    rejected: dict[str, float] = {}

    keep = depth > 0

    # LiDAR range is limited; beyond it readings are extrapolation, not signal.
    if max_depth is not None:
        too_far = keep & (depth > max_depth)
        rejected["beyond_max_depth"] = int(too_far.sum())
        keep &= ~too_far

    if drop_flying_pixels:
        flying = flying_pixel_mask(
            depth, max_step=max_step, relative_step=relative_step
        )
        rejected["flying_pixels"] = int((flying & keep).sum())
        keep &= ~flying

    if confidence is not None:
        low = confidence_mask(confidence, min_confidence)
        rejected["low_confidence"] = int((low & keep).sum())
        keep &= ~low

    cleaned = np.where(keep, depth, 0.0).astype(np.float32)

    if bilateral:
        cleaned = bilateral_filter_depth(
            cleaned, sigma_spatial=sigma_spatial, sigma_depth=sigma_depth
        )

    remaining = int((cleaned > 0).sum())
    rejected["kept"] = remaining
    rejected["kept_fraction"] = remaining / initial_valid if initial_valid else 0.0
    return cleaned, rejected
