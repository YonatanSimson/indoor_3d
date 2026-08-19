"""Phase 3: raise depth to image resolution using RGB as a guide.

The two streams are wildly mismatched -- 256x192 of LiDAR against 1920x1440 of
RGB, a 56x gap in pixel count -- and fusion currently takes all its geometry
from the smaller one. Plain bilinear upsampling adds pixels but no information:
edges land wherever the coarse grid put them, so object silhouettes stay soft
and bleed into the background.

Joint upsampling instead lets the image say where the edges are. Colour changes
sharply at an object boundary, so pixels that differ in colour are kept from
averaging their depths together, and a depth discontinuity ends up aligned with
the visible edge rather than a grid cell.

Two methods, both from OpenCV's ximgproc:

- the **fast bilateral solver**, which takes a per-pixel confidence and solves
  globally for a smooth depth that agrees with the confident measurements. It
  fills holes properly, and ARKit's confidence map feeds it directly.
- **joint bilateral upsampling**, a local filter: cheaper, no confidence, and it
  cannot invent depth where there is none.
"""

from __future__ import annotations

import cv2
import numpy as np


def _guide_from_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB to the target grid as the 8-bit guide image."""
    if rgb.shape[1] != width or rgb.shape[0] != height:
        # INTER_AREA when shrinking avoids aliasing the very edges that are the
        # whole point of the guide.
        interpolation = cv2.INTER_AREA if rgb.shape[1] > width else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(rgb)


def upsample_bilinear(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Plain bilinear upsampling, the baseline this phase aims to beat.

    Nearest-neighbour handling of holes matters: interpolating across a zero
    would invent depth halfway to the camera.
    """
    valid = (depth > 0).astype(np.uint8)
    scaled = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(valid, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, scaled, 0.0).astype(np.float32)


def upsample_joint_bilateral(
    depth: np.ndarray,
    rgb: np.ndarray,
    width: int,
    height: int,
    sigma_spatial: float = 8.0,
    sigma_colour: float = 12.0,
) -> np.ndarray:
    """Guided upsampling with a local joint-bilateral filter.

    Holes stay holes: the filter is applied to depth and to the validity mask
    with identical weights, then divided, so pixels with no nearby measurement
    end up at zero instead of borrowing depth from across an edge.
    """
    # jointBilateralFilter requires the guide and source to share a dtype, and
    # depth is float32, so the guide is converted rather than the depth
    # quantised to 8-bit. sigmaColour stays in 0-255 units.
    guide = _guide_from_rgb(rgb, width, height).astype(np.float32)
    coarse = upsample_bilinear(depth, width, height)
    mask = (coarse > 0).astype(np.float32)

    filtered = cv2.ximgproc.jointBilateralFilter(
        guide, coarse * mask, d=-1, sigmaColor=sigma_colour, sigmaSpace=sigma_spatial
    )
    weight = cv2.ximgproc.jointBilateralFilter(
        guide, mask, d=-1, sigmaColor=sigma_colour, sigmaSpace=sigma_spatial
    )

    out = np.zeros_like(filtered)
    usable = weight > 1e-3
    out[usable] = filtered[usable] / weight[usable]
    return np.where(coarse > 0, out, 0.0).astype(np.float32)


def upsample_bilateral_solver(
    depth: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray | None,
    width: int,
    height: int,
    sigma_spatial: float = 8.0,
    sigma_luma: float = 8.0,
    sigma_chroma: float = 8.0,
    lambda_: float = 128.0,
    num_iter: int = 25,
) -> np.ndarray:
    """Guided upsampling with the fast bilateral solver.

    Unlike a local filter this solves globally, so depth propagates across holes
    from whatever confident measurements surround them, guided by image edges.
    ARKit's confidence map is used directly as the per-pixel weight, which is
    what stops the low-confidence returns on glossy and dark surfaces from
    dragging their neighbourhoods with them.
    """
    guide = _guide_from_rgb(rgb, width, height)
    coarse = upsample_bilinear(depth, width, height)

    if confidence is not None:
        # ARKit grades 0/1/2; map to a weight that ignores level 0 entirely and
        # trusts level 2 fully.
        weights = np.take(np.array([0.0, 0.5, 1.0], np.float32), confidence.astype(np.int32))
        weights = cv2.resize(weights, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        weights = np.ones((height, width), np.float32)
    weights = np.where(coarse > 0, weights, 0.0).astype(np.float32)

    if not np.any(weights > 0):
        return coarse

    solver = cv2.ximgproc.createFastBilateralSolverFilter(
        guide, sigma_spatial, sigma_luma, sigma_chroma, lambda_, num_iter
    )
    solved = solver.filter(coarse.astype(np.float32), weights)

    # The solver extrapolates everywhere, including well beyond the measured
    # region; keep only pixels near real support so it cannot invent surfaces.
    support = cv2.dilate(
        (weights > 0).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1
    )
    return np.where(support > 0, solved, 0.0).astype(np.float32)


def upsample_anchored(
    depth: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray | None,
    width: int,
    height: int,
    min_confidence: int = 1,
    **solver_kwargs,
) -> np.ndarray:
    """Solve for the gaps, but keep measured depths exactly as measured.

    The bilateral solver is a regressor: it fits a smooth field everywhere,
    including over pixels where the sensor already gave an answer, and moves
    those by ~110 mm in practice. That is the wrong behaviour for an upsampler.
    Where the LiDAR reported a confident reading there is nothing to infer, and
    changing it discards the only real information in the frame.

    So the solver's output is used for its structure -- edges aligned to the
    image, holes filled -- and then corrected so it passes exactly through the
    measurements. The correction is the residual at measured pixels, smoothly
    spread over the rest of the grid, which pins the data points without
    introducing a seam at each one: simply overwriting them would leave the
    solved field and the measurements disagreeing across every boundary.

    Low-confidence pixels are deliberately not anchored. Those readings are the
    ones worth overriding, which is what the confidence map is for.
    """
    solved = upsample_bilateral_solver(
        depth, rgb, confidence, width, height, **solver_kwargs
    )

    anchor = depth > 0
    if confidence is not None:
        anchor &= confidence >= min_confidence
    if not anchor.any():
        return solved

    # Measurements placed on the fine grid, nearest-neighbour so values are
    # carried across unaltered.
    measured = cv2.resize(
        np.where(anchor, depth, 0.0).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    anchor_fine = cv2.resize(
        anchor.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0
    anchor_fine &= solved > 0

    if not anchor_fine.any():
        return solved

    # Residual where the solver disagrees with the sensor, then diffused so the
    # correction is smooth between anchors rather than a per-pixel patch.
    residual = np.zeros((height, width), np.float32)
    residual[anchor_fine] = measured[anchor_fine] - solved[anchor_fine]

    weight = anchor_fine.astype(np.float32)
    kernel = max(3, int(round(width / depth.shape[1])) * 4 + 1)
    blurred_residual = cv2.blur(residual, (kernel, kernel))
    blurred_weight = cv2.blur(weight, (kernel, kernel))

    correction = np.zeros_like(residual)
    usable = blurred_weight > 1e-6
    correction[usable] = blurred_residual[usable] / blurred_weight[usable]

    corrected = solved + correction
    # Exactness at the anchors themselves, which the blur only approximates.
    corrected[anchor_fine] = measured[anchor_fine]
    return np.where(solved > 0, corrected, 0.0).astype(np.float32)


def upsample(
    depth: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray | None,
    width: int,
    height: int,
    method: str = "solver",
) -> np.ndarray:
    """Dispatch to one of the upsampling methods.

    `method` is one of "bilinear", "joint", "solver", "anchored", or "none".
    """
    if method == "none" or (width, height) == (depth.shape[1], depth.shape[0]):
        return depth.astype(np.float32)
    if method == "bilinear":
        return upsample_bilinear(depth, width, height)
    if method == "joint":
        return upsample_joint_bilateral(depth, rgb, width, height)
    if method == "solver":
        return upsample_bilateral_solver(depth, rgb, confidence, width, height)
    if method == "anchored":
        return upsample_anchored(depth, rgb, confidence, width, height)
    raise ValueError(f"unknown upsampling method {method!r}")
