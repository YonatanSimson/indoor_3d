"""Pin a predicted dense depth field to the depths the sensor actually measured.

Any dense predictor -- a bilateral solver, a monocular network, a stereo matcher
-- produces a field that disagrees with the LiDAR wherever both have an opinion.
Measured in practice: the fast bilateral solver moved confident readings by
110 mm, and a monocular network can be off by a scale factor entirely.

Where the sensor gave a confident reading there is nothing to infer. The
predictor's value is its *structure* -- edges in the right place, plausible
geometry across holes -- not its numbers. So the predictor is corrected to pass
through the measurements, in two steps:

1. an affine fit, for predictors whose depth is only defined up to scale and
   offset (most monocular networks). Skipped when the predictor is already
   metric.
2. a residual correction, which pins the remaining per-pixel disagreement.
   Simply overwriting measured pixels would leave the corrected and predicted
   values disagreeing across every anchor boundary; diffusing the residual
   spreads the correction smoothly instead.

Low-confidence pixels are deliberately not anchored -- those are the readings
worth overriding, which is what a confidence map is for.
"""

from __future__ import annotations

import cv2
import numpy as np


def fit_affine(
    predicted: np.ndarray, measured: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """Least-squares (scale, offset) mapping predicted depth onto measured.

    Fitted on inliers only: a handful of wild predictions, which monocular
    networks produce on windows and mirrors, would otherwise drag the fit.
    """
    p = predicted[mask].astype(np.float64)
    m = measured[mask].astype(np.float64)
    if len(p) < 16:
        return 1.0, 0.0

    scale, offset = np.polyfit(p, m, 1)
    for _ in range(3):
        residual = np.abs(scale * p + offset - m)
        keep = residual < max(3.0 * np.median(residual), 1e-3)
        if keep.sum() < 16:
            break
        scale, offset = np.polyfit(p[keep], m[keep], 1)
    return float(scale), float(offset)


def anchor_to_measurements(
    predicted: np.ndarray,
    measured: np.ndarray,
    confidence: np.ndarray | None = None,
    min_confidence: int = 1,
    fit_scale: bool = False,
    smooth_px: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Correct `predicted` so it agrees with `measured` where measurement exists.

    Both arrays must already be on the same grid. Returns the corrected depth
    and a report describing how much correction was needed -- a large affine
    scale or a large residual is a signal the predictor disagrees badly, which
    is worth seeing rather than silently absorbing.
    """
    predicted = predicted.astype(np.float32)
    measured = measured.astype(np.float32)

    anchor = (measured > 0) & (predicted > 0)
    if confidence is not None:
        anchor &= confidence >= min_confidence

    report: dict[str, float] = {"anchor_fraction": float(anchor.mean())}
    if not anchor.any():
        report["scale"] = 1.0
        report["offset"] = 0.0
        report["residual_mm"] = float("nan")
        return predicted, report

    corrected = predicted
    scale, offset = 1.0, 0.0
    if fit_scale:
        scale, offset = fit_affine(predicted, measured, anchor)
        corrected = np.where(predicted > 0, corrected * scale + offset, 0.0)
    report["scale"] = scale
    report["offset"] = offset

    residual = np.zeros_like(corrected)
    residual[anchor] = measured[anchor] - corrected[anchor]
    report["residual_mm"] = float(np.median(np.abs(residual[anchor])) * 1000)

    # Diffuse the residual so the correction varies smoothly between anchors.
    kernel = smooth_px or max(3, int(round(min(predicted.shape) / 48)) | 1)
    weight = anchor.astype(np.float32)
    blurred_residual = cv2.blur(residual, (kernel, kernel))
    blurred_weight = cv2.blur(weight, (kernel, kernel))

    correction = np.zeros_like(residual)
    usable = blurred_weight > 1e-6
    correction[usable] = blurred_residual[usable] / blurred_weight[usable]

    out = corrected + correction
    # Exact at the anchors themselves, which the blur only approximates.
    out[anchor] = measured[anchor]
    return np.where(predicted > 0, out, 0.0).astype(np.float32), report
