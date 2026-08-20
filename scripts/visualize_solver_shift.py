"""Diagnostic: show that the raw bilateral `solver` upsampler moves already-
measured depth, and that `anchored` fixes it.

Picks one frame, cleans its depth the same way fusion does, then upsamples it
with both methods at the same target resolution and compares each against the
original measurement at the pixels where the sensor actually reported data.

Usage:
    python scripts/visualize_solver_shift.py [--frame 250] [--session PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample_anchored, upsample_bilateral_solver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument("--frame", type=int, default=250)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--min-confidence", type=int, default=1)
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    frame = sequence[args.frame]

    raw_depth = sequence.load_depth(frame)
    confidence = sequence.load_confidence(frame)
    rgb = sequence.load_rgb(frame)

    depth, report = clean_depth(
        raw_depth, confidence, min_confidence=args.min_confidence
    )
    print(f"frame {args.frame}: {report}")

    target_w = int(round(depth.shape[1] * args.scale))
    target_h = int(round(depth.shape[0] * args.scale))

    solver_depth = upsample_bilateral_solver(depth, rgb, confidence, target_w, target_h)
    anchored_depth = upsample_anchored(
        depth, rgb, confidence, target_w, target_h, min_confidence=args.min_confidence
    )

    # Where the sensor actually measured something, upsampled to the fine grid
    # with nearest-neighbour so values are carried across unaltered.
    anchor_mask = depth > 0
    if confidence is not None:
        anchor_mask &= confidence >= args.min_confidence
    measured_fine = cv2.resize(
        np.where(anchor_mask, depth, 0.0).astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    anchor_fine = (
        cv2.resize(
            anchor_mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )

    solver_shift_mm = np.zeros((target_h, target_w), np.float32)
    solver_shift_mm[anchor_fine] = (
        solver_depth[anchor_fine] - measured_fine[anchor_fine]
    ) * 1000.0
    anchored_shift_mm = np.zeros((target_h, target_w), np.float32)
    anchored_shift_mm[anchor_fine] = (
        anchored_depth[anchor_fine] - measured_fine[anchor_fine]
    ) * 1000.0

    solver_vals = solver_shift_mm[anchor_fine]
    anchored_vals = anchored_shift_mm[anchor_fine]
    print(
        f"solver   shift at measured pixels: mean {solver_vals.mean():+.1f} mm, "
        f"mean|.| {np.abs(solver_vals).mean():.1f} mm, max|.| {np.abs(solver_vals).max():.1f} mm"
    )
    print(
        f"anchored shift at measured pixels: mean {anchored_vals.mean():+.1f} mm, "
        f"mean|.| {np.abs(anchored_vals).mean():.1f} mm, max|.| {np.abs(anchored_vals).max():.1f} mm"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"frame {args.frame} -- solver moves measured depth, anchored does not"
    )

    # Every image panel below is drawn with the same `extent`, so they all share
    # one physical coordinate system regardless of native array resolution (the
    # raw depth panel is 256x192 pixels but covers the same frame as the rest).
    # That makes sharing x/y limits across them meaningful: zoom/pan on one and
    # the others follow, pointing at the same patch of the scene. The histogram
    # is a different kind of plot (mm on x, pixel count on y) and is
    # deliberately left out of the sharing group.
    extent = [0, target_w, target_h, 0]
    image_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax in image_axes[1:]:
        ax.sharex(image_axes[0])
        ax.sharey(image_axes[0])

    # load_rgb() already returns RGB (it converts internally); re-converting
    # here would swap the channels right back to BGR.
    axes[0, 0].imshow(rgb, extent=extent)
    axes[0, 0].set_title("RGB")

    vmax = np.percentile(depth[depth > 0], 99) if (depth > 0).any() else 1.0
    axes[0, 1].imshow(depth, cmap="viridis", vmin=0, vmax=vmax, extent=extent)
    axes[0, 1].set_title(f"raw cleaned depth ({depth.shape[1]}x{depth.shape[0]})")

    axes[0, 2].imshow(anchored_depth, cmap="viridis", vmin=0, vmax=vmax, extent=extent)
    axes[0, 2].set_title(f"anchored upsample ({target_w}x{target_h})")

    shift_abs_max = max(np.abs(solver_vals).max(), np.abs(anchored_vals).max(), 1.0)
    im = axes[1, 0].imshow(
        solver_shift_mm,
        cmap="coolwarm",
        vmin=-shift_abs_max,
        vmax=shift_abs_max,
        extent=extent,
    )
    axes[1, 0].set_title("solver shift at measured px (mm)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(
        anchored_shift_mm,
        cmap="coolwarm",
        vmin=-shift_abs_max,
        vmax=shift_abs_max,
        extent=extent,
    )
    axes[1, 1].set_title("anchored shift at measured px (mm)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].hist(
        solver_vals,
        bins=80,
        alpha=0.6,
        label=f"solver (mean|.|={np.abs(solver_vals).mean():.0f}mm)",
    )
    axes[1, 2].hist(
        anchored_vals,
        bins=80,
        alpha=0.6,
        label=f"anchored (mean|.|={np.abs(anchored_vals).mean():.0f}mm)",
    )
    axes[1, 2].set_title("shift distribution at measured pixels")
    axes[1, 2].set_xlabel("mm")
    axes[1, 2].legend()

    for ax in image_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
