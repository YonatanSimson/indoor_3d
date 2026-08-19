"""Measure whether guided upsampling puts depth edges where the image says.

Vertex counts and surface area cannot judge this: hole-filling inflates both
without adding a single correct edge. The actual claim guided upsampling makes
is that a depth discontinuity should coincide with a visible object boundary,
and that is measurable directly.

Three numbers per method, which have to be read together:

- alignment  : distance from each depth edge to the nearest image edge. Lower is
               better, but a method that outputs no edges at all scores
               perfectly, so it is meaningless without the next one.
- edge sharpness : depth gradient at image-edge pixels. Higher means the
               transition is crisp rather than smeared across many pixels.
- flat-region noise : depth gradient away from any image edge. Lower is better;
               this is what catches a method that buys sharp edges by making
               everything noisy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample


def gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def image_edges(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    guide = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(grey, 60, 160) > 0


def evaluate_frame(depth, rgb, confidence, width, height, method, edge_quantile):
    up = upsample(depth, rgb, confidence, width, height, method)
    valid = up > 0
    if valid.sum() < 100:
        return None

    grad = gradient_magnitude(np.where(valid, up, 0.0))
    grad[~valid] = 0.0

    edges = image_edges(rgb, width, height)
    # Depth edges are taken as the strongest gradients so the comparison does
    # not depend on an absolute threshold that suits one method over another.
    strong = grad > np.quantile(grad[valid], edge_quantile)

    # Distance to the nearest image edge, in pixels.
    distance = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)

    near_edge = edges & valid
    away = valid & (distance > 8)

    return {
        "alignment_px": float(np.median(distance[strong & valid])) if (strong & valid).any() else np.nan,
        "sharpness": float(grad[near_edge].mean()) if near_edge.any() else np.nan,
        "flat_noise": float(grad[away].mean()) if away.any() else np.nan,
        "filled": float(valid.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941")
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--edge-quantile", type=float, default=0.95)
    parser.add_argument("--methods", nargs="+", default=["bilinear", "joint", "solver"])
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)
    print(f"{len(indices)} frames, upsampling x{args.scale:g}\n")

    results: dict[str, list[dict]] = {m: [] for m in args.methods}
    for index in indices:
        frame = sequence[index]
        rgb = sequence.load_rgb(frame)
        confidence = sequence.load_confidence(frame)
        depth, _ = clean_depth(
            sequence.load_depth(frame), confidence, min_confidence=args.min_confidence
        )
        width = int(depth.shape[1] * args.scale)
        height = int(depth.shape[0] * args.scale)
        for method in args.methods:
            r = evaluate_frame(depth, rgb, confidence, width, height, method, args.edge_quantile)
            if r:
                results[method].append(r)

    header = (f"{'method':>10} | {'alignment px':>13} | {'edge sharpness':>15} | "
              f"{'flat noise':>11} | {'filled':>7}")
    print(header)
    print("-" * len(header))
    for method in args.methods:
        rows = results[method]
        if not rows:
            print(f"{method:>10} | no data")
            continue
        align = np.nanmean([r["alignment_px"] for r in rows])
        sharp = np.nanmean([r["sharpness"] for r in rows])
        noise = np.nanmean([r["flat_noise"] for r in rows])
        filled = np.nanmean([r["filled"] for r in rows])
        print(f"{method:>10} | {align:13.2f} | {sharp:15.4f} | {noise:11.4f} | {filled:6.1%}")

    print("\nalignment px   = depth-edge to nearest image-edge (lower better)")
    print("edge sharpness = depth gradient at image edges (higher better)")
    print("flat noise     = depth gradient away from edges (lower better)")
    print("\nsharpness/noise is the ratio that matters: a method can raise both,")
    print("which buys crisper edges at the cost of a noisier surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
