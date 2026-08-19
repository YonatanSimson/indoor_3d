"""How well does a predicted depth field agree with LiDAR, and in what way?

Three levels of agreement, which fail for different reasons and need different
fixes:

- **raw**: absolute difference. Fails if the model is not metric.
- **after a per-frame affine fit**: what is left once scale and offset are
  removed. If this is small the shape is right and only the scale was wrong,
  which anchoring fixes completely. If it stays large the geometry itself is
  wrong, and no amount of rescaling will save it.
- **correlation**: whether the depth ordering agrees at all, independent of
  any affine relationship.

The scale spread across frames matters as much as its average. A model that is
consistently 1.2x off can be corrected once; one that varies between 0.9x and
1.7x from frame to frame will place the same wall in different spots in a fused
reconstruction, and must be anchored per frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.anchoring import fit_affine
from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth


def evaluate_directory(
    directory: Path, sequence: RawRecordingSequence, min_confidence: int
) -> dict[str, float] | None:
    by_name = {sequence[i].folder.name: sequence[i] for i in range(len(sequence))}

    raw, after, relative, correlation, scales = [], [], [], [], []
    for path in sorted(directory.glob("*.npz")):
        frame = by_name.get(path.stem)
        if frame is None:
            continue

        predicted = np.load(path)["predicted"].astype(np.float32)
        height, width = predicted.shape

        confidence = sequence.load_confidence(frame)
        lidar, _ = clean_depth(
            sequence.load_depth(frame), confidence, min_confidence=min_confidence
        )
        lidar = cv2.resize(lidar, (width, height), interpolation=cv2.INTER_NEAREST)

        mask = (lidar > 0) & (predicted > 0)
        if confidence is not None:
            mask &= (
                cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST)
                >= min_confidence
            )
        if mask.sum() < 500:
            continue

        raw.append(np.median(np.abs(predicted[mask] - lidar[mask])) * 1000)

        scale, offset = fit_affine(predicted, lidar, mask)
        residual = np.abs(scale * predicted[mask] + offset - lidar[mask])
        after.append(np.median(residual) * 1000)
        relative.append(after[-1] / (np.median(lidar[mask]) * 1000) * 100)
        correlation.append(float(np.corrcoef(predicted[mask], lidar[mask])[0, 1]))
        scales.append(scale)

    if not raw:
        return None
    return {
        "frames": len(raw),
        "raw_mm": float(np.median(raw)),
        "after_mm": float(np.median(after)),
        "relative_pct": float(np.median(relative)),
        "correlation": float(np.median(correlation)),
        "scale_median": float(np.median(scales)),
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941")
    parser.add_argument("--dirs", nargs="+", default=["output/da3", "output/da3_large"])
    parser.add_argument("--min-confidence", type=int, default=1)
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)

    header = (f"{'prediction':>18} | {'n':>3} | {'raw':>8} | {'after affine':>13} | "
              f"{'rel':>6} | {'corr':>6} | {'scale range':>14}")
    print(header)
    print("-" * len(header))

    for directory in args.dirs:
        result = evaluate_directory(Path(directory), sequence, args.min_confidence)
        if result is None:
            print(f"{Path(directory).name:>18} | no usable frames")
            continue
        print(
            f"{Path(directory).name:>18} | {result['frames']:>3} | {result['raw_mm']:6.1f}mm | "
            f"{result['after_mm']:10.1f}mm | {result['relative_pct']:5.1f}% | "
            f"{result['correlation']:6.3f} | "
            f"{result['scale_min']:.2f}-{result['scale_max']:.2f}".rjust(0)
        )

    print("\nraw          = absolute disagreement with LiDAR")
    print("after affine = what remains once per-frame scale and offset are removed")
    print("               (small = shape is right, anchoring will fix the rest)")
    print("scale range  = across frames; a wide range means per-frame anchoring is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
