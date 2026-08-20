"""Phase 4, stages 3 and 6: judge MVS depth maps against LiDAR.

MVS never sees the LiDAR, so every LiDAR sample is a held-out test -- a stronger
position than any evaluation so far in this repo, where the depth being measured
against was usually the same depth that built the surface.

Three questions, and they must be read together:

- **accuracy**, where both have an opinion. MVS being close to LiDAR on
  well-textured surfaces is the basic sanity check.
- **completeness**, split by whether LiDAR had a confident reading there. MVS
  earns its place in the gaps: pixels where LiDAR was unconfident but MVS
  resolved something. Completeness where LiDAR already knew the answer adds
  nothing.
- **gross disagreement**, the rejection rate. MVS fails by producing confident
  wrong depth on blank walls and repeating texture, so the share of pixels that
  contradict LiDAR outright is the failure signal -- and a per-image rejection
  rate flags which views had too little baseline or too little texture.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth


def read_colmap_array(path: Path) -> np.ndarray:
    """Read COLMAP's binary depth/normal map format."""
    with path.open("rb") as handle:
        header = ""
        while len(header.split("&")) < 4:
            header += handle.read(1).decode()
        width, height, channels = (int(v) for v in header.split("&")[:3])
        data = np.fromfile(handle, dtype=np.float32)
    return data.reshape((height, width, channels) if channels > 1 else (height, width))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument(
        "--workspaces",
        nargs="+",
        required=True,
        help="dense workspace directories to compare",
    )
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument(
        "--kind", default="geometric", choices=["geometric", "photometric"]
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--reject-abs", type=float, default=0.05, help="metres")
    parser.add_argument(
        "--reject-rel", type=float, default=0.05, help="fraction of depth"
    )
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    by_name = {sequence[i].folder.name: sequence[i] for i in range(len(sequence))}
    labels = args.labels or [Path(w).name for w in args.workspaces]

    header = (
        f"{'run':>14} | {'maps':>5} | {'accuracy':>9} | {'reject':>7} | "
        f"{'cover(lidar)':>12} | {'cover(gap)':>10} | {'px/map':>8}"
    )
    print(header)
    print("-" * len(header))

    for workspace, label in zip(args.workspaces, labels):
        depth_dir = Path(workspace) / "stereo" / "depth_maps"
        maps = sorted(depth_dir.glob(f"*.{args.kind}.bin"))
        if not maps:
            print(f"{label:>14} | no {args.kind} depth maps found")
            continue

        errors, rejected, total = [], 0, 0
        covered_lidar, seen_lidar = 0, 0
        covered_gap, seen_gap = 0, 0
        pixels = []

        for path in maps:
            frame = by_name.get(path.name.split(".")[0])
            if frame is None:
                continue
            mvs = read_colmap_array(path)
            valid_mvs = mvs > 0
            pixels.append(int(valid_mvs.sum()))
            total += valid_mvs.size

            confidence = sequence.load_confidence(frame)
            lidar, _ = clean_depth(
                sequence.load_depth(frame),
                confidence,
                min_confidence=args.min_confidence,
            )
            h, w = mvs.shape
            lidar_up = cv2.resize(lidar, (w, h), interpolation=cv2.INTER_NEAREST)
            conf_up = (
                cv2.resize(confidence, (w, h), interpolation=cv2.INTER_NEAREST)
                if confidence is not None
                else np.full((h, w), 2, np.uint8)
            )

            trusted = (lidar_up > 0) & (conf_up >= args.min_confidence)
            gap = ~trusted

            both = trusted & valid_mvs
            if both.any():
                difference = np.abs(mvs[both] - lidar_up[both])
                errors.append(difference)
                threshold = np.maximum(
                    args.reject_abs, args.reject_rel * lidar_up[both]
                )
                rejected += int((difference > threshold).sum())

            # Completeness where LiDAR knew the answer versus where it did not.
            seen_lidar += int(trusted.sum())
            covered_lidar += int(both.sum())
            seen_gap += int(gap.sum())
            covered_gap += int((gap & valid_mvs).sum())

        if not errors:
            print(f"{label:>14} | {len(maps):5d} | no overlap with LiDAR")
            continue
        allerr = np.concatenate(errors)
        print(
            f"{label:>14} | {len(maps):5d} | {np.median(allerr)*1000:6.1f} mm | "
            f"{rejected/len(allerr):6.1%} | {covered_lidar/max(seen_lidar,1):11.1%} | "
            f"{covered_gap/max(seen_gap,1):9.1%} | {np.mean(pixels)/1000:7.0f}k"
        )

    print(
        "\naccuracy     = median |MVS - LiDAR| where both are valid (held out: MVS never sees LiDAR)"
    )
    print(
        f"reject       = share disagreeing by more than max({args.reject_abs} m, "
        f"{args.reject_rel:.0%}) -- MVS failure, since LiDAR does not fail on blank walls"
    )
    print("cover(lidar) = share of confident-LiDAR pixels MVS also resolved")
    print(
        "cover(gap)   = share of low/no-confidence pixels MVS resolved -- where MVS earns its place"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
