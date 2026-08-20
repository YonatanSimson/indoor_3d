"""Export a RawRecorder session to TUM RGB-D format, for supereight2.

Verified against supereight2's actual reader (app/src/reader_tum.cpp), not
generic TUM docs -- the layout it enforces is stricter than the format
description suggests:

    <out>/depth.txt         "timestamp depth/frameNNNNNN.png"
    <out>/rgb.txt           "timestamp rgb/frameNNNNNN.png"
    <out>/groundtruth.txt   "timestamp tx ty tz qx qy qz qw"  (camera-to-world, RDF)
    <out>/depth/frameNNNNNN.png   uint16, metres * 5000 (supereight2's TUMReader default)
    <out>/rgb/frameNNNNNN.png

All three files use identical synthetic per-frame timestamps (frame index),
so supereight2's nearest-timestamp association/interpolation always lands on
an exact match -- no approximation error from that step.

Same anchored-upsampled depth and COLMAP-refined poses as export_replica.py;
only the on-disk format differs. Output goes in the data folder, not the repo
(see working agreements in claude.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample

DEPTH_SCALE = 5000.0  # supereight2 TUMReader's tum_inverse_scale_ default is 1/5000


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument("--out", default=None, help="defaults to <session>/tum")
    parser.add_argument(
        "--refined-poses",
        nargs="?",
        const="auto",
        default="auto",
        help="use COLMAP-refined poses; bare flag or default looks for <session>/colmap/refined_poses.json",
    )
    parser.add_argument("--no-refined-poses", action="store_true")
    parser.add_argument(
        "--upsample",
        default="anchored",
        choices=["none", "bilinear", "joint", "solver", "anchored"],
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    args = parser.parse_args()

    session_path = Path(args.session)
    out_root = Path(args.out) if args.out else session_path / "tum"
    (out_root / "depth").mkdir(parents=True, exist_ok=True)
    (out_root / "rgb").mkdir(parents=True, exist_ok=True)

    sequence = RawRecordingSequence(session_path, intrinsics_space="rgb")
    print(f"session: {session_path.name}, {len(sequence)} frames")

    if not args.no_refined_poses and args.refined_poses:
        poses_path = (
            session_path / "colmap" / "refined_poses.json"
            if args.refined_poses == "auto"
            else Path(args.refined_poses)
        )
        if poses_path.is_file():
            matched = sequence.load_refined_poses(poses_path)
            print(
                f"using refined poses for {matched}/{len(sequence)} frames ({poses_path.name})"
            )
        else:
            print(f"no refined poses at {poses_path}; using raw ARKit poses")

    intrinsics, rgb_width, rgb_height = sequence.intrinsics(sequence[0])
    print(
        f"sensor: {rgb_width}x{rgb_height}, fx={intrinsics[0, 0]:.2f}, fy={intrinsics[1, 1]:.2f}, "
        f"cx={intrinsics[0, 2]:.2f}, cy={intrinsics[1, 2]:.2f}"
    )

    depth_lines: list[str] = []
    rgb_lines: list[str] = []
    gt_lines: list[str] = []
    start = time.time()
    for i, frame in enumerate(sequence):
        timestamp = f"{i:.6f}"

        raw_depth = sequence.load_depth(frame)
        confidence = sequence.load_confidence(frame)
        rgb = sequence.load_rgb(frame)  # already RGB order

        depth, _ = clean_depth(
            raw_depth,
            confidence,
            min_confidence=args.min_confidence,
            max_depth=args.max_depth,
        )
        upsampled = upsample(
            depth, rgb, confidence, rgb_width, rgb_height, method=args.upsample
        )

        depth_name = f"depth/frame{i:06d}.png"
        rgb_name = f"rgb/frame{i:06d}.png"
        depth_units = np.clip(np.round(upsampled * DEPTH_SCALE), 0, 65535).astype(
            np.uint16
        )
        cv2.imwrite(str(out_root / depth_name), depth_units)
        cv2.imwrite(str(out_root / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        depth_lines.append(f"{timestamp} {depth_name}")
        rgb_lines.append(f"{timestamp} {rgb_name}")

        camera_to_world = np.linalg.inv(sequence.world_to_camera(frame, "rdf"))
        t = camera_to_world[:3, 3]
        q = Rotation.from_matrix(camera_to_world[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        gt_lines.append(
            f"{timestamp} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"
        )

        if (i + 1) % 50 == 0 or i + 1 == len(sequence):
            elapsed = time.time() - start
            print(
                f"  {i + 1}/{len(sequence)} frames ({(i + 1) / elapsed:.1f}/s, {elapsed:.0f}s elapsed)"
            )

    # newline="\n": write_text() otherwise translates to CRLF on Windows, and
    # supereight2's TUM reader parses each line's last column with an exact
    # std::stof length check -- a trailing \r fails it outright.
    (out_root / "depth.txt").write_text("\n".join(depth_lines) + "\n", newline="\n")
    (out_root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", newline="\n")
    (out_root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", newline="\n")

    print(f"\nwrote {len(sequence)} frames to {out_root}")
    print(f"supereight2 sequence_path: {out_root}")
    print(f"supereight2 ground_truth_file: {out_root / 'groundtruth.txt'}")
    print(
        f"sensor config: width={rgb_width} height={rgb_height} "
        f"fx={intrinsics[0, 0]:.4f} fy={intrinsics[1, 1]:.4f} "
        f"cx={intrinsics[0, 2]:.4f} cy={intrinsics[1, 2]:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
