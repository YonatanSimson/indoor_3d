"""Export a RawRecorder session to Replica-dataset format for external tools.

Both nvblox's `fuse_replica` and mrhash's DepthReader expect the same layout:

    <out>/cam_params.json          -- shared intrinsics + depth scale
    <out>/scene/traj.txt           -- one flattened 4x4 camera-to-world matrix per line
    <out>/scene/results/frameNNNNNN.jpg
    <out>/scene/results/depthNNNNNN.png   -- uint16, metres * scale

nvblox reads cam_params.json one level above the scene folder
(datasets/src/replica.cpp: `base_path + "/../cam_params.json"`); mrhash reads
intrinsics from its own config and only needs traj.txt + results/ inside the
scene folder. Point both tools at <out>/scene.

Depth is cleaned exactly as fusion does (confidence + flying-pixel filtering),
then upsampled to the native RGB resolution with the anchored solver -- real
LiDAR measurements are kept exact, gaps are filled RGB-guided.

Per the workspace convention, output goes in the data folder next to the
recording (<session>/replica/), not into the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.depth_upsampling import upsample


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument("--out", default=None, help="defaults to <session>/replica")
    parser.add_argument(
        "--refined-poses",
        nargs="?",
        const="auto",
        default="auto",
        help="use COLMAP-refined poses; bare flag or default looks for <session>/colmap/refined_poses.json",
    )
    parser.add_argument(
        "--no-refined-poses", action="store_true", help="use raw ARKit poses instead"
    )
    parser.add_argument(
        "--upsample",
        default="anchored",
        choices=["none", "bilinear", "joint", "solver", "anchored"],
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--depth-scale", type=float, default=1000.0, help="uint16 PNG units per metre"
    )
    args = parser.parse_args()

    session_path = Path(args.session)
    out_root = Path(args.out) if args.out else session_path / "replica"
    scene_dir = out_root / "scene"
    results_dir = scene_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

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
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    traj_lines: list[str] = []
    start = time.time()
    for i, frame in enumerate(sequence):
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

        depth_units = np.clip(np.round(upsampled * args.depth_scale), 0, 65535).astype(
            np.uint16
        )
        cv2.imwrite(str(results_dir / f"depth{i:06d}.png"), depth_units)
        cv2.imwrite(
            str(results_dir / f"frame{i:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        camera_to_world = np.linalg.inv(sequence.world_to_camera(frame, "rdf"))
        traj_lines.append(" ".join(f"{v:.12e}" for v in camera_to_world.flatten()))

        if (i + 1) % 50 == 0 or i + 1 == len(sequence):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(
                f"  {i + 1}/{len(sequence)} frames ({rate:.1f}/s, {elapsed:.0f}s elapsed)"
            )

    # newline="\n": write_text() otherwise translates to CRLF on Windows.
    # nvblox/mrhash's parsers tolerated the trailing \r; not all readers do
    # (see export_tum.py) -- fixed here too for consistency.
    (scene_dir / "traj.txt").write_text("\n".join(traj_lines) + "\n", newline="\n")

    cam_params = {
        "camera": {
            "w": rgb_width,
            "h": rgb_height,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "scale": args.depth_scale,
        }
    }
    (out_root / "cam_params.json").write_text(json.dumps(cam_params, indent=2))

    print(f"\nwrote {len(sequence)} frames to {scene_dir}")
    print(f"cam_params.json: {out_root / 'cam_params.json'}")
    print(f"nvblox:  fuse_replica {scene_dir}")
    print(f"mrhash:  point the RGB-D reader at {scene_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
