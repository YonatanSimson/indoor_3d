"""Phase 3+: dense depth from Depth Anything 3, anchored to LiDAR.

The information imbalance is the whole motivation: LiDAR gives 49k depth samples
per frame while the camera gives 2.8M pixels, and until now all geometry came
from the smaller stream. A monocular network predicts depth from the image
itself, so it works at image resolution and can resolve detail the LiDAR
physically cannot.

What it does not have is metric truth. Networks hallucinate on windows, mirrors
and textureless walls, and predict each frame independently, so depth wanders
between neighbouring views -- which a TSDF turns into smeared surfaces. The
LiDAR is the opposite: sparse and low-resolution, but metric and temporally
stable.

So the two are combined the way the confidence map suggests: the network
supplies structure, and every confident LiDAR reading is held fixed
(see `indoor3d.anchoring`). DA3 is additionally given the COLMAP-refined poses,
so its multi-view reasoning is anchored in our metric frame rather than a
guessed one. It expects world-to-camera extrinsics, which is exactly what the
loader already produces.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.anchoring import anchor_to_measurements
from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
DEFAULT_POSES = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941\colmap\refined_poses.json"


def scale_intrinsics(intrinsics: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = intrinsics.copy()
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument(
        "--poses", default=DEFAULT_POSES, help="COLMAP refined poses, or none"
    )
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--stride", type=int, default=20, help="use every Nth frame")
    parser.add_argument(
        "--limit", type=int, default=16, help="how many frames in total"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="views per forward pass. DA3 attends across views, so memory grows "
        "with this; the known poses keep separate batches in a common frame",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 internal resolution; higher is sharper and slower",
    )
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--no-pose-conditioning", action="store_true")
    parser.add_argument(
        "--out", default="output/da3", help="directory for depth .npz files"
    )
    args = parser.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    sequence = RawRecordingSequence(args.session, intrinsics_space="rgb")
    if args.poses and args.poses.lower() != "none":
        used = sequence.load_refined_poses(args.poses)
        print(f"poses: {used}/{len(sequence)} refined")
    else:
        print("poses: ARKit only")

    indices = list(range(0, len(sequence), args.stride))[: args.limit]
    print(f"frames: {len(indices)} (stride {args.stride})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} onto {device}...", flush=True)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    # Pose conditioning needs a camera encoder, and only the multi-view
    # checkpoints carry one -- the metric variants are monocular and would fail
    # deep inside the forward pass with a bare "NoneType is not callable".
    inner = getattr(model, "model", model)
    pose_conditioning = not args.no_pose_conditioning
    if pose_conditioning and getattr(inner, "cam_enc", None) is None:
        print(
            f"note: {args.model} has no camera encoder, so it cannot be "
            "pose-conditioned; running monocular instead"
        )
        print(
            "      (use a multi-view checkpoint such as DA3-LARGE for pose conditioning)"
        )
        pose_conditioning = False

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict] = []
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start : start + args.batch_size]
        frames = [sequence[i] for i in chunk]

        images = [sequence.load_rgb(f) for f in frames]
        extrinsics = np.stack([sequence.world_to_camera(f) for f in frames]).astype(
            np.float32
        )
        intrinsics = np.stack([sequence.intrinsics(f)[0] for f in frames]).astype(
            np.float32
        )

        started = time.perf_counter()
        prediction = model.inference(
            images,
            extrinsics=extrinsics if pose_conditioning else None,
            intrinsics=intrinsics if pose_conditioning else None,
            process_res=args.process_res,
        )
        elapsed = time.perf_counter() - started

        depths = np.asarray(prediction.depth)
        confs = (
            np.asarray(prediction.conf)
            if getattr(prediction, "conf", None) is not None
            else None
        )
        print(
            f"  batch {start // args.batch_size}: {len(chunk)} views in {elapsed:.1f}s "
            f"-> depth {depths.shape}",
            flush=True,
        )

        for position, frame in zip(range(len(chunk)), frames):
            predicted = depths[position].astype(np.float32)
            height, width = predicted.shape

            # LiDAR is cleaned on its own grid first, then placed on DA3's grid
            # so the two can be compared pixel for pixel.
            confidence = sequence.load_confidence(frame)
            lidar, _ = clean_depth(
                sequence.load_depth(frame),
                confidence,
                min_confidence=args.min_confidence,
            )
            lidar_up = cv2.resize(
                lidar, (width, height), interpolation=cv2.INTER_NEAREST
            )
            conf_up = (
                cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST)
                if confidence is not None
                else None
            )

            # DA3METRIC claims metric depth, but indoor metric scale is rarely
            # reliable, so the affine fit stays on and its recovered scale is
            # reported: a value far from 1 means the claim did not hold here.
            anchored, report = anchor_to_measurements(
                predicted, lidar_up, conf_up, args.min_confidence, fit_scale=True
            )

            overlap = (lidar_up > 0) & (predicted > 0)
            raw_gap = (
                float(np.median(np.abs(predicted[overlap] - lidar_up[overlap])) * 1000)
                if overlap.any()
                else float("nan")
            )

            base_intrinsics, base_w, base_h = sequence.intrinsics(frame)
            np.savez_compressed(
                out_dir / f"{frame.folder.name}.npz",
                depth=anchored,
                predicted=predicted,
                confidence=(
                    confs[position] if confs is not None else np.zeros_like(predicted)
                ),
                intrinsics=scale_intrinsics(
                    base_intrinsics, width / base_w, height / base_h
                ),
                world_to_camera=sequence.world_to_camera(frame),
            )
            stats.append({"frame": frame.folder.name, "raw_gap_mm": raw_gap, **report})

    print(
        f"\n{'frame':>14} | {'raw gap':>9} | {'scale':>7} | {'offset':>8} | {'anchors':>8}"
    )
    print("-" * 58)
    for s in stats[:12]:
        print(
            f"{s['frame']:>14} | {s['raw_gap_mm']:6.1f} mm | {s['scale']:7.4f} | "
            f"{s['offset']:8.4f} | {s['anchor_fraction']:7.1%}"
        )

    scales = np.array([s["scale"] for s in stats])
    gaps = np.array([s["raw_gap_mm"] for s in stats])
    print(f"\nmedian raw disagreement with LiDAR: {np.nanmedian(gaps):.1f} mm")
    print(
        f"affine scale: median {np.median(scales):.4f}, "
        f"spread {scales.min():.3f}-{scales.max():.3f}"
    )
    print(
        "(scale far from 1.0 means the metric output needed rescaling to match LiDAR;"
    )
    print(" a large spread across frames means it is not even self-consistent)")
    print(f"\nwrote {len(stats)} depth maps to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
