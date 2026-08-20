"""Decide empirically how ICL-NUIM's poses and intrinsics are stored.

Mirrors `check_convention.py`, which settled the same class of question for
ARKitScenes: guessing the camera convention wrong still produces a
plausible-looking point cloud (just mirrored about one or two axes), so the
failure is silent until compared against something known.

This dataset had a second, nastier trap that the axes/direction/fy-sign sweep
below caught by accident: `livingRoom2.gt.freiburg` -- the file that looks
like the ground-truth trajectory, and is what the hand-off doc named -- does
not back-project anywhere near `living-room.ply` under *any* hypothesis
tested here (best case ~27cm median, and RANSAC+ICP registration doesn't
rescue it either -- see `indoor3d/data_iclnuim.py`'s module docstring for the
full story). `ICLNUIMSequence` uses `livingRoom2n.gt.sim` instead, ICL-NUIM's
own native trajectory format, which is what this script actually sweeps.

Back-projects a sample of frames under every (axes, direction, fy sign)
hypothesis and measures how far the resulting points land from the
ground-truth point cloud (`living-room.ply` ships points and normals, no
faces). The correct combination should sit within a few centimetres of the
laser-accurate surface; wrong ones are off by a wide margin.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_iclnuim import ICLNUIMSequence


def back_project(depth: np.ndarray, intrinsics: np.ndarray, stride: int) -> np.ndarray:
    """Lift a depth image to 3D points in the RDF camera frame, shape (N, 3)."""
    depth = depth[::stride, ::stride]
    rows, cols = np.indices(depth.shape)
    u = cols * stride
    v = rows * stride

    valid = depth > 0
    z = depth[valid]
    u = u[valid]
    v = v[valid]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    # A negative fy hypothesis flips the sign of this axis relative to a
    # positive one -- exactly the trap this script exists to catch.
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def to_world(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ camera_to_world.T)[:, :3]


def distances_to_cloud(points: np.ndarray, tree: cKDTree) -> np.ndarray:
    distances, _ = tree.query(points, workers=-1)
    return distances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4, help="pixel subsampling")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.01,
        help="downsample the ~10M-point ground-truth cloud to this resolution "
        "before building the nearest-neighbour tree; only bounds one-time setup "
        "cost, not the distances reported",
    )
    args = parser.parse_args()

    sequence = ICLNUIMSequence(
        args.root
    )  # defaults are irrelevant; orientation is reset below
    print(f"sequence: {len(sequence)} posed frames (of 881 total)")

    # living-room.ply ships as points + normals, no faces (confirmed by
    # `open3d.io.read_triangle_mesh` reporting zero triangles on it) -- it is
    # ICL-NUIM's own dense point sample of the scene, not a mesh, so distance
    # queries use a nearest-neighbour tree rather than raycasting.
    print(
        f"reading ground-truth point cloud {sequence.ground_truth_cloud_path.name} ...",
        flush=True,
    )
    cloud = o3d.io.read_point_cloud(str(sequence.ground_truth_cloud_path))
    print(f"  {len(cloud.points):,} points, downsampling to {args.voxel * 1000:.0f} mm")
    cloud = cloud.voxel_down_sample(args.voxel)
    cloud_points = np.asarray(cloud.points)
    print(f"  {len(cloud_points):,} points after downsampling")
    print(
        f"  bounds min {cloud.get_min_bound().round(2)} max {cloud.get_max_bound().round(2)}"
    )
    tree = cKDTree(cloud_points)

    indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)
    print(f"\ntesting {len(indices)} frames, pixel stride {args.stride}")

    header = f"{'axes':>5} | {'direction':>9} | {'fy':>7} | {'median':>8} | {'mean':>8} | {'p90':>8} | {'<5cm':>7}"
    print(header)
    print("-" * len(header))

    results: dict[tuple[str, str, float], float] = {}
    for pose_axes, pose_direction, fy_sign in itertools.product(
        ("native", "rub_to_rdf"), ("c2w", "w2c"), (1.0, -1.0)
    ):
        trial = ICLNUIMSequence(
            args.root,
            fy_sign=fy_sign,
            pose_axes=pose_axes,
            pose_direction=pose_direction,
        )
        all_distances = []
        for index in indices:
            frame = trial[index]
            depth = trial.load_depth(frame)
            intrinsics, _, _ = trial.intrinsics(frame)
            points = back_project(depth, intrinsics, args.stride)
            if len(points) == 0:
                continue
            world = to_world(points, trial.camera_to_world(frame))
            all_distances.append(distances_to_cloud(world, tree))

        distances = np.concatenate(all_distances)
        key = (pose_axes, pose_direction, fy_sign)
        results[key] = float(np.median(distances))
        print(
            f"{pose_axes:>5} | {pose_direction:>9} | {fy_sign:>+7.0f} | "
            f"{results[key]:8.3f} | {distances.mean():8.3f} | "
            f"{np.percentile(distances, 90):8.3f} | {(distances < 0.05).mean():6.1%}"
        )

    best = min(results, key=results.get)  # type: ignore[arg-type]
    axes, direction, fy_sign = best
    print(
        f"\nverdict: pose_axes={axes!r}, pose_direction={direction!r}, fy_sign={fy_sign:+.0f} "
        f"-- median {results[best]:.3f} m to the ground-truth surface."
    )
    print(
        f"use ICLNUIMSequence(..., fy_sign={fy_sign}, pose_axes={axes!r}, "
        f"pose_direction={direction!r})"
    )
    if results[best] > 0.05:
        print(
            "\nWARNING: even the best hypothesis is >5cm off. That is not a sign "
            "convention any more -- re-check that living-room.ply is actually in "
            "the same world frame as livingRoom2n.gt.sim."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
