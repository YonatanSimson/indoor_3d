"""Decide empirically which camera convention the capture trajectory uses.

Back-projecting depth needs to know whether a trajectory pose maps points from
an RDF camera frame (OpenCV: X right, Y down, Z forward) or an RUB one
(ARKit/OpenGL: X right, Y up, Z back). Guessing wrong still produces a
plausible-looking point cloud, just one that is mirrored about two axes -- so
the failure is silent until the geometry is compared against something known.

This compares both hypotheses against the ARKit reference mesh: back-project a
sample of frames under each and measure how far the resulting points land from
that mesh's surface. The correct convention sits within a few centimetres; the
wrong one is off by metres.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence


def back_project(depth: np.ndarray, intrinsics: np.ndarray, stride: int) -> np.ndarray:
    """Lift a depth image to 3D points in the RDF camera frame, shape (N, 3)."""
    depth = depth[::stride, ::stride]
    rows, cols = np.indices(depth.shape)

    # Undo the stride so pixel coordinates still refer to the full-size image.
    u = cols * stride
    v = rows * stride

    valid = depth > 0
    z = depth[valid]
    u = u[valid]
    v = v[valid]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def to_world(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ camera_to_world.T)[:, :3]


def distances_to_mesh(
    points: np.ndarray, mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(query).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=r"C:\Users\user\work\ARKit3d\sample_data\Validation",
        help="directory holding <video_id>/ capture folders",
    )
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=4, help="pixel subsampling")
    args = parser.parse_args()

    sequence = Sequence(args.root, args.video_id)
    print(f"sequence {args.video_id}: {len(sequence)} posed frames")
    print(f"confidence maps present: {sequence.has_confidence}")

    mesh_path = sequence.arkit_mesh_path
    if mesh_path is None:
        print(
            "ERROR: no ARKit reference mesh, cannot decide the convention",
            file=sys.stderr,
        )
        return 1

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    print(
        f"reference mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    print(
        f"mesh bounds min {mesh.get_min_bound().round(2)} max {mesh.get_max_bound().round(2)}"
    )

    # Spread the sample across the whole session rather than clustering at the
    # start, so a single badly-tracked segment cannot dominate the verdict.
    indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)

    print(f"\ntesting {len(indices)} frames, pixel stride {args.stride}")
    print(
        f"{'convention':>12} | {'median':>8} | {'mean':>8} | {'p90':>8} | {'<5cm':>7}"
    )
    print("-" * 56)

    results: dict[str, float] = {}
    for traj_convention in ("rdf", "rub"):
        all_distances = []
        for index in indices:
            frame = sequence[index]
            depth = sequence.load_depth(frame)
            intrinsics, _, _ = sequence.intrinsics(frame)
            points = back_project(depth, intrinsics, args.stride)
            if len(points) == 0:
                continue
            world = to_world(points, sequence.camera_to_world(frame, traj_convention))
            all_distances.append(distances_to_mesh(world, mesh))

        distances = np.concatenate(all_distances)
        median = float(np.median(distances))
        results[traj_convention] = median
        print(
            f"{traj_convention:>12} | {median:8.3f} | {distances.mean():8.3f} | "
            f"{np.percentile(distances, 90):8.3f} | {(distances < 0.05).mean():6.1%}"
        )

    best = min(results, key=results.get)
    other = "rub" if best == "rdf" else "rdf"
    frame_name = {
        "rdf": "Right-Down-Forward (OpenCV)",
        "rub": "Right-Up-Back (ARKit/OpenGL)",
    }
    print(
        f"\nverdict: the trajectory is stored in {frame_name[best]}"
        f" -- median {results[best]:.3f} m to the reference mesh,"
        f" versus {results[other]:.3f} m assuming {frame_name[other]}."
    )
    print(f"use Sequence(...).world_to_camera(frame, traj_convention={best!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
