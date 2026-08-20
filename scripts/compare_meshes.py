"""Compare a fused mesh against ARKit's on-device mesh.

"Visibly sharper" is the goal, but eyeballing two meshes is a weak test, so this
measures two things that stand in for it:

- resolution: how finely the surface is tessellated (mean triangle edge);
- fidelity: how far the raw LiDAR measurements land from the reconstructed
  surface. A mesh that explains the sensor data more closely is the better
  reconstruction of what was actually scanned.

IMPORTANT -- what this is not: there is no ground truth here. The reference is
the LiDAR depth itself, so "error" means disagreement with the sensor, not with
reality, and the sensor's own noise sets the floor.

Worse, by default the comparison is circular for our meshes: `--num-frames`
samples frames that fusion already integrated, so a mesh is partly being scored
on data it was built from. That bias favours finer voxels, which can fit sensor
noise more closely, and it does not apply to ARKit's mesh, which was built by a
separate pipeline. Treat a fused-vs-ARKit gap as indicative, not measured.

Pass `--holdout` to fuse and evaluate on disjoint frames, which removes the
memorisation advantage and makes the comparison fair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence
from indoor3d.depth_filters import clean_depth


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, float]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return {"vertices": 0, "triangles": 0, "mean_edge_mm": float("nan")}

    corners = vertices[triangles]
    edges = np.concatenate(
        [
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
        ]
    )
    return {
        "vertices": len(vertices),
        "triangles": len(triangles),
        "mean_edge_mm": float(np.linalg.norm(edges, axis=1).mean() * 1000),
        "area_m2": float(mesh.get_surface_area()),
    }


def sample_measurements(
    sequence: Sequence, indices: np.ndarray, stride: int, traj_convention: str
) -> np.ndarray:
    """Back-project cleaned depth from the given frames into world points."""
    clouds = []
    for index in indices:
        frame = sequence[index]
        depth, _ = clean_depth(
            depth=sequence.load_depth(frame), confidence=sequence.load_confidence(frame)
        )
        intrinsics, _, _ = sequence.intrinsics(frame)

        sub = depth[::stride, ::stride]
        rows, cols = np.indices(sub.shape)
        valid = sub > 0
        if not valid.any():
            continue

        z = sub[valid]
        u = (cols * stride)[valid]
        v = (rows * stride)[valid]
        x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]

        points = np.stack([x, y, z], axis=1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        camera_to_world = sequence.camera_to_world(frame, traj_convention)
        clouds.append((homogeneous @ camera_to_world.T)[:, :3])

    return np.concatenate(clouds) if clouds else np.empty((0, 3))


def distances(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(query).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=r"C:\Users\user\work\ARKit3d\sample_data\Validation"
    )
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument(
        "--meshes", nargs="+", required=True, help="fused mesh .ply paths"
    )
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="evaluate only on frames a fusion run with this stride skipped "
        "(e.g. --holdout 2 tests a mesh fused with --stride 2 --offset 0). "
        "Without it the comparison is circular for our own meshes",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--traj-convention", default="rdf")
    args = parser.parse_args()

    sequence = Sequence(args.root, args.video_id)
    baseline_path = sequence.arkit_mesh_path
    if baseline_path is None:
        print("ERROR: no ARKit baseline mesh to compare against", file=sys.stderr)
        return 1

    if args.holdout > 1:
        # Frames the fused mesh never saw, so it cannot score well by having
        # memorised them. ARKit's mesh saw everything, which handicaps us
        # slightly -- the conservative direction for this comparison.
        available = np.array([i for i in range(len(sequence)) if i % args.holdout != 0])
        if len(available) == 0:
            raise SystemExit(f"--holdout {args.holdout} leaves no evaluation frames")
        picks = np.linspace(
            0, len(available) - 1, min(args.num_frames, len(available)), dtype=int
        )
        indices = available[picks]
        print(
            f"holdout: evaluating on {len(indices)} frames where index % {args.holdout} != 0"
        )
    else:
        indices = np.linspace(0, len(sequence) - 1, args.num_frames, dtype=int)
        print("WARNING: evaluation frames overlap the fused frames; see --holdout")
    measurements = sample_measurements(
        sequence, indices, args.stride, args.traj_convention
    )
    print(f"sampled {len(measurements):,} LiDAR points from {len(indices)} frames\n")

    candidates = [("ARKit (on-device)", baseline_path)] + [
        (Path(p).stem, Path(p)) for p in args.meshes
    ]

    header = (
        f"{'mesh':>22} | {'verts':>9} | {'tris':>10} | {'edge mm':>8} | "
        f"{'median mm':>9} | {'mean mm':>8} | {'<2cm':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, path in candidates:
        mesh = o3d.io.read_triangle_mesh(str(path))
        stats = mesh_stats(mesh)
        error = distances(measurements, mesh)
        print(
            f"{name:>22} | {stats['vertices']:>9,} | {stats['triangles']:>10,} | "
            f"{stats['mean_edge_mm']:>8.1f} | {np.median(error) * 1000:>9.1f} | "
            f"{error.mean() * 1000:>8.1f} | {(error < 0.02).mean():>5.1%}"
        )

    print(
        "\nedge mm  = mean triangle edge length (smaller = finer tessellation)\n"
        "median/mean = distance from raw LiDAR points to the surface (smaller = "
        "explains the measurements better)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
