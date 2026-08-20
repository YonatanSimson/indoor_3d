"""Reconstruction accuracy vs true ground truth, across voxel sizes.

Stage B question 1 from `hand_off_original_plan.md`: does 5 mm actually
resolve noise, now that real geometry is available to check against instead
of the raw-LiDAR-distance metric that misled this project twice (ranked
15mm/10mm/5mm the *opposite* way to visual quality both times, because it
scores a mesh against the same noisy sensor that built it)?

Fuses the same simulated-sensor depth at 15mm/10mm/5mm with ground-truth
poses (the pose-ceiling experiment already established GT poses as the clean
baseline), scores each against `living-room.ply`, and prints both this
script's own accuracy figures *and* the old raw-LiDAR proxy metric side by
side -- so the two can be compared directly instead of trusting either one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.fusion import FusionConfig, fuse, save_mesh
from indoor3d.iclnuim_eval import REPORT_HEADER, load_ground_truth_tree, report
from indoor3d.sensor_sim import DegradedSequence


def raw_lidar_error(
    sequence: DegradedSequence,
    mesh: o3d.geometry.TriangleMesh,
    num_frames: int,
    stride: int,
) -> tuple[float, float]:
    """The old proxy metric: distance from the simulated depth to the mesh.

    Mirrors `scripts/compare_meshes.py`'s `sample_measurements`, scored
    against *this* mesh's own input data -- deliberately circular, which is
    the point: this is the metric the hand-off doc showed ranking meshes
    backwards, reproduced here so it can be compared against real ground
    truth on the same runs.
    """
    if len(mesh.triangles) == 0:
        return float("nan"), float("nan")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    clouds = []
    indices = np.linspace(0, len(sequence) - 1, num_frames, dtype=int)
    for index in indices:
        frame = sequence[index]
        depth, _ = clean_depth(
            sequence.load_depth(frame), sequence.load_confidence(frame)
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

        # Points are in camera space from the back-projection above, so
        # lifting them to world needs camera_to_world; `DegradedSequence`
        # (like `PosedSequence` generally) only exposes world_to_camera, so
        # invert it here rather than assuming every sequence has the inverse.
        camera_to_world = np.linalg.inv(sequence.world_to_camera(frame))
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        world = (homogeneous @ camera_to_world.T)[:, :3]
        clouds.append(world)

    if not clouds:
        return float("nan"), float("nan")
    query = o3d.core.Tensor(
        np.concatenate(clouds).astype(np.float32), dtype=o3d.core.Dtype.Float32
    )
    distances = scene.compute_distance(query).numpy()
    return float(np.median(distances)), float(distances.mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument("--out-dir", default=r"C:\Users\user\data\iclnuim_output")
    parser.add_argument(
        "--voxels", type=float, nargs="+", default=[0.015, 0.010, 0.005]
    )
    parser.add_argument("--gt-voxel", type=float, default=0.005)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sequence = ICLNUIMSequence(args.root)
    print(f"sequence: {len(sequence)} posed frames, ground-truth poses")

    print(f"\nsimulating our sensor over {len(sequence)} frames...")
    working_sequence = DegradedSequence(sequence, seed=args.seed)

    gt_tree, gt_points = load_ground_truth_tree(
        sequence.ground_truth_cloud_path, args.gt_voxel
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nscoring against ground truth (correspondences beyond {args.max_distance}m excluded)"
    )
    print(REPORT_HEADER + " |  old-metric median")
    print("-" * (len(REPORT_HEADER) + 21))

    results: dict[float, dict[str, float]] = {}
    for voxel in sorted(args.voxels, reverse=True):
        config = FusionConfig(voxel_length=voxel)
        mesh, _ = fuse(working_sequence, config, progress_every=0)
        save_mesh(mesh, out_dir / f"voxel_sweep_v{int(voxel * 1000)}mm.ply")

        stats = report(
            f"{voxel * 1000:.0f}mm",
            mesh,
            gt_tree,
            gt_points,
            args.max_distance,
            print_row=False,
        )
        old_median, _ = raw_lidar_error(working_sequence, mesh, num_frames=40, stride=4)
        stats["old_metric_median_mm"] = old_median * 1000
        results[voxel] = stats

        print(
            f"{voxel * 1000:>17.0f}mm | {stats['vertices']:>10,} | {stats['area_m2']:>7.1f} | "
            f"{stats['accuracy_median_mm']:>10.1f} | {stats['accuracy_mean_mm']:>8.1f} | "
            f"{stats['complete_5cm']:>9.1%} | {old_median * 1000:>18.1f}"
        )

    print(
        "\nacc median/mean = reconstruction -> ground truth (mm, this script's own metric)"
    )
    print(
        "old-metric      = distance from this run's own simulated LiDAR to its mesh (the"
    )
    print(
        "                  proxy metric shown misleading twice in hand_off_original_plan.md)"
    )

    by_truth = sorted(results, key=lambda v: results[v]["accuracy_median_mm"])
    by_old = sorted(results, key=lambda v: results[v]["old_metric_median_mm"])
    print(f"\nranked by ground truth : {[f'{v*1000:.0f}mm' for v in by_truth]}")
    print(f"ranked by old metric    : {[f'{v*1000:.0f}mm' for v in by_old]}")
    if by_truth == by_old:
        print(
            "agree: the old proxy metric ranks these the same way ground truth does here."
        )
    else:
        print(
            "DISAGREE: the old proxy metric's ranking does not match ground truth -- same "
            "failure mode the hand-off doc documented against raw LiDAR twice already."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
