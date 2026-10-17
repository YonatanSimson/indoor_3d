"""Score a reconstruction against ICL-NUIM's ground-truth point cloud.

Factored out of `scripts/eval_iclnuim_pose_ceiling.py` once a second script
(`scripts/eval_iclnuim_voxel_sweep.py`) needed the same "mesh vs
`living-room.ply`" scoring -- both accuracy (reconstruction -> truth) and
completeness (truth -> reconstruction), the same two numbers
`scripts/eval_vs_laser.py` reports for the ARKitScenes laser scan.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# 0.1mm resolution: an order of magnitude finer than any voxel size this
# project uses, so LAZ's fixed-point quantisation never becomes a source of
# error in its own right.
_LAZ_POINT_SCALE = 0.0001


def _write_laz_cache(path: Path, points: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [_LAZ_POINT_SCALE] * 3
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(str(path))


def _read_laz_cache(path: Path) -> np.ndarray:
    las = laspy.read(str(path))
    return np.column_stack([las.x, las.y, las.z]).astype(np.float64)


def load_ground_truth_tree(path: Path, voxel: float) -> tuple[cKDTree, np.ndarray]:
    """Ground truth is a ~10M-point cloud (no faces) -- see data_iclnuim.py.

    Reading and voxel-downsampling that cloud is the single heaviest,
    slowest step every one of these eval scripts runs, and it produces the
    *same* result every time for a given voxel size -- so the downsampled
    result is cached as compressed LAZ (via `laspy`, the standard point-cloud
    interchange format, rather than a bespoke `.npy` blob) next to the
    source file, in the data folder rather than the project tree per
    CLAUDE.md, so it's inspectable in CloudCompare/PDAL/any LiDAR tool too.
    """
    cache_path = path.parent / f".{path.stem}_cache_{voxel * 1000:.1f}mm.laz"
    if cache_path.is_file():
        print(f"loading cached ground-truth cloud {cache_path.name} ...", flush=True)
        points = _read_laz_cache(cache_path)
        print(f"  {len(points):,} points (cached, skipping the full read + downsample)")
        return cKDTree(points), points

    print(f"reading ground-truth cloud {path.name} ...", flush=True)
    cloud = o3d.io.read_point_cloud(str(path))
    print(f"  {len(cloud.points):,} points, downsampling to {voxel * 1000:.0f} mm")
    cloud = cloud.voxel_down_sample(voxel)
    points = np.asarray(cloud.points)
    print(f"  {len(points):,} points after downsampling")

    _write_laz_cache(cache_path, points)
    print(
        f"  cached to {cache_path.name} for next time ({cache_path.stat().st_size / 1e6:.1f} MB)"
    )
    return cKDTree(points), points


def mesh_accuracy(
    mesh: o3d.geometry.TriangleMesh, gt_tree: cKDTree, max_distance: float
) -> np.ndarray:
    """Reconstruction -> ground truth: how wrong is the geometry we produced?"""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return np.empty(0)
    distances, _ = gt_tree.query(vertices, workers=-1)
    return distances[distances < max_distance]


def mesh_completeness(
    gt_points: np.ndarray, mesh: o3d.geometry.TriangleMesh, max_distance: float
) -> np.ndarray:
    """Ground truth -> reconstruction: how much of the room did we capture?"""
    if len(mesh.triangles) == 0:
        return np.full(len(gt_points), np.inf)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(gt_points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(query).numpy()
    return distances[distances < max_distance]


def report(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    gt_tree: cKDTree,
    gt_points: np.ndarray,
    max_distance: float,
    print_row: bool = True,
) -> dict[str, float]:
    accuracy = mesh_accuracy(mesh, gt_tree, max_distance)
    completeness = mesh_completeness(gt_points, mesh, max_distance)
    stats = {
        "vertices": len(mesh.vertices),
        "area_m2": mesh.get_surface_area() if len(mesh.triangles) else 0.0,
        "accuracy_median_mm": (
            float(np.median(accuracy)) * 1000 if len(accuracy) else float("nan")
        ),
        "accuracy_mean_mm": (
            float(accuracy.mean()) * 1000 if len(accuracy) else float("nan")
        ),
        "complete_5cm": (
            float((completeness < 0.05).mean()) if len(completeness) else 0.0
        ),
    }
    if print_row:
        print(
            f"{name:>18} | {stats['vertices']:>10,} | {stats['area_m2']:>7.1f} | "
            f"{stats['accuracy_median_mm']:>10.1f} | {stats['accuracy_mean_mm']:>8.1f} | "
            f"{stats['complete_5cm']:>9.1%}"
        )
    return stats


REPORT_HEADER = (
    f"{'run':>18} | {'verts':>10} | {'area m2':>7} | {'acc median':>10} | "
    f"{'acc mean':>8} | {'<5cm cov':>9}"
)
