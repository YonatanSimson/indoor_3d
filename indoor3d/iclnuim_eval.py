"""Score a reconstruction against ICL-NUIM's ground-truth point cloud.

Factored out of `scripts/eval_iclnuim_pose_ceiling.py` once a second script
(`scripts/eval_iclnuim_voxel_sweep.py`) needed the same "mesh vs
`living-room.ply`" scoring -- both accuracy (reconstruction -> truth) and
completeness (truth -> reconstruction), the same two numbers
`scripts/eval_vs_laser.py` reports for the ARKitScenes laser scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_ground_truth_tree(path: Path, voxel: float) -> tuple[cKDTree, np.ndarray]:
    """Ground truth is a ~10M-point cloud (no faces) -- see data_iclnuim.py."""
    print(f"reading ground-truth cloud {path.name} ...", flush=True)
    cloud = o3d.io.read_point_cloud(str(path))
    print(f"  {len(cloud.points):,} points, downsampling to {voxel * 1000:.0f} mm")
    cloud = cloud.voxel_down_sample(voxel)
    points = np.asarray(cloud.points)
    print(f"  {len(points):,} points after downsampling")
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
