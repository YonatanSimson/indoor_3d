"""Measure a reconstruction against an ARKitScenes Faro laser scan.

This is the only absolute-accuracy test in the repo. Everywhere else "error"
means distance to the LiDAR that built the mesh, which cannot reveal a bias the
sensor itself has. The laser scan is independent of the iPhone capture, so it
can.

Two numbers, and they trade off against each other:

- accuracy     : reconstruction -> laser. How wrong is the geometry we produced?
- completeness : laser -> reconstruction. How much of the room did we miss?

A mesh can win on accuracy by reconstructing only the easy surfaces, so neither
number means much alone.

The scan arrives in the scanner's own frame, so it is registered to the mesh
first (coarse feature matching, then ICP). Registration quality is reported --
if fitness is low, treat every number below it as meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence


def load_laser(path: Path, voxel: float) -> o3d.geometry.PointCloud:
    """Read the laser scan and immediately downsample; these are ~40M points."""
    print(f"reading {path.name} ({path.stat().st_size / 1e6:.0f} MB)...", flush=True)
    cloud = o3d.io.read_point_cloud(str(path))
    print(f"  {len(cloud.points):,} points, downsampling to {voxel * 1000:.0f} mm")
    cloud = cloud.voxel_down_sample(voxel)
    print(f"  {len(cloud.points):,} points after downsampling")
    return cloud


def mesh_to_cloud(mesh: o3d.geometry.TriangleMesh, voxel: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    return cloud.voxel_down_sample(voxel)


def register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel: float,
) -> tuple[np.ndarray, float, float]:
    """Align source onto target, returning (transform, fitness, inlier RMSE)."""
    radius_normal = voxel * 2
    radius_feature = voxel * 5

    for cloud in (source, target):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    def features(cloud):
        return o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

    print("  global registration (RANSAC on FPFH)...", flush=True)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        features(source),
        features(target),
        mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.999),
    )
    print(f"    fitness {coarse.fitness:.3f}, RMSE {coarse.inlier_rmse * 1000:.1f} mm")

    print("  refining (point-to-plane ICP)...", flush=True)
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        voxel * 1.5,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(f"    fitness {fine.fitness:.3f}, RMSE {fine.inlier_rmse * 1000:.1f} mm")
    return fine.transformation, fine.fitness, fine.inlier_rmse


def distances_to_mesh(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(
        o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laser", default=r"C:\Users\user\data\arkitscenes_gt\421337\170078.ply")
    parser.add_argument("--root", default=r"C:\Users\user\work\ARKit3d\sample_data\Validation")
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument("--meshes", nargs="+", required=True)
    parser.add_argument("--voxel", type=float, default=0.02, help="registration/eval resolution")
    parser.add_argument(
        "--crop-around",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="keep only laser points near this site-frame point. A Faro scan sees "
        "through doors and windows, so it spans the whole building while the "
        "capture covers one room; cropping to the scanner's own neighbourhood "
        "is what makes registration tractable",
    )
    parser.add_argument("--crop-size", type=float, default=9.0, help="half-extent of the crop, metres")
    parser.add_argument(
        "--transform",
        default=None,
        help="use a saved laser->mesh transform from align_laser_manual.py instead "
        "of registering automatically",
    )
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="ignore laser points further than this from any mesh; the scan "
                             "covers areas the phone never visited")
    args = parser.parse_args()

    laser_path = Path(args.laser)
    if not laser_path.is_file():
        print(f"ERROR: no laser scan at {laser_path}", file=sys.stderr)
        return 1

    laser = load_laser(laser_path, args.voxel)

    if args.crop_around is not None:
        centre = np.asarray(args.crop_around, dtype=float)
        half = args.crop_size
        box = o3d.geometry.AxisAlignedBoundingBox(centre - half, centre + half)
        before = len(laser.points)
        laser = laser.crop(box)
        print(f"cropped to {args.crop_size * 2:.0f} m box at {np.round(centre, 1)}: "
              f"{before:,} -> {len(laser.points):,} points")

    sequence = Sequence(args.root, args.video_id)
    candidates = []
    if sequence.arkit_mesh_path:
        candidates.append(("ARKit (on-device)", sequence.arkit_mesh_path))
    candidates += [(Path(p).stem, Path(p)) for p in args.meshes]

    # Register once, against the first mesh, and reuse that transform so every
    # mesh is judged in the same frame -- registering per mesh would let each
    # one shift the reference to flatter itself.
    reference_mesh = o3d.io.read_triangle_mesh(str(candidates[0][1]))
    print(f"\nregistering laser scan to {candidates[0][0]}")
    transform, fitness, rmse = register(
        laser, mesh_to_cloud(reference_mesh, args.voxel), args.voxel
    )
    if fitness < 0.3:
        print(f"\nWARNING: registration fitness {fitness:.3f} is too low to trust. "
              "The scan may cover a different area, or need manual alignment.")

    laser.transform(transform)
    laser_points = np.asarray(laser.points)
    print(f"\nregistered laser cloud: {len(laser_points):,} points\n")

    header = (f"{'mesh':>22} | {'accuracy median':>15} | {'accuracy mean':>13} | "
              f"{'completeness <5cm':>17}")
    print(header)
    print("-" * len(header))

    for name, path in candidates:
        mesh = o3d.io.read_triangle_mesh(str(path))

        # Accuracy: mesh vertices -> laser surface.
        tree = o3d.geometry.KDTreeFlann(laser)
        vertices = np.asarray(mesh.vertices)
        sample = vertices[np.random.default_rng(0).choice(
            len(vertices), min(200_000, len(vertices)), replace=False)]
        accuracy = []
        for point in sample:
            _, _, squared = tree.search_knn_vector_3d(point, 1)
            accuracy.append(np.sqrt(squared[0]))
        accuracy = np.array(accuracy)
        accuracy = accuracy[accuracy < args.max_distance]

        # Completeness: laser points -> mesh surface.
        completeness = distances_to_mesh(laser_points, mesh)
        covered = completeness < args.max_distance

        print(f"{name:>22} | {np.median(accuracy) * 1000:12.1f} mm | "
              f"{accuracy.mean() * 1000:10.1f} mm | "
              f"{(completeness[covered] < 0.05).mean():16.1%}")

    print("\naccuracy     = reconstruction -> laser (smaller is better)")
    print("completeness = share of laser points with surface within 5 cm")
    print(f"points beyond {args.max_distance} m excluded: the scan covers rooms "
          "the phone never entered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
