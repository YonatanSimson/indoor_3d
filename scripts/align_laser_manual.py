"""Align a laser scan to a reconstruction by picking matching points by hand.

Automatic global registration struggles here: a Faro scan sees through doors and
windows, so most of it is geometry the phone never captured, and the two clouds
share no coordinate frame. A person recognising the same room corner in both
solves in seconds what RANSAC cannot solve reliably at all.

Pick at least three clearly corresponding points -- room corners where two walls
meet the floor are ideal, door frames and window corners work too. Spread them
around the room rather than clustering them along one wall, or the fit is
poorly conditioned about the axis they share.

    python scripts/align_laser_manual.py

Controls in each window:
    shift + left click   pick a point (a sphere marks it)
    shift + right click  undo the last pick
    Q or Escape          done with this window

The laser scan opens first, then the mesh. Pick the same features in the same
order in both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

DEFAULT_LASER = r"C:\Users\user\data\arkitscenes_gt\421337\170078.ply"
DEFAULT_MESH = (
    r"C:\Users\user\work\ARKit3d\sample_data\Validation\42444946\42444946_3dod_mesh.ply"
)


def pick_points(geometry, title: str) -> list[int]:
    print(f"\n=== {title} ===")
    print("shift+click to pick, shift+right-click to undo, Q when done")
    visualiser = o3d.visualization.VisualizerWithEditing()
    visualiser.create_window(window_name=title, width=1500, height=950)
    visualiser.add_geometry(geometry)
    visualiser.run()
    visualiser.destroy_window()
    picked = visualiser.get_picked_points()
    print(f"picked {len(picked)} points")
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--laser", default=DEFAULT_LASER)
    parser.add_argument("--mesh", default=DEFAULT_MESH)
    parser.add_argument("--voxel", type=float, default=0.03)
    parser.add_argument(
        "--crop-around", nargs=3, type=float, default=[1.35, 0.51, 228.87]
    )
    parser.add_argument("--crop-size", type=float, default=12.0)
    parser.add_argument("--out", default="output/laser_to_mesh.json")
    args = parser.parse_args()

    print(f"reading {Path(args.laser).name}...", flush=True)
    laser = o3d.io.read_point_cloud(args.laser).voxel_down_sample(args.voxel)
    if args.crop_around:
        centre = np.asarray(args.crop_around, dtype=float)
        laser = laser.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                centre - args.crop_size, centre + args.crop_size
            )
        )
    print(f"  {len(laser.points):,} points after downsample and crop")

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh_cloud = o3d.geometry.PointCloud()
    mesh_cloud.points = mesh.vertices
    mesh_cloud.colors = mesh.vertex_colors
    mesh_cloud = mesh_cloud.voxel_down_sample(args.voxel)
    print(f"  reconstruction: {len(mesh_cloud.points):,} points")

    laser_ids = pick_points(laser, "1/2  LASER SCAN - pick room corners")
    mesh_ids = pick_points(
        mesh_cloud, "2/2  RECONSTRUCTION - pick the SAME corners, same order"
    )

    if len(laser_ids) < 3 or len(laser_ids) != len(mesh_ids):
        print(
            f"\nneed at least 3 points, and equal counts in both "
            f"(got {len(laser_ids)} and {len(mesh_ids)})",
            file=sys.stderr,
        )
        return 1

    correspondences = o3d.utility.Vector2iVector(
        np.array([[laser_ids[i], mesh_ids[i]] for i in range(len(laser_ids))])
    )
    transform = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=False
    ).compute_transformation(laser, mesh_cloud, correspondences)

    residual = np.linalg.norm(
        np.asarray(laser.points)[laser_ids] @ transform[:3, :3].T
        + transform[:3, 3]
        - np.asarray(mesh_cloud.points)[mesh_ids],
        axis=1,
    )
    print(
        f"\nmanual fit residual: median {np.median(residual) * 1000:.0f} mm, "
        f"max {residual.max() * 1000:.0f} mm"
    )
    if residual.max() > 0.5:
        print("  that is loose -- picks may not correspond; ICP may still recover it")

    print("refining with ICP...")
    laser.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2, max_nn=30)
    )
    mesh_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2, max_nn=30)
    )
    result = o3d.pipelines.registration.registration_icp(
        laser,
        mesh_cloud,
        args.voxel * 2,
        transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(
        f"  fitness {result.fitness:.3f}, inlier RMSE {result.inlier_rmse * 1000:.1f} mm"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "laser": args.laser,
                "mesh": args.mesh,
                "crop_around": list(args.crop_around) if args.crop_around else None,
                "crop_size": args.crop_size,
                "manual_residual_mm": float(np.median(residual) * 1000),
                "icp_fitness": float(result.fitness),
                "icp_rmse_mm": float(result.inlier_rmse * 1000),
                "laser_to_mesh": np.asarray(result.transformation).tolist(),
            },
            indent=1,
        )
    )
    print(f"\nwrote {out_path}")
    print(
        "then:  python scripts/eval_vs_laser.py --transform "
        f"{out_path} --meshes output/42444946_v10mm.ply"
    )

    laser.transform(result.transformation)
    laser.paint_uniform_color([0.9, 0.35, 0.15])
    mesh_cloud.paint_uniform_color([0.35, 0.55, 0.9])
    print(
        "\nshowing the result: orange = laser, blue = reconstruction. "
        "They should sit on top of each other."
    )
    o3d.visualization.draw_geometries(
        [laser, mesh_cloud], window_name="alignment check", width=1600, height=950
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
