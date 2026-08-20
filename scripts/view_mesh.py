"""View one or more meshes, laid out side by side under a shared camera.

    python scripts/view_mesh.py output/myroom_v15mm.ply
    python scripts/view_mesh.py output/a.ply output/b.ply --colour
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

GREY = (0.74, 0.76, 0.78)

# Distinguishable when several models are shown together.
DISTINCT = [
    (0.90, 0.40, 0.15),  # orange
    (0.25, 0.55, 0.90),  # blue
    (0.35, 0.75, 0.40),  # green
    (0.85, 0.75, 0.20),  # yellow
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes", nargs="+", help="mesh files to display, left to right"
    )
    parser.add_argument(
        "--grey",
        action="store_true",
        help="flat grey shading instead of captured RGB, which shows geometry more clearly",
    )
    parser.add_argument(
        "--gap", type=float, default=0.15, help="gap as a fraction of width"
    )
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument(
        "--distinct",
        action="store_true",
        help="paint each input a different flat colour and print the legend, so "
        "there is no doubt which is which when comparing",
    )
    parser.add_argument(
        "--lit",
        action="store_true",
        help="shade with the default lit material. Off for colour, because the lit "
        "shader adds specular highlights that make every surface look glossy -- "
        "carpet included -- which is the renderer talking, not the capture",
    )
    args = parser.parse_args()

    geometries = []
    offset = 0.0
    for path_string in args.meshes:
        path = Path(path_string)
        if not path.is_file():
            print(f"skipping missing mesh: {path}")
            continue

        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            # MVS output is a point cloud, not a mesh. Reading it as a mesh
            # yields zero triangles, and skipping on that silently drops it from
            # the comparison -- leaving one model on screen and no explanation.
            cloud = o3d.io.read_point_cloud(str(path))
            if len(cloud.points) == 0:
                print(f"skipping empty file: {path}")
                continue
            print(f"{path.name}: {len(cloud.points):,} points (cloud)")
            if args.grey or not cloud.has_colors():
                cloud.paint_uniform_color(GREY)
            mesh = cloud
        else:
            print(
                f"{path.name}: {len(mesh.vertices):,} verts, "
                f"{len(mesh.triangles):,} tris, {mesh.get_surface_area():.1f} m2"
            )
            if args.grey or not mesh.has_vertex_colors():
                mesh.paint_uniform_color(GREY)
            mesh.compute_vertex_normals()

        box = mesh.get_axis_aligned_bounding_box()
        width = box.get_extent()[0]
        # Move each model so its left edge sits at the running offset, whatever
        # frame it arrived in.
        mesh.translate(
            (offset - box.min_bound[0], -box.min_bound[1], -box.min_bound[2])
        )
        if args.distinct or (args.grey and len(args.meshes) > 1):
            colour = DISTINCT[len(geometries) % len(DISTINCT)]
            mesh.paint_uniform_color(colour)
            names = {0: "orange", 1: "blue", 2: "green", 3: "yellow"}
            print(
                f"    -> shown in {names[len(geometries) % 4]}, "
                f"x from {offset:.1f} to {offset + width:.1f} m"
            )
        offset += width * (1.0 + args.gap)
        geometries.append(mesh)

    if not geometries:
        print("nothing to show")
        return 1

    print("\ndrag to orbit, scroll to zoom, close the window when done")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=" | ".join(Path(m).stem for m in args.meshes),
        width=1800,
        height=1000,
        mesh_show_back_face=True,
        mesh_show_wireframe=args.wireframe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
