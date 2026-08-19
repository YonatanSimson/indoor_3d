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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", help="mesh files to display, left to right")
    parser.add_argument(
        "--grey",
        action="store_true",
        help="flat grey shading instead of captured RGB, which shows geometry more clearly",
    )
    parser.add_argument("--gap", type=float, default=0.15, help="gap as a fraction of width")
    parser.add_argument("--wireframe", action="store_true")
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
            print(f"skipping empty mesh: {path}")
            continue

        print(f"{path.name}: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris, "
              f"{mesh.get_surface_area():.1f} m2")

        if args.grey or not mesh.has_vertex_colors():
            mesh.paint_uniform_color(GREY)
        mesh.compute_vertex_normals()

        width = mesh.get_axis_aligned_bounding_box().get_extent()[0]
        mesh.translate((offset, 0.0, 0.0))
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
