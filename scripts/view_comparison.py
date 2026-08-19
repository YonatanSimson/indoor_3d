"""Show the ARKit mesh and a fused mesh side by side in one Open3D window.

Both meshes live in the same world frame, so they would sit exactly on top of
each other; the fused one is translated sideways to put them next to each other
under a single shared camera.

Geometry is shaded in flat grey by default. Vertex colour is prettier but it
hides the thing being judged here -- blockiness and smeared edges read clearly
under plain shading and much less so under texture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence

ARKIT_GREY = (0.72, 0.72, 0.75)
FUSED_GREY = (0.75, 0.78, 0.80)


def prepare(
    mesh: o3d.geometry.TriangleMesh, colour: tuple[float, float, float], use_vertex_colour: bool
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)
    if not use_vertex_colour or not mesh.has_vertex_colors():
        mesh.paint_uniform_color(colour)
    mesh.compute_vertex_normals()
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=r"C:\Users\user\work\ARKit3d\sample_data\Validation"
    )
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument(
        "--fused", default="output/42444946_v10mm.ply", help="fused mesh to compare"
    )
    parser.add_argument(
        "--vertex-colour",
        action="store_true",
        help="show captured RGB instead of flat shading",
    )
    parser.add_argument(
        "--gap", type=float, default=0.15, help="gap between meshes, as a fraction of width"
    )
    args = parser.parse_args()

    sequence = Sequence(args.root, args.video_id)
    baseline_path = sequence.arkit_mesh_path
    if baseline_path is None:
        print("ERROR: no ARKit mesh found", file=sys.stderr)
        return 1

    fused_path = Path(args.fused)
    if not fused_path.is_file():
        print(f"ERROR: fused mesh not found: {fused_path}", file=sys.stderr)
        return 1

    arkit = o3d.io.read_triangle_mesh(str(baseline_path))
    fused = o3d.io.read_triangle_mesh(str(fused_path))
    print(f"ARKit : {len(arkit.vertices):>9,} verts  {len(arkit.triangles):>10,} tris")
    print(f"fused : {len(fused.vertices):>9,} verts  {len(fused.triangles):>10,} tris")

    arkit = prepare(arkit, ARKIT_GREY, args.vertex_colour)
    fused = prepare(fused, FUSED_GREY, args.vertex_colour)

    width = arkit.get_axis_aligned_bounding_box().get_extent()[0]
    fused.translate((width * (1.0 + args.gap), 0.0, 0.0))

    print("\nleft  = ARKit on-device mesh")
    print(f"right = {fused_path.name}")
    print("\ndrag to orbit, scroll to zoom, close the window when done")

    o3d.visualization.draw_geometries(
        [arkit, fused],
        window_name=f"ARKit (left) vs {fused_path.stem} (right)",
        width=1800,
        height=1000,
        mesh_show_back_face=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
