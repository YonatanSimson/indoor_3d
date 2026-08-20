"""Side-by-side Open3D viewer comparing our fusion, nvblox, and mrhash meshes."""

from __future__ import annotations

import sys
from pathlib import Path

import open3d as o3d

DATA = Path(r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941")
OURS = Path(
    r"C:\Users\user\work\indoor_3d\output\RawRecording_2026-08-18_215941_refined_anchored_v10mm.ply"
)


def main() -> int:
    to_show = [
        ("ours (anchored, 10mm)", OURS),
        ("nvblox (20mm)", DATA / "replica" / "nvblox_mesh_20mm.ply"),
        ("mrhash (10mm)", DATA / "replica" / "mrhash_mesh_10mm.ply"),
    ]

    offset = 0.0
    geoms = []
    for name, path in to_show:
        if not path.is_file():
            print(f"skipping {name}: {path} not found")
            continue
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.compute_vertex_normals()
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        mesh.translate((offset, 0, 0))
        offset += extent[0] * 1.3
        print(f"{name}: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
        geoms.append(mesh)

    if not geoms:
        print("nothing to show")
        return 1

    o3d.visualization.draw_geometries(
        geoms, window_name="ours | nvblox | mrhash (left to right)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
