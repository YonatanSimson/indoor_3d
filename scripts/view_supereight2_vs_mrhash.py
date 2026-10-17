"""Side-by-side Open3D viewer: supereight2 vs mrhash, the current visual co-leaders."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

DATA = Path(r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941\replica")


def main() -> int:
    to_show = [
        ("supereight2 (10mm, CPU-only)", DATA / "supereight2_mesh_10mm.ply"),
        ("mrhash (10mm, GPU)", DATA / "mrhash_mesh_10mm.ply"),
    ]

    offset = 0.0
    geoms = []
    for name, path in to_show:
        if not path.is_file():
            print(f"skipping {name}: {path} not found")
            continue
        print(f"loading {name} ({path.stat().st_size / 1e6:.0f} MB)...")
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
        geoms, window_name="supereight2 | mrhash (left to right)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
