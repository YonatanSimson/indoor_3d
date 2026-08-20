"""Phase 1: fuse a captured sequence into a mesh.

Example:
    python scripts/run_fusion.py --voxel 0.02 --out output/42444946_v20mm.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import Sequence
from indoor3d.fusion import FusionConfig, fuse, save_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=r"C:\Users\user\work\ARKit3d\sample_data\Validation",
        help="directory holding <video_id>/ capture folders",
    )
    parser.add_argument("--video-id", default="42444946")
    parser.add_argument("--out", default=None, help="output mesh path (.ply)")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.02,
        help="voxel size in metres (ARKit uses ~0.04-0.05)",
    )
    parser.add_argument("--sdf-trunc-voxels", type=float, default=4.0)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="offset into the stride; fuse with --stride 2 --offset 0 and evaluate "
        "with compare_meshes.py --holdout 2 for a disjoint train/test split",
    )
    parser.add_argument("--traj-convention", default="rdf", choices=["rdf", "rub"])
    parser.add_argument("--no-bilateral", action="store_true")
    parser.add_argument("--keep-flying-pixels", action="store_true")
    args = parser.parse_args()

    sequence = Sequence(args.root, args.video_id)
    config = FusionConfig(
        voxel_length=args.voxel,
        sdf_trunc_voxels=args.sdf_trunc_voxels,
        max_depth=args.max_depth,
        frame_stride=args.stride,
        frame_offset=args.offset,
        traj_convention=args.traj_convention,
        bilateral=not args.no_bilateral,
        drop_flying_pixels=not args.keep_flying_pixels,
    )

    out_path = Path(
        args.out or f"output/{args.video_id}_v{int(args.voxel * 1000)}mm.ply"
    )

    print(f"sequence {args.video_id}: {len(sequence)} posed frames")
    print(
        f"voxel {config.voxel_length * 1000:.0f} mm, sdf_trunc {config.sdf_trunc * 1000:.0f} mm"
    )
    print(
        f"bilateral={config.bilateral} drop_flying_pixels={config.drop_flying_pixels}"
    )
    if not sequence.has_confidence:
        print(
            "note: no confidence maps in this capture, so confidence weighting is inactive"
        )
    print("fusing...")

    mesh, report = fuse(sequence, config)

    print(f"\nintegrated {report.frames_integrated} frames in {report.seconds:.1f}s")
    if report.frames_skipped:
        print(f"skipped {report.frames_skipped} frames")

    totals = report.rejection_totals
    kept = totals.get("kept", 0)
    discarded = {k: v for k, v in totals.items() if k != "kept"}
    total_seen = kept + sum(discarded.values())
    if total_seen:
        print("depth pixels:")
        for name, value in sorted(discarded.items(), key=lambda kv: -kv[1]):
            print(
                f"  rejected {name:>18}: {int(value):>10,} ({value / total_seen:.1%})"
            )
        print(f"  {'kept':>27}: {int(kept):>10,} ({kept / total_seen:.1%})")

    print(f"\nmesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    saved = save_mesh(mesh, out_path)
    print(f"wrote {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
