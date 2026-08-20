"""Phase 1 fusion for a RawRecorder session captured by the model_3d app.

With `--compare-conventions` it fuses the sequence both ways and reports which
camera convention holds together. There is no reference mesh for an ad-hoc
recording, so the test is self-consistency: when the convention is right, every
frame's depth lands on the same surfaces and they reinforce each other; when it
is wrong, each frame is mirrored about its own camera axes and the surfaces
scatter, inflating surface area and pushing the measurements off the mesh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth
from indoor3d.fusion import FusionConfig, fuse, save_mesh


def measurement_error(
    sequence: RawRecordingSequence,
    mesh: o3d.geometry.TriangleMesh,
    convention: str,
    num_frames: int,
    stride: int,
    min_confidence: int,
) -> tuple[float, float]:
    """Median and mean distance from raw depth points to the mesh, in metres."""
    if len(mesh.triangles) == 0:
        return float("nan"), float("nan")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    clouds = []
    for index in np.linspace(0, len(sequence) - 1, num_frames, dtype=int):
        frame = sequence[index]
        depth, _ = clean_depth(
            sequence.load_depth(frame),
            sequence.load_confidence(frame),
            min_confidence=min_confidence,
        )
        intrinsics, _, _ = sequence.intrinsics(frame)

        sub = depth[::stride, ::stride]
        rows, cols = np.indices(sub.shape)
        valid = sub > 0
        if not valid.any():
            continue

        z = sub[valid]
        u = (cols * stride)[valid]
        v = (rows * stride)[valid]
        x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points = np.stack([x, y, z], axis=1)

        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        clouds.append(
            (homogeneous @ sequence.camera_to_world(frame, convention).T)[:, :3]
        )

    if not clouds:
        return float("nan"), float("nan")

    query = o3d.core.Tensor(
        np.concatenate(clouds).astype(np.float32), dtype=o3d.core.Dtype.Float32
    )
    distances = scene.compute_distance(query).numpy()
    return float(np.median(distances)), float(distances.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--traj-convention", default="rub", choices=["rub", "rdf"])
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=1,
        help="drop depth below this ARKit confidence level (0 keeps everything)",
    )
    parser.add_argument(
        "--upsample",
        default="none",
        choices=["none", "bilinear", "joint", "solver", "anchored"],
        help="Phase 3: raise depth to a finer grid using RGB as a guide",
    )
    parser.add_argument("--upsample-scale", type=float, default=2.0)
    parser.add_argument("--no-bilateral", action="store_true")
    parser.add_argument("--keep-flying-pixels", action="store_true")
    parser.add_argument("--compare-conventions", action="store_true")
    parser.add_argument(
        "--refined-poses",
        nargs="?",
        const="auto",
        default=None,
        help="fuse with COLMAP-refined poses; bare flag looks for "
        "<session>/colmap/refined_poses.json",
    )
    args = parser.parse_args()

    sequence = RawRecordingSequence(args.session)
    print(f"session: {Path(args.session).name}")
    print(f"{len(sequence)} frames, confidence maps: {sequence.has_confidence}")

    if args.refined_poses:
        poses_path = (
            Path(args.session) / "colmap" / "refined_poses.json"
            if args.refined_poses == "auto"
            else Path(args.refined_poses)
        )
        if not poses_path.is_file():
            raise SystemExit(
                f"no refined poses at {poses_path}; run scripts/run_colmap_raw.py first"
            )
        matched = sequence.load_refined_poses(poses_path)
        print(
            f"using refined poses for {matched}/{len(sequence)} frames ({poses_path.name})"
        )
        if matched < len(sequence):
            print(
                f"  {len(sequence) - matched} unregistered frames keep their ARKit pose"
            )

    def build(convention: str) -> tuple[o3d.geometry.TriangleMesh, object]:
        config = FusionConfig(
            voxel_length=args.voxel,
            max_depth=args.max_depth,
            frame_stride=args.stride,
            traj_convention=convention,
            bilateral=not args.no_bilateral,
            drop_flying_pixels=not args.keep_flying_pixels,
            min_confidence=args.min_confidence,
            upsample_method=args.upsample,
            upsample_scale=args.upsample_scale,
        )
        return fuse(sequence, config, progress_every=0)

    if args.compare_conventions:
        print(
            f"\nfusing both conventions at {args.voxel * 1000:.0f} mm to see which coheres\n"
        )
        header = f"{'convention':>10} | {'verts':>9} | {'tris':>10} | {'area m2':>8} | {'median mm':>9} | {'mean mm':>8}"
        print(header)
        print("-" * len(header))

        scores = {}
        for convention in ("rub", "rdf"):
            mesh, _ = build(convention)
            median, mean = measurement_error(
                sequence, mesh, convention, 40, 4, args.min_confidence
            )
            scores[convention] = mean
            print(
                f"{convention:>10} | {len(mesh.vertices):>9,} | {len(mesh.triangles):>10,} | "
                f"{mesh.get_surface_area():>8.1f} | {median * 1000:>9.1f} | {mean * 1000:>8.1f}"
            )

        best = min(scores, key=scores.get)
        print(
            f"\nverdict: '{best}' -- measurements sit closest to the surface it builds"
        )
        return 0

    print(
        f"\nvoxel {args.voxel * 1000:.0f} mm, convention '{args.traj_convention}', "
        f"min confidence {args.min_confidence}"
    )
    mesh, report = build(args.traj_convention)

    print(f"integrated {report.frames_integrated} frames in {report.seconds:.1f}s")
    totals = report.rejection_totals
    kept = totals.get("kept", 0)
    discarded = {k: v for k, v in totals.items() if k != "kept"}
    total_seen = kept + sum(discarded.values())
    if total_seen:
        print("depth pixels:")
        for name, value in sorted(discarded.items(), key=lambda kv: -kv[1]):
            print(
                f"  rejected {name:>18}: {int(value):>12,} ({value / total_seen:.1%})"
            )
        print(f"  {'kept':>27}: {int(kept):>12,} ({kept / total_seen:.1%})")

    print(
        f"\nmesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles, "
        f"{mesh.get_surface_area():.1f} m2"
    )

    out_path = Path(
        args.out
        or f"output/raw_{Path(args.session).name}_v{int(args.voxel*1000)}mm.ply"
    )
    print(f"wrote {save_mesh(mesh, out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
