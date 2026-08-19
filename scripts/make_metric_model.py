"""Phase 4, stage 0: turn the sparse model into a clean, metric one.

Two problems have to be fixed before dense stereo runs, and both are silent:

**Scale.** Bundle adjustment leaves the reconstruction in an arbitrary gauge --
ours came out 3.8x off metric. Dense stereo inherits whatever the model says, so
its output would be fused into a metric LiDAR TSDF at the wrong size, and
nothing would announce it.

The scale is recovered from the LiDAR rather than from the ARKit alignment.
Every sparse point is observed in some image at a known pixel; the LiDAR has a
metric reading at that same pixel; the ratio of the two depths is the scale.
That is an independent reference, whereas reusing the ARKit similarity would
inherit any error in that trajectory -- and comparing the two afterwards
cross-checks both for free.

**Outlier points.** One point at 130 m drags the model's mean reprojection error
to 1e152. Those points also feed source-view selection, which ranks candidate
views by shared point count, so bad points distort which views get matched
against which.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence
from indoor3d.depth_filters import clean_depth

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"


def lidar_scale(
    reconstruction: pycolmap.Reconstruction,
    sequence: RawRecordingSequence,
    max_images: int,
    min_confidence: int,
) -> tuple[float, dict[str, float]]:
    """Median ratio of LiDAR depth to model depth over sparse observations.

    For each observed sparse point: its depth in that camera comes from the
    model, and the LiDAR gives a metric depth at the same pixel. Their ratio is
    the scale, estimated robustly because plenty of individual observations sit
    on depth discontinuities where the two legitimately disagree.
    """
    by_name = {sequence[i].folder.name: sequence[i] for i in range(len(sequence))}
    ratios: list[np.ndarray] = []
    used_images = 0

    image_ids = sorted(reconstruction.images)
    step = max(1, len(image_ids) // max_images)

    for image_id in image_ids[::step]:
        image = reconstruction.images[image_id]
        frame = by_name.get(Path(image.name).stem)
        if frame is None:
            continue

        confidence = sequence.load_confidence(frame)
        lidar, _ = clean_depth(
            sequence.load_depth(frame), confidence, min_confidence=min_confidence
        )
        depth_h, depth_w = lidar.shape
        _, rgb_w, rgb_h = sequence.intrinsics(frame)
        sx, sy = depth_w / rgb_w, depth_h / rgb_h

        world_to_cam = np.asarray(image.cam_from_world().matrix())  # 3x4

        model_depths, lidar_depths = [], []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            xyz = reconstruction.points3D[point2D.point3D_id].xyz
            # Depth along the optical axis, which is what LiDAR also reports.
            z = float(world_to_cam[2, :3] @ xyz + world_to_cam[2, 3])
            if z <= 0:
                continue

            u = int(round(point2D.xy[0] * sx))
            v = int(round(point2D.xy[1] * sy))
            if not (0 <= u < depth_w and 0 <= v < depth_h):
                continue
            measured = lidar[v, u]
            if measured <= 0:
                continue

            model_depths.append(z)
            lidar_depths.append(measured)

        if len(model_depths) < 20:
            continue
        ratios.append(np.asarray(lidar_depths) / np.asarray(model_depths))
        used_images += 1

    if not ratios:
        raise SystemExit("no sparse observations coincided with LiDAR readings")

    all_ratios = np.concatenate(ratios)
    scale = float(np.median(all_ratios))

    # Per-image medians show whether the scale is a property of the model or
    # just noise: a consistent model gives the same ratio everywhere.
    per_image = np.array([np.median(r) for r in ratios])
    return scale, {
        "images": used_images,
        "observations": len(all_ratios),
        "spread_p10": float(np.percentile(per_image, 10)),
        "spread_p90": float(np.percentile(per_image, 90)),
        "relative_spread_pct": float(
            (np.percentile(per_image, 90) - np.percentile(per_image, 10)) / scale * 100
        ),
    }


def filter_points(
    reconstruction: pycolmap.Reconstruction, max_error: float, min_track: int
) -> dict[str, int]:
    """Drop points that are badly triangulated or barely observed.

    Two-view tracks are the usual source of runaway points: with only two rays
    and a small angle between them, the intersection slides far along the
    viewing direction, which is how a single point ends up 130 m away in a 10 m
    room.
    """
    before = len(reconstruction.points3D)
    reconstruction.update_point_3d_errors()

    doomed_error, doomed_track = [], []
    for point3D_id, point in reconstruction.points3D.items():
        if point.track.length() < min_track:
            doomed_track.append(point3D_id)
        elif point.has_error and point.error > max_error:
            doomed_error.append(point3D_id)

    for point3D_id in doomed_error + doomed_track:
        reconstruction.delete_point3D(point3D_id)

    return {
        "before": before,
        "removed_reprojection": len(doomed_error),
        "removed_short_track": len(doomed_track),
        "after": len(reconstruction.points3D),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--model", default=None, help="default: <session>/colmap/sparse/seed_src")
    parser.add_argument("--out", default=None, help="default: <session>/colmap/sparse/metric")
    parser.add_argument("--max-images", type=int, default=120, help="images sampled for scale")
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--max-reproj-error", type=float, default=4.0)
    parser.add_argument("--min-track-length", type=int, default=3)
    args = parser.parse_args()

    session = Path(args.session)
    model_dir = Path(args.model) if args.model else session / "colmap" / "sparse" / "seed_src"
    out_dir = Path(args.out) if args.out else session / "colmap" / "sparse" / "metric"

    sequence = RawRecordingSequence(session, intrinsics_space="rgb")
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    print(f"loaded {model_dir.name}: {reconstruction.num_images()} images, "
          f"{len(reconstruction.points3D):,} points")

    stats = filter_points(reconstruction, args.max_reproj_error, args.min_track_length)
    print(f"\nfiltering: {stats['before']:,} -> {stats['after']:,} points "
          f"({stats['removed_reprojection']:,} by reprojection > {args.max_reproj_error}px, "
          f"{stats['removed_short_track']:,} with track < {args.min_track_length})")

    scale, report = lidar_scale(reconstruction, sequence, args.max_images, args.min_confidence)
    print(f"\nLiDAR scale from {report['observations']:,} observations "
          f"across {report['images']} images:")
    print(f"  scale = {scale:.5f}   (model is {1/scale:.3f}x larger than metric)")
    print(f"  per-image spread p10-p90: {report['spread_p10']:.4f} - {report['spread_p90']:.4f} "
          f"({report['relative_spread_pct']:.1f}% of scale)")

    # Cross-check against the independently derived ARKit alignment.
    import json
    poses_path = session / "colmap" / "refined_poses.json"
    if poses_path.is_file():
        arkit_scale = json.loads(poses_path.read_text())["alignment"]["scale"]
        disagreement = abs(scale - arkit_scale) / arkit_scale * 100
        print(f"\ncross-check vs align_to_arkit: {arkit_scale:.5f} "
              f"(disagreement {disagreement:.1f}%)")
        if disagreement > 5:
            print("  WARNING: these are independent estimates and should agree closely.")
            print("  A gap this large means one of them is wrong -- resolve before dense stereo.")
        else:
            print("  the two independent estimates agree; the gauge is trustworthy")

    reconstruction.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(), np.zeros(3)))
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(out_dir))
    print(f"\nwrote metric model to {out_dir}")

    centres = np.array([img.projection_center() for img in reconstruction.images.values()])
    print(f"camera-centre extent after rescale: {np.round(centres.max(0) - centres.min(0), 2)} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
