"""Fuse MVS depth maps into a point cloud, optionally rejecting against LiDAR.

`stereo_fusion` merges the per-view depth maps into one cloud, keeping points
that several views agree on. Its own consistency checks catch depth that is
merely inconsistent; they cannot catch depth that is consistently wrong, which
is what a blank wall or a repeating carpet pattern produces -- every view agrees,
and they are all agreeing on the same mistake.

LiDAR catches that, because it does not fail where photometric matching fails.
Points contradicting a confident LiDAR reading are dropped; points where LiDAR
had nothing are kept, since that is exactly the territory MVS is meant to cover.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_COLMAP_ENV = r"C:\Users\User\anaconda3\envs\colmap"


def run_colmap(colmap_env: Path, command: str, **options) -> None:
    argv = [
        "conda",
        "run",
        "--no-capture-output",
        "--prefix",
        str(colmap_env),
        "colmap.exe",
        command,
    ]
    for key, value in options.items():
        if value is not None:
            argv += [f"--{key.replace('__', '.')}", str(value)]
    print(f"$ colmap {command}", flush=True)
    started = time.perf_counter()
    if subprocess.run(argv, check=False).returncode != 0:
        raise SystemExit(f"colmap {command} failed")
    print(f"  [{time.perf_counter() - started:.1f}s]", flush=True)


def reject_against_lidar(
    cloud_path: Path, session: Path, max_abs: float, max_rel: float, out_path: Path
) -> None:
    """Drop fused points that contradict a confident LiDAR reading."""
    import cv2
    import open3d as o3d
    import pycolmap

    from indoor3d.data_rawrecording import RawRecordingSequence
    from indoor3d.depth_filters import clean_depth

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise SystemExit(f"{cloud_path} is empty")

    sequence = RawRecordingSequence(session, intrinsics_space="rgb")
    by_name = {sequence[i].folder.name: sequence[i] for i in range(len(sequence))}
    model = pycolmap.Reconstruction(str(session / "colmap" / "sparse" / "mvs_refs_tri"))

    # The test has to be asymmetric, because disagreement has two very different
    # causes. If the LiDAR reads *further* than the point, the point floats in
    # space the sensor saw through -- a free-space violation, and the point is
    # wrong. If the LiDAR reads *nearer*, something simply occludes the point in
    # that view, which says nothing about the point at all.
    #
    # Treating both as disagreement rejects almost everything: with 119 views
    # nearly every point is occluded somewhere, and a single-view veto then
    # condemns it. Measured once at 97.5% rejected, against 11% of pixels
    # actually disagreeing.
    votes_against = np.zeros(len(points), np.int32)
    votes_for = np.zeros(len(points), np.int32)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])

    for image in model.images.values():
        frame = by_name.get(Path(image.name).stem)
        if frame is None or not image.has_pose:
            continue
        confidence = sequence.load_confidence(frame)
        lidar, _ = clean_depth(sequence.load_depth(frame), confidence, min_confidence=1)
        dh, dw = lidar.shape
        intrinsics, rgb_w, rgb_h = sequence.intrinsics(frame)
        sx, sy = dw / rgb_w, dh / rgb_h

        world_to_cam = np.asarray(image.cam_from_world().matrix())
        camera = homogeneous @ world_to_cam.T
        z = camera[:, 2]
        in_front = z > 0
        if not in_front.any():
            continue

        u = (
            intrinsics[0, 0] * camera[:, 0] / np.where(in_front, z, 1)
            + intrinsics[0, 2]
        ) * sx
        v = (
            intrinsics[1, 1] * camera[:, 1] / np.where(in_front, z, 1)
            + intrinsics[1, 2]
        ) * sy
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = in_front & (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
        if not inside.any():
            continue

        measured = np.zeros(len(points), np.float32)
        measured[inside] = lidar[vi[inside], ui[inside]]
        usable = inside & (measured > 0)
        if not usable.any():
            continue

        threshold = np.maximum(max_abs, max_rel * measured)
        # measured - z > threshold: the sensor saw past where the point claims
        # to be, so the point is in free space.
        votes_against += (usable & ((measured - z) > threshold)).astype(np.int32)
        votes_for += (usable & (np.abs(z - measured) <= threshold)).astype(np.int32)

    # A single view can be wrong, so require the objections to outweigh the
    # support rather than letting one veto decide.
    keep = ~((votes_against > 0) & (votes_against >= votes_for))
    filtered = cloud.select_by_index(np.nonzero(keep)[0])
    o3d.io.write_point_cloud(str(out_path), filtered)

    never_seen = (votes_against + votes_for) == 0
    print(
        f"LiDAR rejection: {len(points):,} -> {len(filtered.points):,} points "
        f"({(~keep).mean():.1%} sat in space the LiDAR saw through)"
    )
    print(f"  supported by at least one view: {(votes_for > 0).mean():.1%}")
    print(f"  never measured by LiDAR (kept): {never_seen.mean():.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session", default=r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--colmap-env", default=DEFAULT_COLMAP_ENV)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--skip-lidar-rejection", action="store_true")
    parser.add_argument("--reject-abs", type=float, default=0.05)
    parser.add_argument("--reject-rel", type=float, default=0.05)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    raw_out = (
        Path(args.out) if args.out else Path("output") / f"mvs_{workspace.name}.ply"
    )
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    run_colmap(
        Path(args.colmap_env),
        "stereo_fusion",
        workspace_path=workspace,
        workspace_format="COLMAP",
        input_type="geometric",
        output_path=raw_out,
        StereoFusion__max_image_size=args.max_image_size,
    )
    print(f"wrote {raw_out}")

    if not args.skip_lidar_rejection:
        filtered = raw_out.with_name(raw_out.stem + "_lidarfilt.ply")
        reject_against_lidar(
            raw_out, Path(args.session), args.reject_abs, args.reject_rel, filtered
        )
        print(f"wrote {filtered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
