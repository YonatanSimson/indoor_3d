"""Dense MVS with perfect poses vs. our TSDF fusion: isolate the real question.

Phase 4 (`phase4_plan.md`) found dense MVS negative on the real capture --
20.8mm accuracy at 11% coverage, against the anchored solver's 4.5mm at
100%. But that result is confounded: COLMAP-refined poses (not perfect),
and a tight, textureless, close-range scene (median depth 1.03m). This
repeats the experiment under conditions that remove both confounds at once:
ICL-NUIM's ground-truth poses (perfect, no gauge/scale ambiguity to fix --
a real simplification over Phase 4's stage 0) and a room with roughly 2.4x
the median depth, more texture (furniture, not blank walls).

If MVS is still poor here, Phase 4's verdict generalizes: the geometry
itself (close range, weak texture) was the problem, not pose noise. If MVS
does well, pose noise was hiding real potential in the original attempt.

Unlike Phase 4 (scored against held-out LiDAR, this project's own
now-distrusted proxy), this scores directly against `living-room.ply` --
the same ground truth every other Phase V number in this repo uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.colmap_utils import (
    read_database,
    resolve_model_dir,
    rotation_to_quaternion,
    run_colmap,
)
from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.iclnuim_eval import load_ground_truth_tree

DEFAULT_ROOT = r"C:\Users\user\data\iclnuim"
DEFAULT_COLMAP_ENV = r"C:\Users\User\anaconda3\envs\colmap"
FX, FY_MAGNITUDE, CX, CY = 481.20, 480.00, 319.50, 239.50


def _flip_and_write(source: Path, destination: Path) -> None:
    """Same fy-sign fix as `run_colmap_iclnuim.py`: flip the image, not the pose."""
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read {source}")
    cv2.imwrite(str(destination), cv2.flip(image, 0))


def prepare_images(sequence: ICLNUIMSequence, stride: int, images_dir: Path) -> dict[str, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, int] = {}
    for index in range(0, len(sequence), stride):
        frame = sequence[index]
        name = f"{frame.frame_id}.png"
        mapping[name] = index
        destination = images_dir / name
        if not destination.exists():
            _flip_and_write(frame.rgb_path, destination)
    print(f"prepared {len(mapping)} flipped images (stride {stride}) in {images_dir}")
    return mapping


def write_gt_seed_model(
    sequence: ICLNUIMSequence,
    gt_poses: dict[int, np.ndarray],
    rows: list[tuple[int, str, int]],
    name_to_index: dict[str, int],
    model_dir: Path,
) -> None:
    """Ground-truth poses as a COLMAP text model, unperturbed, unrefined.

    No gauge/scale recovery needed -- unlike Phase 4's sparse model (3.8x off
    metric), these poses are already exactly metric and already correct.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    image_lines: list[str] = []

    for image_id, name, camera_id in rows:
        index = name_to_index.get(name)
        if index is None:
            continue
        frame = sequence[index]
        world_to_camera = np.linalg.inv(gt_poses[frame.frame_id])
        q = rotation_to_quaternion(world_to_camera[:3, :3])
        t = world_to_camera[:3, 3]
        image_lines.append(
            f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
            f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {camera_id} {name}"
        )

    with (model_dir / "cameras.txt").open("w") as handle:
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id in sorted({row[2] for row in rows}):
            handle.write(f"{camera_id} PINHOLE 640 480 {FX:.6f} {FY_MAGNITUDE:.6f} {CX:.6f} {CY:.6f}\n")

    with (model_dir / "images.txt").open("w") as handle:
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for line in image_lines:
            handle.write(line + "\n\n")

    (model_dir / "points3D.txt").write_text("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    print(f"wrote GT seed model: {len(image_lines)} images -> {model_dir}")


def score_point_cloud(
    points: np.ndarray, gt_tree: cKDTree, gt_points: np.ndarray, max_distance: float
) -> None:
    acc_dist, _ = gt_tree.query(points, workers=-1)
    acc_dist = acc_dist[acc_dist < max_distance]

    fused_tree = cKDTree(points)
    comp_dist, _ = fused_tree.query(gt_points, workers=-1)
    comp_dist = comp_dist[comp_dist < max_distance]

    print(f"\n{len(points):,} dense points")
    print(f"accuracy  (MVS -> GT):  median {np.median(acc_dist)*1000:.1f}mm  mean {acc_dist.mean()*1000:.1f}mm")
    print(f"completeness (GT -> MVS, <5cm): {(comp_dist < 0.05).mean():.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=None, help="default: <root>/colmap_dense")
    parser.add_argument("--colmap-env", default=DEFAULT_COLMAP_ENV)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--matcher", default="exhaustive", choices=["sequential", "exhaustive"])
    parser.add_argument("--max-image-size", type=int, default=640)
    parser.add_argument("--cache-size", type=int, default=8, help="GB, patch_match_stereo cache")
    parser.add_argument("--gt-voxel", type=float, default=0.005)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "colmap_dense"
    colmap_env = Path(args.colmap_env)
    if not (colmap_env / "Library" / "bin" / "colmap.exe").is_file():
        raise SystemExit(f"no colmap.exe under {colmap_env}")

    sequence = ICLNUIMSequence(root)
    gt_poses = sequence.ground_truth_poses()
    print(f"sequence: {len(sequence)} frames, ground-truth poses (no perturbation, no refinement)")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    database_path = out_dir / "database.db"
    seed_dir = out_dir / "sparse" / "gt_seed"
    triangulated_dir = out_dir / "sparse" / "triangulated"
    undistorted_dir = out_dir / "dense"

    name_to_index = prepare_images(sequence, args.stride, images_dir)
    use_gpu = 0 if args.no_gpu else 1

    run_colmap(
        colmap_env,
        "feature_extractor",
        database_path=database_path,
        image_path=images_dir,
        ImageReader__camera_model="PINHOLE",
        ImageReader__single_camera=1,
        ImageReader__camera_params=f"{FX:.6f},{FY_MAGNITUDE:.6f},{CX:.6f},{CY:.6f}",
        FeatureExtraction__use_gpu=use_gpu,
    )

    if args.matcher == "exhaustive":
        run_colmap(colmap_env, "exhaustive_matcher", database_path=database_path, FeatureMatching__use_gpu=use_gpu)
    else:
        run_colmap(
            colmap_env,
            "sequential_matcher",
            database_path=database_path,
            SequentialMatching__overlap=10,
            SequentialMatching__quadratic_overlap=1,
            FeatureMatching__use_gpu=use_gpu,
        )

    rows = read_database(database_path)
    write_gt_seed_model(sequence, gt_poses, rows, name_to_index, seed_dir)

    triangulated_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        colmap_env,
        "point_triangulator",
        database_path=database_path,
        image_path=images_dir,
        input_path=seed_dir,
        output_path=triangulated_dir,
    )
    triangulated_dir = resolve_model_dir(triangulated_dir)

    undistorted_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        colmap_env,
        "image_undistorter",
        image_path=images_dir,
        input_path=triangulated_dir,
        output_path=undistorted_dir,
        output_type="COLMAP",
        max_image_size=args.max_image_size,
    )

    run_colmap(
        colmap_env,
        "patch_match_stereo",
        workspace_path=undistorted_dir,
        workspace_format="COLMAP",
        PatchMatchStereo__geom_consistency=1,
        PatchMatchStereo__cache_size=args.cache_size,
        PatchMatchStereo__gpu_index=-1 if args.no_gpu else 0,
    )

    fused_path = undistorted_dir / "fused.ply"
    run_colmap(
        colmap_env,
        "stereo_fusion",
        workspace_path=undistorted_dir,
        workspace_format="COLMAP",
        input_type="geometric",
        output_path=fused_path,
    )

    if not fused_path.is_file():
        raise SystemExit(f"stereo_fusion did not produce {fused_path}")
    cloud = o3d.io.read_point_cloud(str(fused_path))
    points = np.asarray(cloud.points)
    print(f"\nfused dense point cloud: {len(points):,} points")

    gt_tree, gt_points = load_ground_truth_tree(sequence.ground_truth_cloud_path, args.gt_voxel)
    score_point_cloud(points, gt_tree, gt_points, args.max_distance)

    print("\nfor comparison, our TSDF fusion with GT poses on the same sequence:")
    print("  full 640x480 depth : 6.8mm median (the ceiling)")
    print("  degraded 256x192   : 10.4mm median")
    print("  Phase 4 (real capture, COLMAP-refined poses): 20.8mm accuracy, 11% coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
