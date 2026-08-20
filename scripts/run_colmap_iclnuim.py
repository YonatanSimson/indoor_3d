"""Pose and rotation accuracy: run the real COLMAP pipeline on ICL-NUIM.

The pose-ceiling experiment (`eval_iclnuim_pose_ceiling.py`) tests
*sensitivity* to a hypothetical pose error. This tests the actual pose
*estimator*: seed COLMAP the same way `run_colmap_raw.py` seeds it for a real
capture -- an approximate, drifted onboard pose, not perfect ground truth --
then measure how far COLMAP's refined poses land from ICL-NUIM's true
trajectory, in both translation and rotation.

ICL-NUIM has no onboard VIO of its own, so the seed is built by perturbing
the true poses by the same ~7.5mm this repo's Phase 2 measured between
ARKit and COLMAP (`pose_perturbation.perturb_poses`) -- standing in for "an
iPhone recording's approximate pose with slight drift", per the real
pipeline's own seeding assumption.

Metrics are computed with `evo` (github.com/MichaelGrupp/evo), the standard
trajectory-evaluation package, not hand-rolled: APE (Absolute Pose Error,
i.e. ATE) after Sim3 (scale-correcting) alignment for overall accuracy, and
RPE (Relative Pose Error) at 1-frame delta for local drift rate. Both are
reported for translation and for rotation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.colmap_utils import (
    read_database,
    read_model_images,
    resolve_model_dir,
    rotation_to_quaternion,
    run_colmap,
)
from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.pose_perturbation import perturb_poses

DEFAULT_ROOT = r"C:\Users\user\data\iclnuim"
DEFAULT_COLMAP_ENV = r"C:\Users\User\anaconda3\envs\colmap"

FX, FY_MAGNITUDE, CX, CY = 481.20, 480.00, 319.50, 239.50


# ICL-NUIM's camera has fy = -480.00 in *our* back-projection convention
# (`indoor3d/data_iclnuim.py` confirms this empirically); COLMAP's PINHOLE
# model has no such sign, it always assumes standard positive-fy pixels.
#
# The fix is NOT to flip the pose: negating one row of a proper rotation
# turns it into an improper one (det -1, a reflection), which is not a
# valid camera orientation and is exactly what silently wrecked the first
# version of this script -- point_triangulator saw 0-30 inlier points per
# image instead of hundreds, because every triangulated ray had its
# chirality flipped. Flipping the *image* instead leaves the pose
# untouched and reproduces the same geometry: v_flipped = H-1-v makes
# `(v_flipped - cy) = -(v - cy)` when cy sits at the image's vertical
# centre (239.5 for a 480-tall image, which it does here), so a standard
# positive-fy back-projection of the flipped image lands on the same 3D
# ray our negative-fy formula does on the original.
def _flip_and_write(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read {source}")
    cv2.imwrite(str(destination), cv2.flip(image, 0))


def prepare_images(
    sequence: ICLNUIMSequence, stride: int, images_dir: Path
) -> dict[str, int]:
    """Vertically flip and write every `stride`-th frame's RGB for COLMAP."""
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


def write_seed_model(
    sequence: ICLNUIMSequence,
    seed_poses: dict[int, np.ndarray],
    rows: list[tuple[int, str, int]],
    name_to_index: dict[str, int],
    model_dir: Path,
) -> None:
    """Write perturbed-GT poses as a COLMAP text model, one shared camera.

    Unlike ARKit's per-frame autofocus, ICL-NUIM's intrinsics are fixed and
    precisely known, so every image shares one PINHOLE camera here. Poses go
    in completely unchanged -- the fy-sign correction lives entirely in
    `prepare_images`' vertical flip, not here (see that function's docstring).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    image_lines: list[str] = []

    for image_id, name, camera_id in rows:
        index = name_to_index.get(name)
        if index is None:
            continue
        frame = sequence[index]
        camera_to_world = seed_poses[frame.frame_id]
        world_to_camera = np.linalg.inv(camera_to_world)

        q = rotation_to_quaternion(world_to_camera[:3, :3])
        t = world_to_camera[:3, 3]
        image_lines.append(
            f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
            f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {camera_id} {name}"
        )

    with (model_dir / "cameras.txt").open("w") as handle:
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        camera_ids = sorted({row[2] for row in rows})
        for camera_id in camera_ids:
            handle.write(
                f"{camera_id} PINHOLE 640 480 {FX:.6f} {FY_MAGNITUDE:.6f} {CX:.6f} {CY:.6f}\n"
            )

    with (model_dir / "images.txt").open("w") as handle:
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for line in image_lines:
            handle.write(line + "\n\n")

    (model_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
    )
    print(f"wrote seed model: {len(image_lines)} images -> {model_dir}")


def _orthonormalize(pose: np.ndarray) -> np.ndarray:
    """Snap a pose's rotation block to the nearest proper SO(3) element.

    Chained matrix inversions and a Sim3 alignment accumulate floating-point
    drift that `evo`'s rotation-error computation has no tolerance for --
    `so3_log` raises outright on a rotation block that isn't exactly
    orthonormal, rather than silently accepting a close approximation.
    `Rotation.from_matrix` already does the nearest-rotation projection
    (SVD-based) internally, so this reuses that instead of hand-rolling one.
    """
    cleaned = pose.copy()
    cleaned[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    return cleaned


def to_trajectory(poses: dict[int, np.ndarray]) -> PoseTrajectory3D:
    """Build an evo trajectory from frame id -> camera_to_world, id as timestamp."""
    ids = sorted(poses)
    poses_se3 = [_orthonormalize(poses[i]) for i in ids]
    timestamps = np.asarray(ids, dtype=np.float64)
    return PoseTrajectory3D(poses_se3=poses_se3, timestamps=timestamps)


def report_metric(name: str, values_mm_or_deg: np.ndarray) -> None:
    print(
        f"{name:>28} | median {np.median(values_mm_or_deg):8.2f} | "
        f"mean {values_mm_or_deg.mean():8.2f} | rmse {np.sqrt((values_mm_or_deg**2).mean()):8.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=None, help="default: <root>/colmap")
    parser.add_argument("--colmap-env", default=DEFAULT_COLMAP_ENV)
    parser.add_argument("--stride", type=int, default=4, help="use every Nth frame")
    parser.add_argument(
        "--matcher", default="sequential", choices=["sequential", "exhaustive"]
    )
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument(
        "--translation-std", type=float, default=0.0075, help="seed pose noise, metres"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "colmap"
    colmap_env = Path(args.colmap_env)
    if not (colmap_env / "Library" / "bin" / "colmap.exe").is_file():
        raise SystemExit(f"no colmap.exe under {colmap_env}")

    sequence = ICLNUIMSequence(root)
    gt_poses = sequence.ground_truth_poses()
    seed_poses = perturb_poses(
        gt_poses, translation_std=args.translation_std, seed=args.seed
    )
    print(
        f"sequence: {len(sequence)} frames; seeding COLMAP with poses perturbed "
        f"{args.translation_std * 1000:.1f}mm from ground truth (stand-in for an approximate, "
        f"drifted onboard pose -- {len(range(0, len(sequence), args.stride))} frames at stride {args.stride})"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    database_path = out_dir / "database.db"
    seed_dir = out_dir / "sparse" / "seed"
    triangulated_dir = out_dir / "sparse" / "triangulated"
    refined_dir = out_dir / "sparse" / "refined"

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

    if args.matcher == "sequential":
        run_colmap(
            colmap_env,
            "sequential_matcher",
            database_path=database_path,
            SequentialMatching__overlap=args.overlap,
            SequentialMatching__quadratic_overlap=1,
            FeatureMatching__use_gpu=use_gpu,
        )
    else:
        run_colmap(
            colmap_env,
            "exhaustive_matcher",
            database_path=database_path,
            FeatureMatching__use_gpu=use_gpu,
        )

    rows = read_database(database_path)
    write_seed_model(sequence, seed_poses, rows, name_to_index, seed_dir)

    triangulated_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        colmap_env,
        "point_triangulator",
        database_path=database_path,
        image_path=images_dir,
        input_path=seed_dir,
        output_path=triangulated_dir,
    )

    refined_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        colmap_env,
        "bundle_adjuster",
        input_path=triangulated_dir,
        output_path=refined_dir,
        BundleAdjustment__refine_focal_length=0,
        BundleAdjustment__refine_principal_point=0,
        BundleAdjustment__refine_extra_params=0,
        # See run_colmap_raw.py: with one camera per rig this parameter left
        # at its default collapses the reconstruction (2.8x scale error
        # measured there); zero it here for the same reason.
        BundleAdjustment__refine_sensor_from_rig=0,
        BundleAdjustment__refine_rig_from_world=1,
    )

    refined_dir = resolve_model_dir(refined_dir)
    run_colmap(
        colmap_env,
        "model_converter",
        input_path=refined_dir,
        output_path=refined_dir,
        output_type="TXT",
    )

    refined_colmap = read_model_images(refined_dir)
    registered = len(refined_colmap)
    attempted = len(name_to_index)
    print(f"\nCOLMAP registered {registered}/{attempted} images")
    if registered < attempted * 0.5:
        print(
            "WARNING: fewer than half the attempted images registered; treat results with caution."
        )

    # No convention correction needed on the way out either -- the pose was
    # never touched, only the image it was seeded/refined against.
    estimated: dict[int, np.ndarray] = {}
    for name, world_to_camera in refined_colmap.items():
        index = name_to_index[name]
        frame_id = sequence[index].frame_id
        estimated[frame_id] = np.linalg.inv(world_to_camera)

    traj_ref = to_trajectory(
        {fid: pose for fid, pose in gt_poses.items() if fid in estimated}
    )
    traj_est = to_trajectory(estimated)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.5)
    print(f"synced {len(traj_ref.timestamps)} poses between ground truth and estimate")

    traj_est.align(traj_ref, correct_scale=True)
    print(
        "aligned estimated trajectory to ground truth (Sim3, evo.core.trajectory.align)"
    )

    print(f"\n{'metric':>28} | {'median':>8} | {'mean':>8} | {'rmse':>8}")
    print("-" * 66)

    ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_trans.process_data((traj_ref, traj_est))
    report_metric("APE translation (mm)", ape_trans.error * 1000)

    ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_rot.process_data((traj_ref, traj_est))
    report_metric("APE rotation (deg)", ape_rot.error)

    rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0)
    rpe_trans.process_data((traj_ref, traj_est))
    report_metric("RPE translation (mm/frame)", rpe_trans.error * 1000)

    rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=1.0)
    rpe_rot.process_data((traj_ref, traj_est))
    report_metric("RPE rotation (deg/frame)", rpe_rot.error)

    print(
        "\nAPE = absolute pose error after Sim3 alignment (i.e. ATE) -- overall accuracy"
    )
    print(
        "RPE = relative pose error at 1-frame delta -- local drift rate, alignment-independent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
