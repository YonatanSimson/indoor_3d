"""Phase V's decisive experiment: is pose accuracy or depth noise the ceiling?

`hand_off_original_plan.md` calls this the single result that decides where
the project goes next, and says it is currently a guess. It stops being one
here: fuse identical simulated-sensor depth at 5 mm twice, once with
ICL-NUIM's ground-truth poses and once with those poses perturbed by the
~7.5 mm this repo's own Phase 2 measured between COLMAP and ARKit, then score
both against `living-room.ply` -- true ground truth, not the raw-LiDAR proxy
every other metric in this repo relies on.

- If the GT-pose mesh comes out clean and the perturbed one crumples, pose
  accuracy is the ceiling: Phase 2 (pose refinement) is the lever, and finer
  voxels would help once poses improve.
- If both come out crumpled, depth noise is the limit regardless of poses:
  Phase 2 is near its useful end, and temporal averaging across frames is the
  lever instead.

Same simulated-sensor depth feeds both runs (`sensor_sim.DegradedSequence`
caches it once), so the only variable between them is the poses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data import PosedSequence
from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.fusion import FusionConfig, fuse, save_mesh
from indoor3d.iclnuim_eval import REPORT_HEADER, load_ground_truth_tree, report
from indoor3d.pose_perturbation import perturb_poses
from indoor3d.sensor_sim import DegradedSequence


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument("--out-dir", default=r"C:\Users\user\data\iclnuim_output")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.005,
        help="fusion voxel, metres -- 5mm per the hand-off doc",
    )
    parser.add_argument(
        "--gt-voxel",
        type=float,
        default=0.005,
        help="ground-truth cloud downsample, metres",
    )
    parser.add_argument(
        "--translation-std",
        type=float,
        default=0.0075,
        help="pose perturbation std, metres",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.3,
        help="ignore correspondences beyond this",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-degrade",
        action="store_true",
        help="bonus diagnostic: fuse native undegraded 640x480 depth instead of "
        "the simulated-sensor depth both experiment runs use by default",
    )
    args = parser.parse_args()

    sequence = ICLNUIMSequence(args.root)
    print(
        f"sequence: {len(sequence)} posed frames, convention confirmed by check_convention_iclnuim.py"
    )
    gt_poses = sequence.ground_truth_poses()

    working_sequence: PosedSequence
    if args.no_degrade:
        print(
            "--no-degrade: fusing native 640x480 depth (bonus diagnostic, not the doc's ask)"
        )
        working_sequence = sequence
    else:
        print(
            f"\nsimulating our sensor over {len(sequence)} frames (256x192, range-dependent noise, synthetic confidence)..."
        )
        working_sequence = DegradedSequence(sequence, seed=args.seed)

    config = FusionConfig(voxel_length=args.voxel)
    print(
        f"\nfusion config: voxel {args.voxel * 1000:.0f}mm, bilateral={config.bilateral}, "
        f"drop_flying_pixels={config.drop_flying_pixels}, min_confidence={config.min_confidence}"
    )

    print("\n--- run A: ground-truth poses ---")
    mesh_gt_poses, report_gt = fuse(working_sequence, config, progress_every=200)
    print(
        f"integrated {report_gt.frames_integrated} frames in {report_gt.seconds:.1f}s"
    )

    print(
        "\n--- run B: poses perturbed by "
        f"{args.translation_std * 1000:.1f}mm (matching real COLMAP-vs-ARKit disagreement) ---"
    )
    perturbed = perturb_poses(
        gt_poses, translation_std=args.translation_std, seed=args.seed
    )
    sequence.use_perturbed_poses(perturbed)
    mesh_perturbed_poses, report_perturbed = fuse(
        working_sequence, config, progress_every=200
    )
    print(
        f"integrated {report_perturbed.frames_integrated} frames in {report_perturbed.seconds:.1f}s"
    )
    sequence.use_perturbed_poses({})  # revert, in case this sequence is reused

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_mesh_path = save_mesh(
        mesh_gt_poses,
        out_dir / f"pose_ceiling_gt_poses_v{int(args.voxel * 1000)}mm.ply",
    )
    perturbed_mesh_path = save_mesh(
        mesh_perturbed_poses,
        out_dir / f"pose_ceiling_perturbed_poses_v{int(args.voxel * 1000)}mm.ply",
    )
    print(f"\nwrote {gt_mesh_path}")
    print(f"wrote {perturbed_mesh_path}")

    gt_tree, gt_points = load_ground_truth_tree(
        sequence.ground_truth_cloud_path, args.gt_voxel
    )

    print(
        f"\nscoring against ground truth (correspondences beyond {args.max_distance}m excluded)"
    )
    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))
    stats_gt = report("GT poses", mesh_gt_poses, gt_tree, gt_points, args.max_distance)
    stats_perturbed = report(
        "perturbed poses", mesh_perturbed_poses, gt_tree, gt_points, args.max_distance
    )

    print("\nacc median/mean = reconstruction -> ground truth (mm, smaller is better)")
    print(
        "<5cm cov        = share of ground-truth points with reconstructed surface within 5cm"
    )

    gt_err = stats_gt["accuracy_median_mm"]
    perturbed_err = stats_perturbed["accuracy_median_mm"]
    print(
        f"\nverdict: GT-pose median {gt_err:.1f}mm vs perturbed-pose median {perturbed_err:.1f}mm "
        f"({perturbed_err / gt_err if gt_err else float('nan'):.2f}x)"
    )
    if gt_err < perturbed_err * 0.7:
        print(
            "  GT poses come out clearly cleaner -> pose accuracy is the ceiling. "
            "Phase 2 (pose refinement) is the lever; finer voxels would help once poses improve."
        )
    elif gt_err > perturbed_err * 0.85:
        print(
            "  GT poses do NOT come out meaningfully cleaner -> depth noise is the limit. "
            "Phase 2 is near its useful end; temporal averaging across frames is the lever instead."
        )
    else:
        print(
            "  Ambiguous: neither run dominates clearly. Re-run with more frames or a tighter "
            "--max-distance before trusting this as decisive."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
