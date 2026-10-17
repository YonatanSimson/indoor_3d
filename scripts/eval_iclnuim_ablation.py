"""Ablation study: where does our accuracy gap vs. SOTA actually come from?

`hand_off_original_plan.md`'s Stage B question 4 compared one configuration
(degraded 256x192 depth, GT poses, 10mm voxel: 9.8mm) against published
ICL-NUIM `kt2` numbers (BundleFusion 7mm, ElasticFusion 8mm). That single
point can't say *why* we're behind -- depth resolution, pose accuracy, or
the fusion algorithm itself could each explain it. This isolates each axis:

- **depth**: `full` (native 640x480 clean depth, no degradation -- the same
  resolution SOTA methods get) vs `degraded` (256x192 ARKit-simulated, what
  a real capture actually gives us) vs `upsampled` (degraded then restored
  with `--upsample anchored`, the production configuration) vs
  `da3_anchored` (degraded, then restored using a cached Depth Anything 3
  monocular prediction as the upsample structure prior instead of the
  bilateral solver's RGB-edge fill -- see `scripts/precompute_da3_depth.py`,
  which must be run first).
- **poses**: `gt` (ground truth) vs `perturbed` (~7.5mm noise, matching real
  COLMAP-vs-ARKit disagreement).

`full` + `gt` is the ceiling question: with perfect information on both
axes, does our fusion algorithm alone get close to SOTA? Everything else
measures what each realistic degradation costs against that ceiling.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.da3_cache import CachedDepthProvider
from indoor3d.data import PosedSequence
from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.fusion import FusionConfig, fuse, save_mesh
from indoor3d.iclnuim_eval import REPORT_HEADER, load_ground_truth_tree, report
from indoor3d.pose_perturbation import perturb_poses
from indoor3d.sensor_sim import DegradedSequence

DEPTH_MODES = ("full", "degraded", "upsampled", "da3_anchored")
POSE_MODES = ("gt", "perturbed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=r"C:\Users\user\data\iclnuim")
    parser.add_argument(
        "--out-dir", default=r"C:\Users\user\data\iclnuim_output\ablation"
    )
    parser.add_argument(
        "--voxel", type=float, default=0.005, help="fusion voxel, metres"
    )
    parser.add_argument("--upsample-scale", type=float, default=3.0)
    parser.add_argument(
        "--da3-cache-dir",
        default=None,
        help="default: <root>/da3_depth_cache (see scripts/precompute_da3_depth.py)",
    )
    parser.add_argument("--gt-voxel", type=float, default=0.005)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--translation-std", type=float, default=0.0075)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depth-modes", nargs="+", default=list(DEPTH_MODES), choices=DEPTH_MODES
    )
    parser.add_argument(
        "--pose-modes", nargs="+", default=list(POSE_MODES), choices=POSE_MODES
    )
    args = parser.parse_args()

    sequence = ICLNUIMSequence(args.root)
    print(f"sequence: {len(sequence)} posed frames")
    gt_poses = sequence.ground_truth_poses()
    perturbed_poses = perturb_poses(
        gt_poses, translation_std=args.translation_std, seed=args.seed
    )

    # Loaded before DegradedSequence, deliberately: reading and voxel-
    # downsampling the ~10M-point ground-truth cloud needs real headroom, and
    # under this machine's memory pressure, running it *after*
    # DegradedSequence has already parked ~880 frames of depth/confidence
    # arrays in memory turned a ~30s call into a 15+ minute stall (reproduced
    # twice). Doing the one-off heavy load first, while memory is freest,
    # sidesteps it without needing to fix the underlying memory pressure.
    gt_tree, gt_points = load_ground_truth_tree(
        sequence.ground_truth_cloud_path, args.gt_voxel
    )

    print(
        f"\nsimulating our sensor over {len(sequence)} frames (for degraded/upsampled depth modes)..."
    )
    degraded_sequence = DegradedSequence(sequence, seed=args.seed, progress_every=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nvoxel {args.voxel * 1000:.0f}mm, scoring against ground truth "
        f"(correspondences beyond {args.max_distance}m excluded)"
    )
    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))

    results: dict[tuple[str, str], dict[str, float]] = {}
    for depth_mode, pose_mode in itertools.product(args.depth_modes, args.pose_modes):
        if pose_mode == "gt":
            sequence.use_perturbed_poses({})
        else:
            sequence.use_perturbed_poses(perturbed_poses)

        working_sequence: PosedSequence
        config = FusionConfig(voxel_length=args.voxel)
        if depth_mode == "full":
            working_sequence = sequence
        elif depth_mode == "degraded":
            working_sequence = degraded_sequence
        elif depth_mode == "upsampled":
            working_sequence = degraded_sequence
            config = FusionConfig(
                voxel_length=args.voxel,
                upsample_method="anchored",
                upsample_scale=args.upsample_scale,
            )
        else:  # da3_anchored
            working_sequence = degraded_sequence
            da3_cache_dir = args.da3_cache_dir or (Path(args.root) / "da3_depth_cache")
            config = FusionConfig(
                voxel_length=args.voxel,
                upsample_method="external",
                upsample_scale=args.upsample_scale,
                depth_prior_provider=CachedDepthProvider(da3_cache_dir),
            )

        mesh, _ = fuse(working_sequence, config, progress_every=0)
        name = f"{depth_mode}_{pose_mode}"
        save_mesh(mesh, out_dir / f"{name}_v{int(args.voxel * 1000)}mm.ply")
        stats = report(name, mesh, gt_tree, gt_points, args.max_distance)
        results[(depth_mode, pose_mode)] = stats

    sequence.use_perturbed_poses({})  # revert, in case this sequence is reused

    print("\nacc median/mean = reconstruction -> ground truth (mm, smaller is better)")
    print(
        "published SOTA on this exact sequence (lr kt2), from arXiv:1711.05166 Table V:"
    )
    print("  BundleFusion 7mm | ElasticFusion 8mm | Kintinuous 9mm | RKD-SLAM 16-18mm")

    if ("full", "gt") in results:
        ceiling = results[("full", "gt")]["accuracy_median_mm"]
        print(f"\nceiling (full depth + GT poses): {ceiling:.1f}mm")
        for (depth_mode, pose_mode), stats in results.items():
            if (depth_mode, pose_mode) == ("full", "gt"):
                continue
            gap = stats["accuracy_median_mm"] - ceiling
            print(
                f"  {depth_mode:>10}/{pose_mode:<10} costs {gap:+.1f}mm vs. the ceiling"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
