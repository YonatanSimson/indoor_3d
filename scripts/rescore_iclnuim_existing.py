"""Re-score already-fused ICL-NUIM meshes against ground truth, without re-fusing.

Written to recover the verdicts from `eval_iclnuim_pose_ceiling.py` and
`eval_iclnuim_voxel_sweep.py` runs whose terminal output wasn't captured
anywhere -- the meshes they produced are still on disk, so re-running the
(expensive) fusion is unnecessary; only `iclnuim_eval.report()`'s scoring
against the (already-cached) ground-truth tree needs to happen again.

Also reports whatever `eval_iclnuim_ablation.py` combinations exist, even
though that run didn't finish all 8 depth x pose combinations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_iclnuim import ICLNUIMSequence
from indoor3d.iclnuim_eval import REPORT_HEADER, load_ground_truth_tree, report

ROOT = Path(r"C:\Users\user\data\iclnuim")
OUT_DIR = Path(r"C:\Users\user\data\iclnuim_output")
MAX_DISTANCE = 0.3
GT_VOXEL = 0.005


def score_all(paths: dict[str, Path], gt_tree, gt_points) -> dict[str, dict[str, float]]:
    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))
    results: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        if not path.is_file():
            print(f"{name:>18} | MISSING ({path.name})")
            continue
        mesh = o3d.io.read_triangle_mesh(str(path))
        results[name] = report(name, mesh, gt_tree, gt_points, MAX_DISTANCE)
    return results


def main() -> int:
    sequence = ICLNUIMSequence(ROOT)
    gt_tree, gt_points = load_ground_truth_tree(sequence.ground_truth_cloud_path, GT_VOXEL)

    print("\n=== pose ceiling (decisive Phase V experiment) ===")
    pose_ceiling = score_all(
        {
            "GT poses": OUT_DIR / "pose_ceiling_gt_poses_v5mm.ply",
            "perturbed poses": OUT_DIR / "pose_ceiling_perturbed_poses_v5mm.ply",
        },
        gt_tree,
        gt_points,
    )
    if "GT poses" in pose_ceiling and "perturbed poses" in pose_ceiling:
        gt_err = pose_ceiling["GT poses"]["accuracy_median_mm"]
        perturbed_err = pose_ceiling["perturbed poses"]["accuracy_median_mm"]
        ratio = perturbed_err / gt_err if gt_err else float("nan")
        print(f"\nverdict: GT-pose median {gt_err:.1f}mm vs perturbed-pose median {perturbed_err:.1f}mm ({ratio:.2f}x)")
        if gt_err < perturbed_err * 0.7:
            print("  GT poses come out clearly cleaner -> pose accuracy is the ceiling. "
                  "Phase 2 (pose refinement) is the lever; finer voxels would help once poses improve.")
        elif gt_err > perturbed_err * 0.85:
            print("  GT poses do NOT come out meaningfully cleaner -> depth noise is the limit. "
                  "Phase 2 is near its useful end; temporal averaging across frames is the lever instead.")
        else:
            print("  Ambiguous: neither run dominates clearly.")

    print("\n=== ablation (partial: only what actually ran) ===")
    score_all(
        {
            "full+gt": OUT_DIR / "ablation" / "full_gt_v5mm.ply",
            "degraded+gt": OUT_DIR / "ablation" / "degraded_gt_v5mm.ply",
            "upsampled+gt": OUT_DIR / "ablation" / "upsampled_gt_v5mm.ply",
            "da3_anchored+gt": OUT_DIR / "ablation" / "da3_anchored_gt_v5mm.ply",
            "full+perturbed": OUT_DIR / "ablation" / "full_perturbed_v5mm.ply",
            "degraded+perturbed": OUT_DIR / "ablation" / "degraded_perturbed_v5mm.ply",
            "upsampled+perturbed": OUT_DIR / "ablation" / "upsampled_perturbed_v5mm.ply",
            "da3_anchored+perturbed": OUT_DIR / "ablation" / "da3_anchored_perturbed_v5mm.ply",
        },
        gt_tree,
        gt_points,
    )
    print("published SOTA on this exact sequence (lr kt2), from arXiv:1711.05166 Table V:")
    print("  BundleFusion 7mm | ElasticFusion 8mm | Kintinuous 9mm | RKD-SLAM 16-18mm")

    print("\n=== voxel sweep ===")
    voxel_results = score_all(
        {
            "15mm": OUT_DIR / "voxel_sweep_v15mm.ply",
            "10mm": OUT_DIR / "voxel_sweep_v10mm.ply",
            "5mm": OUT_DIR / "voxel_sweep_v5mm.ply",
        },
        gt_tree,
        gt_points,
    )
    if voxel_results:
        by_truth = sorted(voxel_results, key=lambda k: voxel_results[k]["accuracy_median_mm"])
        print(f"\nranked by ground truth: {by_truth}")

    print("\n=== dense MVS ablation ===")
    print("  did not run (no ~/data/iclnuim/colmap_dense/ output exists)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
