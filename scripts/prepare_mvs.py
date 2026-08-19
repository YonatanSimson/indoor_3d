"""Phase 4, stages 1-2: choose reference frames and build the dense workspace.

Reference frames and source views play different roles, and conflating them is
the easy mistake here. A *reference* frame is one that gets its own depth map,
which is the entire cost of dense stereo. A *source* view is matched against it,
costs almost nothing extra, and is what supplies parallax. COLMAP picks source
views itself, filtering by triangulation angle and then ranking by shared points.

Thinning frames is a runtime decision, but it cannot be applied to references
alone. Geometric consistency checks a reference's depth against the depth maps
of its *source* views, so every source must itself have been solved as a
reference; listing fewer references while leaving all frames selectable as
sources makes the stereo pass abort looking for depth maps that were never
computed.

So the model is reduced, not just the config. That costs little here: 119 frames
still leave ~96 candidates per reference above 30 degrees, against the 20
actually used, because the kept frames span the whole trajectory instead of
clustering.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
DEFAULT_COLMAP_ENV = r"C:\Users\User\anaconda3\envs\colmap"


def select_references(
    reconstruction: pycolmap.Reconstruction, min_baseline: float
) -> list[str]:
    """Greedily keep frames at least `min_baseline` apart, in capture order."""
    images = sorted(reconstruction.images.values(), key=lambda im: im.name)
    kept: list[str] = []
    last_centre: np.ndarray | None = None

    for image in images:
        centre = np.asarray(image.projection_center())
        if last_centre is None or np.linalg.norm(centre - last_centre) >= min_baseline:
            kept.append(image.name)
            last_centre = centre
    return kept


def run_colmap(colmap_env: Path, command: str, **options) -> None:
    argv = ["conda", "run", "--no-capture-output", "--prefix", str(colmap_env), "colmap.exe", command]
    for key, value in options.items():
        if value is not None:
            argv += [f"--{key.replace('__', '.')}", str(value)]
    print(f"\n$ colmap {command}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise SystemExit(f"colmap {command} failed with exit code {result.returncode}")
    print(f"  [{command} took {time.perf_counter() - started:.1f}s]", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--colmap-env", default=DEFAULT_COLMAP_ENV)
    parser.add_argument("--model", default=None, help="default: <session>/colmap/sparse/metric")
    parser.add_argument("--dense", default=None, help="default: <session>/colmap/dense")
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=0.15,
        help="metres between consecutive reference frames",
    )
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--num-src-images", type=int, default=20)
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="if set, keep only this many reference frames, spread evenly, to "
        "measure runtime before committing to the full set",
    )
    args = parser.parse_args()

    session = Path(args.session)
    model_dir = Path(args.model) if args.model else session / "colmap" / "sparse" / "metric"
    dense_dir = Path(args.dense) if args.dense else session / "colmap" / "dense"
    images_dir = session / "colmap" / "images"

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    print(f"model: {reconstruction.num_images()} images, {len(reconstruction.points3D):,} points")

    references = select_references(reconstruction, args.min_baseline)
    print(f"\nreference selection at {args.min_baseline * 100:.0f} cm: "
          f"{len(references)}/{reconstruction.num_images()} frames kept")

    if args.pilot:
        picks = np.linspace(0, len(references) - 1, min(args.pilot, len(references)), dtype=int)
        references = [references[i] for i in picks]
        print(f"pilot: reduced to {len(references)} references spread across the sequence")

    # Remove non-reference frames from the model. Restricting only the config
    # is not enough: with geometric consistency, a reference is checked against
    # its source views' depth maps, so a source that was never solved as a
    # reference aborts the run partway through.
    keep = set(references)
    dropped = 0
    for image in list(reconstruction.images.values()):
        if image.name not in keep:
            reconstruction.deregister_frame(image.frame_id)
            dropped += 1

    reduced_dir = model_dir.parent / "mvs_refs"
    reduced_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(reduced_dir))
    print(f"reduced model: dropped {dropped} frames, "
          f"{reconstruction.num_reg_images()} remain, "
          f"{len(reconstruction.points3D):,} points survive -> {reduced_dir.name}")

    # Deregistering frames strips their observations from every track, so most
    # points lose the support that made them useful and the covisibility graph
    # collapses -- measured once at 136,558 points down to 2,052, leaving a
    # median of 3 covisible frames per reference. Source selection ranks by
    # shared points, so that starves it: covisibility, not triangulation angle,
    # becomes the binding constraint. Re-triangulating from the database
    # rebuilds points among the frames that remain.
    triangulated_dir = model_dir.parent / "mvs_refs_tri"
    triangulated_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        Path(args.colmap_env),
        "point_triangulator",
        database_path=session / "colmap" / "database.db",
        image_path=images_dir,
        input_path=reduced_dir,
        output_path=triangulated_dir,
    )
    retriangulated = pycolmap.Reconstruction(str(triangulated_dir))
    print(f"re-triangulated: {len(retriangulated.points3D):,} points "
          f"across {retriangulated.num_reg_images()} images")

    run_colmap(
        Path(args.colmap_env),
        "image_undistorter",
        image_path=images_dir,
        input_path=triangulated_dir,
        output_path=dense_dir,
        output_type="COLMAP",
        max_image_size=args.max_image_size,
    )

    # Every image in the reduced model is a reference now, which is exactly
    # what geometric consistency requires. Only the source count needs touching.
    config_path = dense_dir / "stereo" / "patch-match.cfg"
    original = config_path.read_text().splitlines()
    print(f"\npatch-match.cfg lists {len(original[0::2])} references, "
          f"source spec '{original[1] if len(original) > 1 else '?'}'")
    if args.num_src_images != 20:
        shutil.copy2(config_path, config_path.with_suffix(".cfg.orig"))
        with config_path.open("w") as handle:
            for name in original[0::2]:
                handle.write(f"{name}\n__auto__, {args.num_src_images}\n")
        print(f"  rewrote source count to {args.num_src_images}")

    undistorted = list((dense_dir / "images").glob("*"))
    print(f"\nworkspace: {dense_dir}")
    print(f"  undistorted images: {len(undistorted)}")
    print(f"  references to solve: {len(references)}")
    print(f"  sources available:   {reconstruction.num_reg_images()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
