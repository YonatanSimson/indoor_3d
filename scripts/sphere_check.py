"""Find a ball-shaped object in a mesh and measure how round it came out.

A Pilates ball is a useful probe: its true shape is known exactly, so the
reconstruction error can be read off directly instead of eyeballed. RANSAC finds
the sphere that best explains a chunk of the mesh, then the residual spread of
its inliers says how lumpy that surface actually is.

    python scripts/sphere_check.py output/myroom_v15mm.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

rng = np.random.default_rng(0)


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere through points, returning (centre, radius).

    Uses the linear form: |p|^2 = 2 p.c + (r^2 - |c|^2), which is linear in the
    unknowns and so needs no initial guess.
    """
    a = np.hstack([2.0 * points, np.ones((len(points), 1))])
    b = (points**2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = solution[:3]
    radius = float(np.sqrt(max(solution[3] + centre @ centre, 0.0)))
    return centre, radius


def ransac_sphere(
    points: np.ndarray,
    min_radius: float,
    max_radius: float,
    tolerance: float,
    iterations: int,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Best sphere with a radius in the given range, as (centre, radius, inliers)."""
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    best_count = 0

    # Four points drawn from a whole room are essentially never all on the same
    # object, so each iteration works inside a neighbourhood instead: pick a
    # seed, then sample only from points close enough to share a sphere with it.
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    neighbourhood = max_radius * 2.2

    for _ in range(iterations):
        seed = points[rng.integers(len(points))]
        count, indices, _ = tree.search_radius_vector_3d(seed, neighbourhood)
        if count < 8:
            continue
        local = points[np.asarray(indices)]

        sample = local[rng.choice(len(local), 4, replace=False)]
        try:
            centre, radius = fit_sphere(sample)
        except np.linalg.LinAlgError:
            continue
        if not (min_radius <= radius <= max_radius):
            continue

        residual = np.abs(np.linalg.norm(points - centre, axis=1) - radius)
        inliers = residual < tolerance
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best = (centre, radius, inliers)

    if best is None:
        return None

    # Refit on the full inlier set; the 4-point estimate is only a seed.
    centre, radius, inliers = best
    centre, radius = fit_sphere(points[inliers])
    residual = np.abs(np.linalg.norm(points - centre, axis=1) - radius)
    return centre, radius, residual < tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--min-radius", type=float, default=0.20)
    parser.add_argument("--max-radius", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=0.05, help="inlier band, metres")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument(
        "--centre",
        type=float,
        nargs=3,
        default=None,
        help="skip the search and measure around this world point",
    )
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()

    header = (
        f"{'mesh':>28} | {'radius':>7} | {'diam cm':>8} | {'inliers':>8} | "
        f"{'RMS mm':>7} | {'p95 mm':>7}"
    )
    print(header)
    print("-" * len(header))

    for path_string in args.meshes:
        path = Path(path_string)
        mesh = o3d.io.read_triangle_mesh(str(path))
        points = np.asarray(mesh.vertices)
        if len(points) == 0:
            print(f"{path.stem:>28} | empty mesh")
            continue

        if args.centre is not None:
            centre_guess = np.asarray(args.centre, dtype=float)
            nearby = points[np.linalg.norm(points - centre_guess, axis=1) < args.max_radius * 1.6]
            if len(nearby) < 50:
                print(f"{path.stem:>28} | too few points near the given centre")
                continue
            centre, radius = fit_sphere(nearby)
            residual = np.abs(np.linalg.norm(nearby - centre, axis=1) - radius)
            inlier_points = nearby[residual < args.tolerance]
        else:
            sample = points
            if len(sample) > args.max_points:
                sample = sample[rng.choice(len(sample), args.max_points, replace=False)]
            found = ransac_sphere(
                sample, args.min_radius, args.max_radius, args.tolerance, args.iterations
            )
            if found is None:
                print(f"{path.stem:>28} | no sphere found in radius range")
                continue
            centre, radius, inlier_mask = found
            inlier_points = sample[inlier_mask]

        deviation = np.linalg.norm(inlier_points - centre, axis=1) - radius
        rms = float(np.sqrt((deviation**2).mean())) * 1000
        p95 = float(np.percentile(np.abs(deviation), 95)) * 1000

        print(
            f"{path.stem:>28} | {radius:7.3f} | {radius * 200:8.1f} | {len(inlier_points):>8,} | "
            f"{rms:7.1f} | {p95:7.1f}"
        )
        if args.centre is None:
            print(f"{'':>28}   centre {np.round(centre, 3)}")

    print("\nRMS/p95 = deviation from a perfect sphere (smaller = rounder)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
