"""Low-level COLMAP helpers shared by every script that drives it as a subprocess.

Moved out of `scripts/run_colmap_raw.py` once a second script
(`scripts/run_colmap_iclnuim.py`) needed the same COLMAP-invocation and
text-model plumbing: running the binary inside its own conda env, converting
between COLMAP's (qw,qx,qy,qz) quaternion order and rotation matrices,
reading its sqlite feature database, reading/writing its text model format,
and recovering camera centres from a world-to-camera pose. None of this is
specific to any one capture source.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skimage.transform import EuclideanTransform, SimilarityTransform


def run_colmap(colmap_env: Path, command: str, **options) -> None:
    """Invoke one COLMAP command inside its conda environment.

    COLMAP lives in its own environment so its Qt/CUDA stack stays away from
    Open3D's; `conda run` gets the DLL search path set up correctly, which
    calling the .exe directly does not reliably do on Windows.
    """
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
        if value is None:
            continue
        argv += [f"--{key.replace('__', '.')}", str(value)]

    print(
        f"\n$ colmap {command} " + " ".join(argv[len(argv) - 2 * len(options) :]),
        flush=True,
    )
    started = time.perf_counter()
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise SystemExit(f"colmap {command} failed with exit code {result.returncode}")
    print(f"  [{command} took {time.perf_counter() - started:.1f}s]", flush=True)


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to COLMAP's (qw, qx, qy, qz) order."""
    return Rotation.from_matrix(rotation).as_quat(scalar_first=True)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion in COLMAP's (qw, qx, qy, qz) order to a 3x3 rotation."""
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def read_database(database_path: Path) -> list[tuple[int, str, int]]:
    """Return (image_id, name, camera_id) rows COLMAP assigned during extraction.

    The model files must use COLMAP's own image ids, not the order frames happen
    to sit in on disk, or triangulation silently associates the wrong features.
    """
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()


def read_model_images(model_dir: Path) -> dict[str, np.ndarray]:
    """Read a COLMAP text model's images.txt into name -> world_to_camera 4x4."""
    poses: dict[str, np.ndarray] = {}

    # Each image occupies two lines: the pose, then its 2D observations. That
    # second line is empty in a seed model, so blank lines cannot simply be
    # skipped -- doing so desyncs the alternation and silently drops every other
    # pose. Identifying pose lines by shape instead is robust to either case.
    for line in (model_dir / "images.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 10 or not tokens[0].isdigit():
            continue

        try:
            q = np.array([float(t) for t in tokens[1:5]])
            t = np.array([float(t) for t in tokens[5:8]])
        except ValueError:
            continue

        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = quaternion_to_rotation(q)
        world_to_camera[:3, 3] = t
        poses[tokens[9]] = world_to_camera

    return poses


def camera_centre(world_to_camera: np.ndarray) -> np.ndarray:
    return -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]


def similarity_transform(
    source: np.ndarray, target: np.ndarray, with_scale: bool = True
) -> np.ndarray:
    """4x4 similarity mapping source points onto target.

    scikit-image's SimilarityTransform is Umeyama (1991) internally and handles
    the reflection case, so it is used rather than a local implementation.
    """
    if with_scale:
        transform = SimilarityTransform.from_estimate(source, target)
    else:
        # Scale locked at 1 keeps the result metric, which matters because the
        # LiDAR fixes real-world scale and COLMAP has no business changing it.
        transform = EuclideanTransform.from_estimate(source, target)

    if not transform:
        raise RuntimeError(
            f"could not estimate a transform from {len(source)} correspondences"
        )
    return np.asarray(transform.params)


def resolve_model_dir(path: Path) -> Path:
    """Return the directory actually holding a model.

    The incremental mappers write their reconstructions into numbered
    subdirectories (`0/`, `1/`, ... one per disconnected component) rather than
    into the output path itself. Reading the parent silently picks up whatever
    older model happens to sit there -- which once produced a full set of
    plausible numbers that were really the previous run's.
    """
    # Sub-models are checked first, deliberately. A previous run's converted
    # .txt files can be left sitting in the parent, and preferring the parent
    # would silently read those stale poses instead of the new reconstruction.
    candidates = sorted(
        (
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "images.bin").is_file()
        ),
        key=lambda child: (
            (child / "points3D.bin").stat().st_size
            if (child / "points3D.bin").is_file()
            else 0
        ),
        reverse=True,
    )
    if not candidates:
        if (path / "images.bin").is_file() or (path / "images.txt").is_file():
            return path
        raise SystemExit(f"no model found in {path} or its subdirectories")
    if len(candidates) > 1:
        print(
            f"note: {len(candidates)} sub-models in {path.name}, using the largest ({candidates[0].name})"
        )
    return candidates[0]
