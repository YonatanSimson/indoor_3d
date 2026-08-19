"""Phase 2: refine RawRecorder poses with COLMAP, seeded from ARKit.

ARKit's VIO has no loop closure, so a trajectory drifts over a longer scan and
surfaces revisited from a different direction fuse into double walls. COLMAP can
fix that, but running plain structure-from-motion would throw away what ARKit
already knows and return a reconstruction in an arbitrary gauge and scale.

So this seeds COLMAP with the ARKit poses instead: features and matches come
from the images, the initial poses come from ARKit, and COLMAP triangulates
against them and then bundle-adjusts. The result stays in metric world scale and
keeps ARKit's gravity alignment, with drift taken out.

    python scripts/run_colmap_raw.py                     # full pipeline
    python scripts/run_colmap_raw.py --matcher exhaustive  # real loop closure
    python scripts/run_colmap_raw.py --from triangulate   # resume partway

Outputs land in <session>/colmap/:
    images/                 flat copies/links the COLMAP tools can read
    database.db             features and matches
    sparse/arkit/           ARKit poses as a COLMAP model (the seed)
    sparse/triangulated/    seed poses + triangulated 3D points
    sparse/refined/         after bundle adjustment  <- the useful output
    refined_poses.json      refined world->camera matrices, for fusion
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from skimage.transform import EuclideanTransform, SimilarityTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor3d.data_rawrecording import RawRecordingSequence

DEFAULT_SESSION = r"C:\Users\user\data\iphone\RawRecording_2026-08-18_215941"
DEFAULT_COLMAP_ENV = r"C:\Users\User\anaconda3\envs\colmap"

STAGES = ("images", "features", "match", "seed", "triangulate", "bundle", "export")


def run_colmap(colmap_env: Path, command: str, **options) -> None:
    """Invoke one COLMAP command inside its conda environment.

    COLMAP lives in its own environment so its Qt/CUDA stack stays away from
    Open3D's; `conda run` gets the DLL search path set up correctly, which
    calling the .exe directly does not reliably do on Windows.
    """
    argv = [
        "conda", "run", "--no-capture-output", "--prefix", str(colmap_env),
        "colmap.exe", command,
    ]
    for key, value in options.items():
        if value is None:
            continue
        argv += [f"--{key.replace('__', '.')}", str(value)]

    print(f"\n$ colmap {command} " + " ".join(argv[len(argv) - 2 * len(options) :]), flush=True)
    started = time.perf_counter()
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise SystemExit(f"colmap {command} failed with exit code {result.returncode}")
    print(f"  [{command} took {time.perf_counter() - started:.1f}s]", flush=True)


def prepare_images(sequence: RawRecordingSequence, images_dir: Path) -> dict[str, int]:
    """Flatten frame_XXXX/image.jpg into one directory COLMAP can read.

    Hard links are used where the filesystem allows it -- these are ~0.5 MB
    each and copying 539 of them is pointless duplication.

    Returns a mapping from image filename to sequence index.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, int] = {}
    linked = copied = 0

    for index in range(len(sequence)):
        frame = sequence[index]
        name = f"{frame.folder.name}.jpg"
        mapping[name] = index

        destination = images_dir / name
        if destination.exists():
            continue
        source = frame.folder / "image.jpg"
        try:
            destination.hardlink_to(source)
            linked += 1
        except (OSError, NotImplementedError):
            shutil.copy2(source, destination)
            copied += 1

    print(f"prepared {len(mapping)} images ({linked} linked, {copied} copied) in {images_dir}")
    return mapping


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to COLMAP's (qw, qx, qy, qz) order."""
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def read_database(database_path: Path) -> list[tuple[int, str, int]]:
    """Return (image_id, name, camera_id) rows COLMAP assigned during extraction.

    The model files must use COLMAP's own image ids, not the order frames happen
    to sit in on disk, or triangulation silently associates the wrong features.
    """
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()


def write_pose_priors(
    sequence: RawRecordingSequence,
    rows: list[tuple[int, str, int]],
    name_to_index: dict[str, int],
    database_path: Path,
    std: float,
    convention: str,
) -> int:
    """Store ARKit camera positions in the database as position priors.

    This constrains the gauge *during* optimisation rather than un-drifting it
    afterwards: with absolute positions pulled toward ARKit's, translation,
    rotation and scale are all pinned, so the reconstruction stays metric
    instead of floating off by the ~3x factor plain bundle adjustment allows.

    `std` sets how firmly. Too tight and COLMAP cannot correct the very drift
    this phase exists to remove; too loose and the prior does nothing. ARKit is
    locally accurate but drifts slowly, so a value well above its local noise
    and well below the expected drift is about right.

    COLMAP creates these rows during feature extraction with NaN positions;
    they are filled in here.
    """
    covariance = np.diag([std**2, std**2, std**2]).astype(np.float64)
    written = 0

    with sqlite3.connect(database_path) as connection:
        for image_id, name, _camera_id in rows:
            index = name_to_index.get(name)
            if index is None:
                continue
            position = sequence.camera_to_world(sequence[index], convention)[:3, 3]
            connection.execute(
                "UPDATE pose_priors SET position = ?, position_covariance = ?, "
                "coordinate_system = 1 WHERE corr_data_id = ?",
                (
                    np.ascontiguousarray(position, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(covariance).tobytes(),
                    image_id,
                ),
            )
            written += connection.total_changes > 0 and 1 or 0

        remaining = connection.execute(
            "SELECT COUNT(*) FROM pose_priors WHERE coordinate_system != 1"
        ).fetchone()[0]

    # coordinate_system 1 is CARTESIAN: a local metric frame, which is what
    # ARKit's world is. The default -1 means undefined and is ignored.
    print(f"wrote {written} pose priors (std {std * 100:.0f} cm), {remaining} left unset")
    return written


def write_seed_model(
    sequence: RawRecordingSequence,
    rows: list[tuple[int, str, int]],
    name_to_index: dict[str, int],
    model_dir: Path,
    convention: str,
) -> None:
    """Write ARKit poses as a COLMAP text model for point_triangulator to use.

    COLMAP stores world-to-camera as a quaternion plus translation in an RDF
    camera frame -- exactly what the loader already produces.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    cameras: dict[int, tuple[int, int, float, float, float, float]] = {}
    image_lines: list[str] = []

    for image_id, name, camera_id in rows:
        index = name_to_index.get(name)
        if index is None:
            continue
        frame = sequence[index]

        intrinsics, width, height = sequence.intrinsics(frame)
        cameras.setdefault(
            camera_id,
            (width, height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]),
        )

        world_to_camera = sequence.world_to_camera(frame, convention)
        q = rotation_to_quaternion(world_to_camera[:3, :3])
        t = world_to_camera[:3, 3]
        image_lines.append(
            f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
            f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {camera_id} {name}"
        )

    with (model_dir / "cameras.txt").open("w") as handle:
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id, (width, height, fx, fy, cx, cy) in sorted(cameras.items()):
            handle.write(f"{camera_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with (model_dir / "images.txt").open("w") as handle:
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for line in image_lines:
            # Every image needs a second, empty line for its 2D points; the
            # triangulator fills these in.
            handle.write(line + "\n\n")

    (model_dir / "points3D.txt").write_text("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    print(f"wrote seed model: {len(image_lines)} images, {len(cameras)} camera(s) -> {model_dir}")


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


def similarity_transform(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> np.ndarray:
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
        raise RuntimeError(f"could not estimate a transform from {len(source)} correspondences")
    return np.asarray(transform.params)


def align_to_arkit(
    sequence: RawRecordingSequence,
    refined: dict[str, np.ndarray],
    name_to_index: dict[str, int],
    convention: str,
    with_scale: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Put refined poses back into ARKit's world frame and measure the change.

    Bundle adjustment leaves seven degrees of freedom unconstrained -- the whole
    reconstruction can rotate, translate and rescale without changing reprojection
    error at all. Comparing raw refined poses against ARKit therefore measures
    that arbitrary float, not drift. Aligning first removes it, and what remains
    is the actual correction: the part of the trajectory COLMAP genuinely
    disagrees with, which is what Phase 2 is after.
    """
    names = [name for name in sorted(refined) if name in name_to_index]
    if not names:
        return refined, {}

    refined_centres = np.array([camera_centre(refined[name]) for name in names])
    arkit_centres = np.array(
        [
            camera_centre(sequence.world_to_camera(sequence[name_to_index[name]], convention))
            for name in names
        ]
    )

    before = np.linalg.norm(refined_centres - arkit_centres, axis=1)
    similarity = similarity_transform(refined_centres, arkit_centres, with_scale)

    # The linear block is s*R, and R is orthonormal, so its determinant gives
    # s^3 -- a direct read on how far COLMAP drifted off metric scale.
    scale = float(abs(np.linalg.det(similarity[:3, :3])) ** (1.0 / 3.0))

    # Move the world, not the cameras: a pose maps world -> camera, so applying
    # the inverse similarity on the right re-expresses it in ARKit's frame.
    # Multiplying the top three rows by the scale afterwards restores an
    # orthonormal rotation, which keeps the pose metric and, less obviously,
    # keeps `camera_centre` valid -- that formula assumes orthonormality and
    # returns nonsense for the scaled matrix the raw product leaves behind.
    inverse = np.linalg.inv(similarity)
    rescale = np.diag([scale, scale, scale, 1.0])
    aligned = {name: rescale @ (pose @ inverse) for name, pose in refined.items()}

    # Measured by transforming the centres directly, independent of the pose
    # bookkeeping above, so the two act as a cross-check on each other.
    homogeneous = np.hstack([refined_centres, np.ones((len(refined_centres), 1))])
    mapped_centres = (homogeneous @ similarity.T)[:, :3]
    residual = np.linalg.norm(mapped_centres - arkit_centres, axis=1)

    check = np.array([camera_centre(aligned[name]) for name in names])
    if not np.allclose(check, mapped_centres, atol=1e-6):
        drift = float(np.abs(check - mapped_centres).max())
        print(f"  note: aligned poses and mapped centres disagree by up to {drift:.2e} m")

    stats = {
        "scale": scale,
        "gauge_shift_m": float(np.median(before)),
        "median_mm": float(np.median(residual) * 1000),
        "mean_mm": float(residual.mean() * 1000),
        "max_mm": float(residual.max() * 1000),
    }

    print(f"\naligned {len(names)} refined poses back into ARKit's frame")
    print(f"  gauge float removed : {stats['gauge_shift_m']:.2f} m (arbitrary, not error)")
    print(f"  scale factor        : {scale:.5f}", end="")
    if with_scale:
        print(f"  ({abs(scale - 1) * 100:.2f}% off metric)" if abs(scale - 1) > 1e-4 else "")
    else:
        print("  (held at 1, metric preserved)")

    print("\ncorrection vs ARKit after alignment -- this is the real drift:")
    print(f"  median {stats['median_mm']:7.1f} mm")
    print(f"  mean   {stats['mean_mm']:7.1f} mm")
    print(f"  max    {stats['max_mm']:7.1f} mm")

    if stats["median_mm"] > 300:
        print(
            "\n  WARNING: a median correction this large usually means the "
            "reconstruction is wrong, not that ARKit drifted that badly. "
            "Check registered image count and match quality before trusting these poses."
        )
    return aligned, stats


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
        (child for child in path.iterdir() if child.is_dir() and (child / "images.bin").is_file()),
        key=lambda child: (child / "points3D.bin").stat().st_size if (child / "points3D.bin").is_file() else 0,
        reverse=True,
    )
    if not candidates:
        if (path / "images.bin").is_file() or (path / "images.txt").is_file():
            return path
        raise SystemExit(f"no model found in {path} or its subdirectories")
    if len(candidates) > 1:
        print(f"note: {len(candidates)} sub-models in {path.name}, using the largest ({candidates[0].name})")
    return candidates[0]


def finish(
    args,
    session: Path,
    out_dir: Path,
    refined_dir: Path,
    sequence: RawRecordingSequence,
    name_to_index: dict[str, int],
    colmap_env: Path,
    begin: int,
) -> int:
    """Convert, align and export whichever model the chosen path produced."""
    if begin <= STAGES.index("export"):
        # COLMAP writes binary; convert so the poses are readable.
        run_colmap(
            colmap_env,
            "model_converter",
            input_path=refined_dir,
            output_path=refined_dir,
            output_type="TXT",
        )

        refined = read_model_images(refined_dir)

        registered = len(refined)
        if registered < len(sequence):
            print(
                f"WARNING: {len(sequence) - registered} of {len(sequence)} images did not "
                "survive; those frames keep their ARKit poses when fusing"
            )

        aligned, stats = align_to_arkit(
            sequence, refined, name_to_index, args.convention, with_scale=not args.fix_scale
        )

        export = {
            "session": session.name,
            "convention": "rdf",
            "frame": "aligned to ARKit world",
            "mode": args.mode,
            "alignment": stats,
            "poses": {
                name: {"frame": Path(name).stem, "world_to_camera_rdf": pose.tolist()}
                for name, pose in sorted(aligned.items())
            },
        }
        export_path = out_dir / f"refined_poses_{args.refined_name}.json" if args.refined_name != "refined" else out_dir / "refined_poses.json"
        export_path.write_text(json.dumps(export, indent=1))
        print(f"\nwrote {len(aligned)} refined poses -> {export_path}")

    print("\ndone")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--out", default=None, help="default: <session>/colmap")
    parser.add_argument("--colmap-env", default=DEFAULT_COLMAP_ENV)
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=["sequential", "exhaustive", "vocabtree"],
        help="sequential is fast; exhaustive finds loop closures but is O(n^2)",
    )
    parser.add_argument("--vocab-tree", default=None, help="required for --matcher vocabtree")
    parser.add_argument("--overlap", type=int, default=10, help="sequential matcher window")
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="share one camera across all frames; ARKit autofocus varies focal "
        "length ~3%% per session, so per-image cameras are the accurate default",
    )
    parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="let bundle adjustment adjust focal length too; off by default so "
        "ARKit's calibration is trusted and only poses and points move",
    )
    parser.add_argument(
        "--fix-scale",
        action="store_true",
        help="align back to ARKit without rescaling, keeping LiDAR-metric scale "
        "exactly; by default scale is solved for and reported",
    )
    parser.add_argument(
        "--mode",
        default="seed",
        choices=["seed", "priors"],
        help="'seed' fixes ARKit poses, triangulates and bundle-adjusts, leaving "
        "the gauge free to drift; 'priors' anchors ARKit positions inside the "
        "optimisation so the result stays metric on its own",
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=0.10,
        help="how firmly to trust ARKit positions, in metres (mode=priors). Well "
        "above ARKit's local noise and below its accumulated drift",
    )
    parser.add_argument(
        "--refined-name",
        default="refined",
        help="subdirectory of sparse/ to write the result into, so runs with "
        "different settings do not overwrite each other",
    )
    parser.add_argument("--convention", default="rub", choices=["rub", "rdf"])
    parser.add_argument("--from", dest="start", default="images", choices=STAGES)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    session = Path(args.session)
    out_dir = Path(args.out) if args.out else session / "colmap"
    colmap_env = Path(args.colmap_env)
    if not (colmap_env / "Library" / "bin" / "colmap.exe").is_file():
        raise SystemExit(f"no colmap.exe under {colmap_env}")

    sequence = RawRecordingSequence(session, intrinsics_space="rgb")
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    database_path = out_dir / "database.db"
    seed_dir = out_dir / "sparse" / "arkit"
    triangulated_dir = out_dir / "sparse" / "triangulated"
    refined_dir = out_dir / "sparse" / args.refined_name

    print(f"session : {session.name}  ({len(sequence)} frames)")
    print(f"output  : {out_dir}")
    print(f"colmap  : {colmap_env}")
    print(f"matcher : {args.matcher}")

    begin = STAGES.index(args.start)
    use_gpu = 0 if args.no_gpu else 1

    name_to_index = prepare_images(sequence, images_dir)

    if begin <= STAGES.index("features"):
        camera_options = {}
        if args.single_camera:
            # Seed the shared camera with the median intrinsics so the initial
            # guess is close before bundle adjustment touches it.
            samples = np.array(
                [
                    [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
                    for k in (sequence.intrinsics(sequence[i])[0] for i in range(0, len(sequence), 10))
                ]
            )
            fx, fy, cx, cy = np.median(samples, axis=0)
            camera_options = {
                "ImageReader__single_camera": 1,
                "ImageReader__camera_params": f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}",
            }
        else:
            camera_options = {"ImageReader__single_camera_per_image": 1}

        run_colmap(
            colmap_env,
            "feature_extractor",
            database_path=database_path,
            image_path=images_dir,
            ImageReader__camera_model="PINHOLE",
            FeatureExtraction__use_gpu=use_gpu,
            **camera_options,
        )

    if begin <= STAGES.index("match"):
        if args.matcher == "sequential":
            run_colmap(
                colmap_env,
                "sequential_matcher",
                database_path=database_path,
                SequentialMatching__overlap=args.overlap,
                SequentialMatching__quadratic_overlap=1,
                FeatureMatching__use_gpu=use_gpu,
            )
        elif args.matcher == "exhaustive":
            run_colmap(
                colmap_env,
                "exhaustive_matcher",
                database_path=database_path,
                FeatureMatching__use_gpu=use_gpu,
            )
        else:
            if not args.vocab_tree:
                raise SystemExit("--matcher vocabtree needs --vocab-tree <path to .bin>")
            run_colmap(
                colmap_env,
                "sequential_matcher",
                database_path=database_path,
                SequentialMatching__overlap=args.overlap,
                SequentialMatching__loop_detection=1,
                SequentialMatching__vocab_tree_path=args.vocab_tree,
                FeatureMatching__use_gpu=use_gpu,
            )

    if begin <= STAGES.index("seed"):
        rows = read_database(database_path)
        write_seed_model(sequence, rows, name_to_index, seed_dir, args.convention)
        if args.mode == "priors":
            write_pose_priors(
                sequence, rows, name_to_index, database_path, args.prior_std, args.convention
            )

    if args.mode == "priors":
        # The prior mapper does its own incremental reconstruction, anchored by
        # the positions written above, so triangulation and a separate bundle
        # adjustment are not run.
        if begin <= STAGES.index("triangulate"):
            refined_dir.mkdir(parents=True, exist_ok=True)
            run_colmap(
                colmap_env,
                "pose_prior_mapper",
                database_path=database_path,
                image_path=images_dir,
                output_path=refined_dir,
                prior_position_std_x=args.prior_std,
                prior_position_std_y=args.prior_std,
                prior_position_std_z=args.prior_std,
                use_robust_loss_on_prior_position=1,
                Mapper__ba_refine_focal_length=1 if args.refine_intrinsics else 0,
                Mapper__ba_refine_principal_point=0,
                Mapper__ba_refine_extra_params=1 if args.refine_intrinsics else 0,
                Mapper__ba_refine_sensor_from_rig=0,
            )
        return finish(
            args, session, out_dir, resolve_model_dir(refined_dir), sequence,
            name_to_index, colmap_env, begin
        )

    if begin <= STAGES.index("triangulate"):
        triangulated_dir.mkdir(parents=True, exist_ok=True)
        run_colmap(
            colmap_env,
            "point_triangulator",
            database_path=database_path,
            image_path=images_dir,
            input_path=seed_dir,
            output_path=triangulated_dir,
        )

    if begin <= STAGES.index("bundle"):
        refined_dir.mkdir(parents=True, exist_ok=True)
        refine = 1 if args.refine_intrinsics else 0
        run_colmap(
            colmap_env,
            "bundle_adjuster",
            input_path=triangulated_dir,
            output_path=refined_dir,
            BundleAdjustment__refine_focal_length=refine,
            BundleAdjustment__refine_principal_point=0,
            BundleAdjustment__refine_extra_params=refine,
            # COLMAP 4.x splits each pose into rig_from_world . sensor_from_rig.
            # With one camera per image every rig holds a single sensor, so
            # sensor_from_rig duplicates rig_from_world exactly: six free
            # parameters per image that cannot improve the fit and leave the
            # system gauge-degenerate. Left at its default of 1, the
            # reconstruction collapses (measured: 2.8x scale error).
            BundleAdjustment__refine_sensor_from_rig=0,
            BundleAdjustment__refine_rig_from_world=1,
        )

    return finish(
        args, session, out_dir, resolve_model_dir(refined_dir), sequence,
        name_to_index, colmap_env, begin
    )


if __name__ == "__main__":
    raise SystemExit(main())
