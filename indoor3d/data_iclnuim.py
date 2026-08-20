"""Loader for the ICL-NUIM living-room sequence, TUM-formatted release.

Ground truth for Phase V (`hand_off_original_plan.md`): every number this
pipeline has produced elsewhere is scored against the same noisy LiDAR that
built the mesh, which cannot reveal a bias the sensor itself has. ICL-NUIM
ships a laser-accurate surface and exact poses instead.

Directory layout expected (`~/data/iclnuim/`), a TUM RGB-D-style repackaging
of the original ICL-NUIM release:

    associations.txt          "<id> depth/<id>.png <id> rgb/<id>.png" per line
    depth/<id>.png             uint16, TUM scale (value / 5000 = metres)
    rgb/<id>.png                uint8 640x480x3
    livingRoom2n.gt.sim        ICL-NUIM's native ground-truth poses (see below)
    livingRoom2.gt.freiburg    present, but NOT used here -- see below
    living-room.ply            ground-truth point cloud (~10M points, no faces)

**`livingRoom2.gt.freiburg` is not this dataset's world frame.**
`scripts/check_convention_iclnuim.py` back-projects against `living-room.ply`
exactly as `scripts/check_convention.py` did for the ARKitScenes mesh, and
every axes/direction/fy-sign hypothesis for `.gt.freiburg` lands 20-50+ cm
off, even after a full RANSAC+ICP registration search -- consistent with it
being expressed in some other frame (plausibly relative to frame 1, given its
first row is exactly identity) rather than a sign or axis error. Whatever
that other frame is, it is not documented anywhere shipped with this data, so
it is left alone rather than guessed at further.

`livingRoom2n.gt.sim` -- 880 blank-line-separated 3x4 `[R|t]` blocks,
ICL-NUIM's own POV-Ray-native trajectory format -- back-projects to a
**14.7 mm median** against the ground-truth cloud (57% within 5 cm) using it
directly as camera-to-world with *no* change of basis, matching the quality
of the ARKitScenes convention check's confirmed result. That combination is
this loader's default. The one confirmed trap from the hand-off doc still
applies: `fy` is negative (POV-Ray's image convention is flipped relative to
OpenCV's) -- confirmed empirically here too, not just carried over from the
doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import RUB_TO_RDF

# ICL-NUIM's published synthetic camera intrinsics for this capture (640x480).
# fy's sign is exactly the trap this loader's docstring warns about --
# `fy_sign` below is a parameter, not baked into these constants, so the
# convention check can flip it without touching this module.
_FX = 481.20
_FY_MAGNITUDE = 480.00
_CX = 319.50
_CY = 239.50

_DEPTH_SCALE = 5000.0  # TUM convention: stored uint16 / 5000 = metres.


@dataclass(frozen=True)
class ICLNUIMFrame:
    """One capture frame. Paths are resolved but nothing is read until asked."""

    frame_id: int
    rgb_path: Path
    depth_path: Path


def _parse_gt_sim(path: Path) -> dict[int, np.ndarray]:
    """Parse ICL-NUIM's native `.gt.sim` trajectory into id -> 4x4 pose.

    Format: one blank-line-separated block per frame (ids 1-based, matching
    `associations.txt`), each block three lines of a `[R | t]` 3x4
    camera-to-world matrix -- confirmed empirically (see module docstring),
    not by any published spec for this specific TUM-style repackaging.
    """
    blocks = path.read_text().strip().split("\n\n")
    poses: dict[int, np.ndarray] = {}
    for frame_id, block in enumerate(blocks, start=1):
        rows = [line.split() for line in block.strip().splitlines()]
        if len(rows) != 3 or any(len(row) != 4 for row in rows):
            raise ValueError(
                f"expected a 3x4 [R|t] block in {path.name} for frame {frame_id}, got {rows!r}"
            )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :4] = np.asarray(rows, dtype=np.float64)
        poses[frame_id] = pose
    return poses


class ICLNUIMSequence:
    """The ICL-NUIM living-room-traj2 sequence, indexable by frame number.

    Args:
        root: directory holding `associations.txt`, `depth/`, `rgb/`,
            `livingRoom2n.gt.sim` and `living-room.ply`.
        fy_sign: +1.0 or -1.0, applied to the published fy magnitude. Default
            (-1.0) is the value `scripts/check_convention_iclnuim.py`
            confirmed against `living-room.ply`, not an assumption.
        pose_axes: "native" (default; `.gt.sim` rotations back-project
            correctly with no change of basis -- also confirmed by the
            convention check) or "rub_to_rdf", kept only so the check script
            can demonstrate that hypothesis is wrong rather than untested.
        pose_direction: "c2w" (default; confirmed) or "w2c".
    """

    def __init__(
        self,
        root: Path | str = r"C:\Users\user\data\iclnuim",
        fy_sign: float = -1.0,
        pose_axes: str = "native",
        pose_direction: str = "c2w",
    ):
        if fy_sign not in (1.0, -1.0):
            raise ValueError(f"fy_sign must be 1.0 or -1.0, got {fy_sign!r}")
        if pose_axes not in ("native", "rub_to_rdf"):
            raise ValueError(
                f"pose_axes must be 'native' or 'rub_to_rdf', got {pose_axes!r}"
            )
        if pose_direction not in ("c2w", "w2c"):
            raise ValueError(
                f"pose_direction must be 'c2w' or 'w2c', got {pose_direction!r}"
            )

        self.root = Path(root)
        self.fy_sign = fy_sign
        self.pose_axes = pose_axes
        self.pose_direction = pose_direction
        self._overridden: dict[int, np.ndarray] = {}

        associations_path = self.root / "associations.txt"
        if not associations_path.is_file():
            raise FileNotFoundError(f"no associations.txt at {associations_path}")
        gt_path = self.root / "livingRoom2n.gt.sim"
        if not gt_path.is_file():
            raise FileNotFoundError(f"no ground-truth trajectory at {gt_path}")

        self._poses_c2w_rdf = self._orient_poses(_parse_gt_sim(gt_path))
        self.frames = self._index_frames(associations_path)
        if not self.frames:
            raise ValueError(
                f"no frames under {self.root} had a matching pose in {gt_path.name}"
            )

    # -- construction ---------------------------------------------------

    def _orient_poses(self, raw: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Normalise stored poses to camera-to-world in RDF axes."""
        oriented: dict[int, np.ndarray] = {}
        for frame_id, pose in raw.items():
            matrix = np.linalg.inv(pose) if self.pose_direction == "w2c" else pose
            if self.pose_axes == "rub_to_rdf":
                matrix = matrix @ RUB_TO_RDF
            oriented[frame_id] = matrix
        return oriented

    def _index_frames(self, associations_path: Path) -> list[ICLNUIMFrame]:
        frames: list[ICLNUIMFrame] = []
        for line in associations_path.read_text().splitlines():
            if not line.strip():
                continue
            depth_id, depth_rel, _rgb_id, rgb_rel = line.split()
            frame_id = int(depth_id)

            # `livingRoom2n.gt.sim` has no entry for frame 0 -- dropped, not
            # id-shifted, since shifting would silently mismatch every later
            # frame's depth/rgb/pose triple.
            if frame_id not in self._poses_c2w_rdf:
                continue

            frames.append(
                ICLNUIMFrame(
                    frame_id=frame_id,
                    rgb_path=self.root / rgb_rel,
                    depth_path=self.root / depth_rel,
                )
            )
        frames.sort(key=lambda f: f.frame_id)
        return frames

    # -- accessors --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> ICLNUIMFrame:
        return self.frames[index]

    @property
    def has_confidence(self) -> bool:
        """No native confidence map; `indoor3d.sensor_sim` synthesises one."""
        return False

    @property
    def ground_truth_cloud_path(self) -> Path:
        """The laser-accurate surface this dataset exists to provide.

        Ships as a dense point cloud (points + normals, no faces), confirmed
        with `open3d.io.read_triangle_mesh` reporting zero triangles on it --
        named accordingly rather than `..._mesh_path`, so callers reach for
        `o3d.io.read_point_cloud` instead of getting an empty mesh silently.
        """
        path = self.root / "living-room.ply"
        if not path.is_file():
            raise FileNotFoundError(f"no ground-truth point cloud at {path}")
        return path

    def load_rgb(self, frame: ICLNUIMFrame) -> np.ndarray:
        image = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read RGB image {frame.rgb_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_depth(self, frame: ICLNUIMFrame) -> np.ndarray:
        """Depth as float32 metres, with 0 marking "no measurement"."""
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"could not read depth image {frame.depth_path}")
        return (depth.astype(np.float32)) / _DEPTH_SCALE

    def load_confidence(self, frame: ICLNUIMFrame) -> np.ndarray | None:
        return None

    def intrinsics(self, frame: ICLNUIMFrame) -> tuple[np.ndarray, int, int]:
        matrix = np.asarray(
            [
                [_FX, 0.0, _CX],
                [0.0, self.fy_sign * _FY_MAGNITUDE, _CY],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return matrix, 640, 480

    def camera_to_world(
        self, frame: ICLNUIMFrame, traj_convention: str = "rdf"
    ) -> np.ndarray:
        """Ground-truth camera-to-world pose, already normalised to RDF axes.

        `traj_convention` is accepted for interface parity with the other
        loaders but ignored: orientation is resolved once in `__init__` from
        `pose_axes`/`pose_direction`, which is what the convention check
        actually varies.
        """
        override = self._overridden.get(frame.frame_id)
        if override is not None:
            return override
        return self._poses_c2w_rdf[frame.frame_id]

    def world_to_camera(
        self, frame: ICLNUIMFrame, traj_convention: str = "rdf"
    ) -> np.ndarray:
        return np.linalg.inv(self.camera_to_world(frame, traj_convention))

    def use_perturbed_poses(self, poses: dict[int, np.ndarray]) -> None:
        """Override ground-truth poses, e.g. with `pose_perturbation.perturb_poses`.

        Mirrors `RawRecordingSequence.load_refined_poses`: an explicit swap
        rather than a second loader, so `fusion.fuse` needs no special case
        for "which poses this run uses". Pass an empty dict to revert to the
        unperturbed ground truth.
        """
        unknown = poses.keys() - {frame.frame_id for frame in self.frames}
        if unknown:
            raise KeyError(
                f"poses given for frame ids not in this sequence: {sorted(unknown)[:5]}..."
            )
        self._overridden = dict(poses)

    def ground_truth_poses(self) -> dict[int, np.ndarray]:
        """Camera-to-world poses for every frame, ignoring any override.

        The source `perturb_poses` perturbs from -- always the real ground
        truth, even while `use_perturbed_poses` has replaced what
        `world_to_camera`/`camera_to_world` return.
        """
        return dict(self._poses_c2w_rdf)
