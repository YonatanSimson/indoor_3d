"""Standalone loader for ARKitScenes-format capture sequences.

Deliberately independent of the ARKit3d / ARKitScenes repos: everything needed
to read frames, intrinsics and poses is reimplemented here so this pipeline has
no cross-project imports.

Directory layout expected (ARKitScenes "lowres" raw format):

    <root>/<video_id>/
        <video_id>_frames/
            lowres_wide/<video_id>_<ts>.png            RGB
            lowres_depth/<video_id>_<ts>.png           depth, uint16 millimetres
            confidence/<video_id>_<ts>.png             optional, 0/1/2 per pixel
            lowres_wide_intrinsics/<video_id>_<ts>.pincam
            lowres_wide.traj                           one pose per line
        <video_id>_3dod_mesh.ply                       optional ARKit mesh
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

# Camera-convention change of basis. Right-Up-Back (ARKit / OpenGL style, camera
# looks down -Z) <-> Right-Down-Forward (OpenCV style, camera looks down +Z).
# Negating Y and Z converts either direction, so the matrix is its own inverse.
RUB_TO_RDF = np.diag([1.0, -1.0, -1.0, 1.0])


class PosedSequence(Protocol):
    """The interface `indoor3d.fusion.fuse` needs from a capture sequence.

    `Sequence` (this module), `RawRecordingSequence` and `ICLNUIMSequence`
    each implement this independently -- their per-frame record types differ
    (`Frame`, `RawFrame`, `ICLNUIMFrame`), so `frame` is typed loosely here
    rather than forcing a shared base class on three otherwise-unrelated
    loaders.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...

    @property
    def has_confidence(self) -> bool: ...

    def load_rgb(self, frame: Any) -> np.ndarray: ...
    def load_depth(self, frame: Any) -> np.ndarray: ...
    def load_confidence(self, frame: Any) -> np.ndarray | None: ...
    def intrinsics(self, frame: Any) -> tuple[np.ndarray, int, int]: ...
    def world_to_camera(self, frame: Any, traj_convention: str = ...) -> np.ndarray: ...


@dataclass(frozen=True)
class Frame:
    """One capture frame. Paths are resolved but nothing is read until asked."""

    timestamp: float
    key: str  # timestamp formatted to millisecond precision, e.g. "223607.897"
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path
    confidence_path: Path | None


def _parse_pose_line(line: str) -> tuple[float, np.ndarray]:
    """Parse one .traj line into (timestamp, world_to_camera 4x4).

    Columns are: timestamp, rotation as an axis-angle vector in radians, then
    translation in metres. Together these give the world-to-camera transform;
    the camera-to-world pose is its inverse.
    """
    tokens = line.split()
    if len(tokens) != 7:
        raise ValueError(
            f"expected 7 columns in trajectory line, got {len(tokens)}: {line!r}"
        )

    timestamp = float(tokens[0])
    axis_angle = np.asarray([float(t) for t in tokens[1:4]], dtype=np.float64)
    translation = np.asarray([float(t) for t in tokens[4:7]], dtype=np.float64)

    rotation, _ = cv2.Rodrigues(axis_angle)

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return timestamp, world_to_camera


class Sequence:
    """A capture sequence, indexable by frame number.

    Frames are ordered by timestamp and are only kept if a pose exists within
    `pose_tolerance` seconds -- the trajectory is sampled independently of the
    image stream, so a handful of frames at the edges typically have no pose.
    """

    def __init__(self, root: Path | str, video_id: str, pose_tolerance: float = 0.05):
        self.root = Path(root)
        self.video_id = video_id
        self.frames_dir = self.root / video_id / f"{video_id}_frames"
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"no frames directory at {self.frames_dir}")

        self.pose_tolerance = pose_tolerance
        self._poses = self._load_trajectory()
        self._pose_times = sorted(self._poses)
        self.frames = self._index_frames()

    # -- construction helpers ------------------------------------------------

    def _load_trajectory(self) -> dict[float, np.ndarray]:
        traj_path = self.frames_dir / "lowres_wide.traj"
        if not traj_path.is_file():
            raise FileNotFoundError(f"no trajectory file at {traj_path}")

        poses: dict[float, np.ndarray] = {}
        for line in traj_path.read_text().splitlines():
            if not line.strip():
                continue
            timestamp, world_to_camera = _parse_pose_line(line)
            poses[timestamp] = world_to_camera
        if not poses:
            raise ValueError(f"trajectory file {traj_path} contained no poses")
        return poses

    def _index_frames(self) -> list[Frame]:
        rgb_dir = self.frames_dir / "lowres_wide"
        depth_dir = self.frames_dir / "lowres_depth"
        intrinsics_dir = self.frames_dir / "lowres_wide_intrinsics"
        confidence_dir = self.frames_dir / "confidence"

        frames: list[Frame] = []
        for rgb_path in sorted(rgb_dir.glob(f"{self.video_id}_*.png")):
            key = rgb_path.stem.split("_", 1)[1]

            depth_path = depth_dir / rgb_path.name
            intrinsics_path = intrinsics_dir / f"{rgb_path.stem}.pincam"
            if not depth_path.is_file() or not intrinsics_path.is_file():
                continue
            if self._nearest_pose_time(float(key)) is None:
                continue

            confidence_path = confidence_dir / rgb_path.name
            frames.append(
                Frame(
                    timestamp=float(key),
                    key=key,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    intrinsics_path=intrinsics_path,
                    confidence_path=(
                        confidence_path if confidence_path.is_file() else None
                    ),
                )
            )
        if not frames:
            raise ValueError(f"no usable frames found under {self.frames_dir}")
        return frames

    def _nearest_pose_time(self, timestamp: float) -> float | None:
        """Closest trajectory timestamp, or None if none lands within tolerance."""
        times = self._pose_times
        position = bisect.bisect_left(times, timestamp)
        best: float | None = None
        best_delta = self.pose_tolerance
        for candidate in times[max(0, position - 1) : position + 1]:
            delta = abs(candidate - timestamp)
            if delta <= best_delta:
                best, best_delta = candidate, delta
        return best

    # -- accessors -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Frame:
        return self.frames[index]

    @property
    def has_confidence(self) -> bool:
        return any(frame.confidence_path is not None for frame in self.frames)

    @property
    def arkit_mesh_path(self) -> Path | None:
        """The on-device ARKit mesh, the baseline this pipeline aims to beat."""
        path = self.root / self.video_id / f"{self.video_id}_3dod_mesh.ply"
        return path if path.is_file() else None

    def load_rgb(self, frame: Frame) -> np.ndarray:
        """RGB image as uint8 (H, W, 3)."""
        image = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read RGB image {frame.rgb_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_depth(self, frame: Frame) -> np.ndarray:
        """Depth as float32 metres, with 0 marking "no measurement"."""
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"could not read depth image {frame.depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32) / 1000.0

    def load_confidence(self, frame: Frame) -> np.ndarray | None:
        """Per-pixel confidence in {0, 1, 2}, or None when not captured."""
        if frame.confidence_path is None:
            return None
        confidence = cv2.imread(str(frame.confidence_path), cv2.IMREAD_UNCHANGED)
        if confidence is None:
            return None
        if confidence.ndim == 3:
            confidence = confidence[:, :, 0]
        return confidence.astype(np.uint8)

    def intrinsics(self, frame: Frame) -> tuple[np.ndarray, int, int]:
        """Return (3x3 intrinsics, width, height) for this frame.

        Intrinsics are stored per frame because ARKit keeps refining the camera
        model during a session.
        """
        width, height, fx, fy, cx, cy = np.loadtxt(frame.intrinsics_path)
        matrix = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        return matrix, int(round(width)), int(round(height))

    def world_to_camera(self, frame: Frame, traj_convention: str = "rdf") -> np.ndarray:
        """Extrinsic mapping world points into an RDF camera frame.

        The result is always Right-Down-Forward, because that is what pinhole
        back-projection and Open3D both assume. `traj_convention` declares what
        the stored trajectory itself uses:

        - "rdf": poses are already OpenCV-style; use them unchanged.
        - "rub": poses are ARKit/OpenGL-style (X right, Y up, Z back), so a
          change of basis is applied to bring them into RDF.

        Which one is correct is a property of the dataset, not a preference --
        `scripts/check_convention.py` settles it against a reference mesh.
        """
        pose_time = self._nearest_pose_time(frame.timestamp)
        if pose_time is None:
            raise KeyError(f"no pose within tolerance for frame at {frame.timestamp}")
        world_to_camera = self._poses[pose_time]

        if traj_convention == "rdf":
            return world_to_camera
        if traj_convention == "rub":
            return RUB_TO_RDF @ world_to_camera
        raise ValueError(
            f"unknown traj_convention {traj_convention!r}, expected 'rdf' or 'rub'"
        )

    def camera_to_world(self, frame: Frame, traj_convention: str = "rdf") -> np.ndarray:
        return np.linalg.inv(self.world_to_camera(frame, traj_convention))
