"""Loader for RawRecorder sessions captured by the model_3d iOS app.

Exposes the same interface as `data.Sequence`, so the fusion pipeline consumes
either source unchanged.

Layout, one folder per frame:

    RawRecording_<stamp>/frame_0001/
        image.jpg                 RGB, full camera resolution (e.g. 1920x1440)
        depth.bin                 float32 metres, row-major, 256x192
        depth_meta.json           width / height / dtype for the above
        confidence.bin            uint8 ARKit confidence, 0/1/2
        confidence_meta.json
        camera_intrinsics.json    for the *RGB* resolution
        camera_extrinsics.json    ARKit camera transform

Four things about this format differ from the ARKitScenes convention and each
one silently corrupts geometry if missed:

1. Matrices are serialised column-by-column, so reading the JSON as row-major
   yields the transpose. `position` doubles as a fingerprint: it matches row 3
   of the raw array, where a row-major 4x4 would keep translation in column 3.
2. `transform_matrix` is camera-to-world, whereas a `.traj` line is
   world-to-camera -- an inversion apart.
3. ARKit's camera frame is Right-Up-Back; back-projection and Open3D both want
   Right-Down-Forward.
4. Intrinsics describe the RGB frame, but depth is captured at a much smaller
   resolution, so they must be rescaled before touching the depth grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import RUB_TO_RDF


@dataclass(frozen=True)
class RawFrame:
    index: int
    timestamp: float
    folder: Path


class RawRecordingSequence:
    """A RawRecorder session, indexable by frame number.

    Args:
        session: folder containing the `frame_XXXX/` directories.
        intrinsics_space: which image grid the returned intrinsics should
            describe. "depth" (the default) rescales them to the depth
            resolution, which is what fusion back-projects on.
    """

    def __init__(self, session: Path | str, intrinsics_space: str = "depth"):
        self.session = Path(session)
        if intrinsics_space not in {"depth", "rgb"}:
            raise ValueError(
                f"intrinsics_space must be 'depth' or 'rgb', got {intrinsics_space!r}"
            )
        self.intrinsics_space = intrinsics_space
        self._refined: dict[str, np.ndarray] = {}

        # Some exports nest a second folder of the same name; step into it so
        # either shape of path works.
        if not any(self.session.glob("frame_*")):
            nested = self.session / self.session.name
            if any(nested.glob("frame_*")):
                self.session = nested

        self.frames = self._index_frames()
        if not self.frames:
            raise ValueError(f"no complete frames found under {self.session}")

    # -- construction --------------------------------------------------------

    def _index_frames(self) -> list[RawFrame]:
        required = (
            "image.jpg",
            "depth.bin",
            "camera_extrinsics.json",
            "camera_intrinsics.json",
        )
        frames: list[RawFrame] = []
        for index, folder in enumerate(sorted(self.session.glob("frame_*"))):
            if not all((folder / name).is_file() for name in required):
                continue
            extrinsics = json.loads((folder / "camera_extrinsics.json").read_text())
            frames.append(
                RawFrame(
                    index=index,
                    timestamp=float(extrinsics.get("timestamp", index)),
                    folder=folder,
                )
            )
        return frames

    @staticmethod
    def _read_meta(
        path: Path, default_width: int, default_height: int
    ) -> tuple[int, int]:
        if not path.is_file():
            return default_width, default_height
        meta = json.loads(path.read_text())
        return int(meta["width"]), int(meta["height"])

    # -- accessors -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> RawFrame:
        return self.frames[index]

    @property
    def has_confidence(self) -> bool:
        return any((frame.folder / "confidence.bin").is_file() for frame in self.frames)

    @property
    def arkit_mesh_path(self) -> Path | None:
        """No on-device mesh is saved alongside a raw recording."""
        return None

    def depth_shape(self, frame: RawFrame) -> tuple[int, int]:
        width, height = self._read_meta(frame.folder / "depth_meta.json", 256, 192)
        return height, width

    def load_depth(self, frame: RawFrame) -> np.ndarray:
        """Depth in metres, 0 where ARKit reported no measurement."""
        height, width = self.depth_shape(frame)
        depth = np.fromfile(frame.folder / "depth.bin", dtype=np.float32)
        if depth.size != width * height:
            raise ValueError(
                f"{frame.folder.name}: depth.bin has {depth.size} values, expected {width * height}"
            )
        depth = depth.reshape(height, width)

        # ARKit signals "no reading" with non-finite values rather than zero.
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)

    def load_confidence(self, frame: RawFrame) -> np.ndarray | None:
        path = frame.folder / "confidence.bin"
        if not path.is_file():
            return None
        width, height = self._read_meta(frame.folder / "confidence_meta.json", 256, 192)
        confidence = np.fromfile(path, dtype=np.uint8)
        if confidence.size != width * height:
            return None
        return confidence.reshape(height, width)

    def load_rgb(self, frame: RawFrame) -> np.ndarray:
        image = cv2.imread(str(frame.folder / "image.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {frame.folder / 'image.jpg'}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def intrinsics(self, frame: RawFrame) -> tuple[np.ndarray, int, int]:
        """Return (3x3 intrinsics, width, height) for the configured image grid."""
        data = json.loads((frame.folder / "camera_intrinsics.json").read_text())

        # Serialised column-major, so transposing recovers the usual layout with
        # the principal point in the last column.
        matrix = np.asarray(data["intrinsic_matrix"], dtype=np.float64).T
        rgb_width, rgb_height = (int(v) for v in data["image_resolution"])

        if self.intrinsics_space == "rgb":
            return matrix, rgb_width, rgb_height

        depth_height, depth_width = self.depth_shape(frame)
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height
        scaled = matrix.copy()
        scaled[0, :] *= scale_x  # fx and cx
        scaled[1, :] *= scale_y  # fy and cy
        return scaled, depth_width, depth_height

    def camera_to_world(
        self, frame: RawFrame, traj_convention: str = "rub"
    ) -> np.ndarray:
        """ARKit's camera-to-world transform, returned in RDF axes."""
        data = json.loads((frame.folder / "camera_extrinsics.json").read_text())
        transform = np.asarray(data["transform_matrix"], dtype=np.float64).T

        position = np.asarray(data["position"], dtype=np.float64)
        if not np.allclose(transform[:3, 3], position, atol=1e-4):
            raise ValueError(
                f"{frame.folder.name}: transform_matrix translation {transform[:3, 3]} does not "
                f"match position {position}; the matrix layout is not what this loader assumes"
            )

        if traj_convention == "rub":
            # Re-express the camera axes without moving the camera itself.
            return transform @ RUB_TO_RDF
        if traj_convention == "rdf":
            return transform
        raise ValueError(f"unknown traj_convention {traj_convention!r}")

    def world_to_camera(
        self, frame: RawFrame, traj_convention: str = "rub"
    ) -> np.ndarray:
        refined = self._refined.get(frame.folder.name)
        if refined is not None:
            # Already RDF world-to-camera, and already aligned back into ARKit's
            # frame, so it replaces the on-device pose outright.
            return refined
        return np.linalg.inv(self.camera_to_world(frame, traj_convention))

    def load_refined_poses(self, path: Path | str) -> int:
        """Adopt COLMAP-refined poses from `run_colmap_raw.py`.

        Frames COLMAP failed to register keep their ARKit poses, which is the
        safe fallback: a slightly drifted pose beats a missing one, and dropping
        those frames would leave holes in the fused surface.

        Returns the number of frames now using a refined pose.
        """
        data = json.loads(Path(path).read_text())
        poses = data.get("poses", data)

        self._refined = {}
        for entry in poses.values():
            matrix = np.asarray(entry["world_to_camera_rdf"], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"expected a 4x4 pose, got {matrix.shape}")
            self._refined[entry["frame"]] = matrix

        known = {frame.folder.name for frame in self.frames}
        return len(self._refined.keys() & known)

    @property
    def refined_pose_count(self) -> int:
        return len(self._refined)
