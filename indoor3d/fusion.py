"""Offline TSDF fusion of a posed RGB-D sequence.

ARKit's on-device `ARMeshAnchor` reconstruction runs in real time on a power
budget, so it fuses at roughly 4-5 cm voxels and cannot revisit earlier
geometry. Running the same idea offline lifts both limits: the voxel size can be
set by the detail wanted rather than the frame rate achievable, and every frame
can be cleaned before it is integrated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .data import Sequence
from .depth_filters import clean_depth
from .depth_upsampling import upsample


@dataclass
class FusionConfig:
    """Fusion parameters.

    `voxel_length` is the headline quality knob and the one worth tuning first.
    """

    voxel_length: float = 0.02
    # Truncation distance for the signed-distance field. Too small and surfaces
    # break up where poses disagree slightly; too large and thin structures get
    # blended away. A few voxels is the usual compromise.
    sdf_trunc_voxels: float = 4.0
    max_depth: float = 5.0
    frame_stride: int = 1
    # Offset into the stride. Fusing with (stride=2, offset=0) and evaluating on
    # (stride=2, offset=1) gives disjoint train/test frames, so a mesh cannot
    # score well merely by having memorised the depth it was built from.
    frame_offset: int = 0

    traj_convention: str = "rdf"

    # Phase 3. Depth arrives at a fraction of the image resolution, so geometry
    # is capped by the smaller grid; upsampling under RGB guidance lifts that
    # cap by putting depth edges where the image says they are.
    upsample_method: str = "none"  # none | bilinear | joint | solver
    upsample_scale: float = 1.0

    bilateral: bool = True
    drop_flying_pixels: bool = True
    min_confidence: int = 1
    sigma_spatial: float = 2.0
    sigma_depth: float = 0.02
    max_step: float = 0.05
    relative_step: float = 0.02

    @property
    def sdf_trunc(self) -> float:
        return self.voxel_length * self.sdf_trunc_voxels


@dataclass
class FusionReport:
    """What actually happened during a run, for reporting and comparison."""

    frames_integrated: int = 0
    frames_skipped: int = 0
    seconds: float = 0.0
    used_confidence: bool = False
    rejection_totals: dict[str, float] = field(default_factory=dict)

    def add_rejections(self, rejections: dict[str, float]) -> None:
        for key, value in rejections.items():
            if key == "kept_fraction":
                continue
            self.rejection_totals[key] = self.rejection_totals.get(key, 0) + value


def _as_open3d_intrinsic(
    intrinsics: np.ndarray, width: int, height: int
) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
    )


def fuse(
    sequence: Sequence,
    config: FusionConfig | None = None,
    progress_every: int = 100,
) -> tuple[o3d.geometry.TriangleMesh, FusionReport]:
    """Integrate a sequence into a TSDF volume and extract its surface."""
    config = config or FusionConfig()
    report = FusionReport(used_confidence=sequence.has_confidence)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_length,
        sdf_trunc=config.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    indices = range(config.frame_offset, len(sequence), config.frame_stride)
    started = time.perf_counter()

    for count, index in enumerate(indices):
        frame = sequence[index]

        try:
            extrinsic = sequence.world_to_camera(frame, config.traj_convention)
        except KeyError:
            report.frames_skipped += 1
            continue

        depth = sequence.load_depth(frame)
        confidence = sequence.load_confidence(frame)
        intrinsics, width, height = sequence.intrinsics(frame)

        # Cleaning happens on the native grid: flying pixels are defined by the
        # sensor's own sampling, and the confidence map has no meaning at any
        # other resolution.
        depth, rejections = clean_depth(
            depth,
            confidence,
            bilateral=config.bilateral,
            drop_flying_pixels=config.drop_flying_pixels,
            min_confidence=config.min_confidence,
            max_depth=config.max_depth,
            sigma_spatial=config.sigma_spatial,
            sigma_depth=config.sigma_depth,
            max_step=config.max_step,
            relative_step=config.relative_step,
        )
        report.add_rejections(rejections)

        if not np.any(depth > 0):
            report.frames_skipped += 1
            continue

        color = sequence.load_rgb(frame)

        if config.upsample_method != "none" and config.upsample_scale > 1.0:
            target_w = int(round(depth.shape[1] * config.upsample_scale))
            target_h = int(round(depth.shape[0] * config.upsample_scale))
            scale_x = target_w / depth.shape[1]
            scale_y = target_h / depth.shape[0]

            depth = upsample(
                depth, color, confidence, target_w, target_h, config.upsample_method
            )

            # Intrinsics describe a pixel grid, so they must follow it.
            intrinsics = intrinsics.copy()
            intrinsics[0, :] *= scale_x
            intrinsics[1, :] *= scale_y
            width, height = target_w, target_h

        # The two streams normally share a resolution, but the intrinsics
        # describe the depth grid, so colour is what has to move if they differ.
        if color.shape[:2] != depth.shape:
            color = cv2.resize(
                color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=config.max_depth,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, _as_open3d_intrinsic(intrinsics, width, height), extrinsic)
        report.frames_integrated += 1

        if progress_every and count % progress_every == 0:
            print(f"  integrated {report.frames_integrated} frames", flush=True)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    report.seconds = time.perf_counter() - started
    return mesh, report


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise RuntimeError(f"failed to write mesh to {path}")
    return path
