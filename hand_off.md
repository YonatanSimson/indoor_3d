Got it — since the app itself isn't cooperating, here's the whole plan inline so you've got it either way:

---

# Offline Indoor 3D Scanning Pipeline — ARKit Frontend

**Goal:** Free, fully offline indoor 3D reconstruction app, using ARKit for capture (RGB + LiDAR depth + poses) and a local processing pipeline for reconstruction — targeting Polycam/Scaniverse-level output quality without any cloud dependency.

**Starting point:** ARKit's built-in `ARMeshAnchor` scene reconstruction — a coarse, real-time voxel TSDF mesh. Usable as a live preview, not as final output.

---

## Phase 0 — Capture Pipeline (foundation)

**Objective:** Record everything needed for offline processing; don't process anything on-device yet.

- `ARSession` with `.sceneDepth` / `.smoothedSceneDepth` frame semantics enabled (turn off `.sceneReconstruction` — not needed if we're doing our own fusion).
- Per frame, capture and persist to disk:
  - RGB image (JPEG)
  - Depth map (EXR or 16-bit PNG)
  - Confidence map
  - Camera pose (`ARCamera.transform`)
  - Camera intrinsics
- **Milestone:** A recorded dataset you can iterate against offline without rescanning physical space each time.

---

## Phase 1 — Custom TSDF Fusion

**Objective:** Beat ARKit's native mesh quality immediately.

- Offline fusion (Open3D `ScalableTSDFVolume`, voxblox, or InfiniTAM) at finer voxel resolution than ARKit's on-device default (~4–5cm).
- Weight depth integration by the confidence map.
- Bilateral-filter depth before integration; drop "flying pixel" artifacts at depth discontinuities.
- Marching cubes → mesh export.
- **Milestone:** Visibly sharper, less blocky mesh than the built-in `ARMeshAnchor` output.

---

## Phase 2 — Pose Refinement

**Objective:** Eliminate drift/smearing on longer or looping scans.

- ARKit's on-device VIO has no loop closure — drifts over longer trajectories.
- Run captured RGB frames through COLMAP (SfM + bundle adjustment), initialized from ARKit poses.
- Re-run Phase 1 fusion using the refined trajectory.
- **Milestone:** No double-walls / ghosting artifacts when a scan path loops back on itself.

---

## Phase 3 — Depth Quality Improvement

**Objective:** Recover detail the LiDAR sensor's native resolution doesn't capture.

- Joint bilateral upsampling: use RGB edges to guide/sharpen the low-res (~256×192) LiDAR depth instead of default bilinear upsampling.
- Optional: lightweight depth-completion model to fill holes on glossy/thin/reflective surfaces.
- **Milestone:** Sharper edges, fewer holes in the fused mesh.

---

## Phase 4 — Multi-View Stereo Pass

**Objective:** This is the step that actually closes the detail gap with Polycam.

- Run COLMAP dense MVS (patch-match stereo) on the posed RGB frames, seeded/constrained by LiDAR depth for fast, metrically-correct convergence.
- Fuse MVS depth with the LiDAR-based TSDF: trust LiDAR where confident, let MVS resolve fine surface detail LiDAR physically can't.
- **Milestone:** Geometric detail comparable to Polycam's LiDAR capture mode.

---

## Phase 5 — Surface & Texture (the "finished app" step)

**Objective:** Untextured geometry never reads as professional, regardless of accuracy — this phase is the biggest perceived-quality jump.

- Poisson surface reconstruction from the fused, oriented point cloud (replaces raw marching-cubes-on-TSDF) — cleaner topology, smoother surfaces.
- Mesh cleanup: remove floating fragments, quadric decimation, hole filling.
- UV-unwrap and project captured RGB frames as texture.
- Export as OBJ/GLTF/USDZ — USDZ gets free AR Quick Look support on iOS.
- **Milestone:** A textured, exportable, shareable model.

---

## Phase 6 — Optional Polish

- Use ARKit mesh classification (wall/floor/ceiling semantic labels) to snap planar regions perfectly flat — cheap regularization, high visual impact.
- Neural implicit surface fitting (NICE-SLAM / GO-Surf style, or SHINE-Mapping — see External Repos below) for detail beyond MVS — GPU-bound, offline, not real-time.
- Split live low-res on-device preview from a full-quality offline reprocess pass — mirrors how Polycam/Scaniverse are actually architected.

---

## External Repos

| Repo | Why | For | Status |
|---|---|---|---|
| [PRBonn/SHINE_mapping](https://github.com/PRBonn/SHINE_mapping) | Sparse octree-encoded neural implicit mapping (ICRA 2023) — beat VDBFusion on accuracy/completeness in recent benchmarks; candidate to replace/augment classical TSDF fusion | Phase 6 | cloned — `work/SHINE_mapping` |

---

## Compute Notes

- **Phases 0–3:** realistically portable to on-device (Metal) over time.
- **Phase 4 (MVS):** heavy — plan for Mac-side offline processing initially; porting patch-match stereo to Metal is a later optimization, not a starting requirement.

---

## Suggested Tooling Summary

| Stage | Tool/Library |
|---|---|
| Capture | ARKit (`ARSession`, `.sceneDepth`) |
| TSDF fusion | Open3D `ScalableTSDFVolume`, voxblox, or InfiniTAM |
| Pose refinement | COLMAP (SfM + BA) |
| Depth upsampling | Joint bilateral filter (custom or OpenCV) |
| Dense MVS | COLMAP patch-match stereo |
| Surface reconstruction | Poisson (Open3D / PoissonRecon) |
| Export | OBJ / GLTF / USDZ |

