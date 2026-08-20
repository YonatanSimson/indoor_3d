# Offline Indoor 3D Scanning — status and hand-off

**Goal:** free, fully offline indoor 3D reconstruction, targeting
Polycam/Scaniverse-level output with no cloud dependency. ARKit captures RGB +
LiDAR depth + poses; everything after that runs locally.

**Baseline to beat:** ARKit's on-device `ARMeshAnchor` mesh — a coarse real-time
voxel TSDF, fine as a live preview, not as output.

This file is the entry point: what is built, what was measured, what failed, and
what to do next. [README.md](README.md) holds the evidence behind each claim;
[phase4_plan.md](phase4_plan.md) documents the MVS attempt in detail.

---

## Status

| Phase | State |
|---|---|
| 0 — capture | done, by the `model_3d` iOS app (separate repo) |
| 1 — custom TSDF fusion | **done** — 2.9x better than ARKit's on-device mesh |
| 2 — pose refinement (COLMAP) | **done** — +8%, and it caps everything downstream |
| 3 — depth upsampling | **done** — anchored upsampling, used in the best config |
| 4 — dense MVS | **done — negative.** Does not help this capture |
| 5 — surface & texture | not started |
| 6 — polish | not started |
| **V — validation on ground truth** | **next, and the priority.** See below |

### Best known configuration

```powershell
conda activate indoor3d
python scripts/run_fusion_raw.py --refined-poses --voxel 0.010 \
    --upsample anchored --upsample-scale 3 --out output/best.ply
```

Seeded COLMAP poses, 10 mm voxel, anchored upsampling. Chosen **by eye** — the
error metric ranks it second, and the metric is not trustworthy (next section).

---

## Read this before trusting any number in this repo

Almost every figure here is the distance from raw LiDAR points to the
reconstructed surface — scored against the same noisy sensor data that built the
mesh. It rewards fitting noise, and it ranked **opposite to visual quality
twice**:

| mesh | median error | surface area | verdict by eye |
|---|---|---|---|
| 15 mm | 10.10 mm | **127.6 m2** | third |
| 10 mm | **8.99 mm** | 155.8 m2 | second |
| 10 mm + anchored | 9.79 mm | 155.3 m2 | **best** |
| 5 mm | **7.34 mm** | 218.6 m2 | **worst** — visibly crumpled |

Surface area proved the better signal: a room that has not changed size gaining
71% of surface area is roughness being resolved, not detail.

Treat the error figures as a **regression check**, not a quality ranking. This
is the main reason phase V matters.

---

## What failed, and why — do not repeat these

**Dense MVS (Phase 4).** 20.8 mm accuracy at 11% pixel coverage, against the
anchored solver's 4.5 mm at 100%. Close-range indoor gives patch-match nothing
to match: median scene depth 1.03 m, large blank walls, and `geom_consistency`
correctly discarding most of what it proposes. Filling LiDAR gaps — MVS's one
unique offering — reached 11%.

**Wide baselines make MVS worse, not better.** We reasoned from triangulation
geometry that 30 degrees would beat 15 (1.43 vs 2.91 mm theoretical limit).
Measured, 30 degrees lost on *every* axis including accuracy (38.5 vs 20.8 mm;
36.8% vs 11.0% gross disagreement). That limit assumes correct matches; at 55 cm
baseline in a 1 m room, perspective distortion breaks NCC matching far faster
than the geometry improves.

**Pose priors.** `pose_prior_mapper` fixed the gauge beautifully — scale error
74% to 0.85% — and wrecked the trajectory (47-70 mm mesh error vs 10 mm). The
cause is structural: **position priors do not constrain rotation**, and fused
surfaces are most sensitive to rotation (1 degree at 2 m displaces geometry
35 mm). ARKit's orientation is its strongest output; that path discards it.

**Voxels below ~10 mm.** You cannot resolve below your pose accuracy. COLMAP and
ARKit disagree by ~7.5 mm on camera position, so at 5 mm the same wall lands in
different voxels per frame and averaging becomes crumpling.

**Laser ground truth for ARKitScenes.** Downloaded 1.9 GB; the scan covers a
whole building (1574 m2) with our scene a 95 m2 room inside it, and the dataset
ships **no laser-to-ARKit transform**. Blind registration reaches 0.076 fitness.
Would need coarse localisation of the room first — a real task, not a flag.

---

## Gotchas that cost real time

**Camera conventions differ between our two data sources.** Verified against
data, not assumed:

| | ARKitScenes `.traj` | model_3d `RawRecorder` |
|---|---|---|
| direction | world → camera | camera → world |
| axes | **RDF** (OpenCV) | **RUB** (ARKit native) |
| serialisation | one line per pose | per-frame JSON, **column-major** |
| into Open3D | pass through | `RUB_TO_RDF @ inv(T)` |

Reading a RawRecorder session needs four corrections, not one: transposed
matrices (pose *and* intrinsics), direction, axes, and intrinsics describing the
1920x1440 RGB frame while depth is 256x192.

**COLMAP specifics.**
- Geometric consistency requires every *source* view to have been solved as a
  *reference*. Restricting references while leaving all frames selectable as
  sources aborts the run partway through.
- Deregistering frames guts the sparse points (136,558 to 2,052 here) because
  tracks lose observations — and source selection ranks by shared points.
  **Re-triangulate after reducing a model.**
- Mappers write into numbered subdirectories (`sparse/refined/0/`). Reading the
  parent silently picks up an older run. This produced a full set of plausible
  numbers belonging to a different experiment.
- Bundle adjustment leaves 7 DOF free; raw refined poses sit in an arbitrary
  gauge, up to 3.8x off metric here.
- `min_triangulation_angle` defaults to 1 degree, far too permissive — but read
  the MVS failure above before raising it much.

**Maths traps that produced confident nonsense.**
- `camera_centre = -R^T t` assumes an orthonormal rotation. After a *scaled*
  similarity it returns garbage — this invented an 8.5 m drift that did not exist.
- `images.txt` has two lines per image and the second may be empty. Parsing by
  alternation while skipping blank lines drops every other pose.
- Rejecting MVS points against LiDAR must be **asymmetric**: a point is only
  disproven when the sensor reads *further* than it (free-space violation); a
  nearer reading just means occlusion. Treating both as disagreement rejected
  97.5% of points instead of 7%.

---

## Next: validate against ground truth (phase V)

Everything above rests on a metric shown to be unreliable, and on visual
judgement that does not scale. The fix is a dataset where truth is known.

**Data is downloaded and ready:** `~/data/iclnuim/` — ICL-NUIM living room
trajectory 2 (882 frames), **CC BY 3.0, commercial use permitted with
attribution**. It ships what nothing else here has:

- `living-room.ply` — ground-truth surface
- `livingRoom2.gt.freiburg` — ground-truth poses, TUM format
- `rgb/` + `depth/` at 640x480, clean (noisy variants also downloadable)

**Simulate our sensor** by degrading it: 640x480 to 256x192, depth-dependent
noise, and a synthetic confidence map — confidence gating does real work in this
pipeline, and a simulation without it would not exercise that path.

### The experiment that matters most

Fuse identical depth at 5 mm twice: once with ground-truth poses, once with
poses perturbed by ~7.5 mm to match our real pose uncertainty.

- **Clean with GT poses** confirms pose accuracy is the ceiling. Effort belongs
  in Phase 2, and better poses would unlock finer voxels and real detail.
- **Still crumpled** means depth noise is the limit. Pose refinement is near its
  useful end, and temporal averaging across frames is the lever instead.

That single result decides the project's direction, and it is currently a guess.

### Then re-derive the metrics

With true geometry available, recheck every claim resting on distance-to-LiDAR:

1. Does 5 mm actually resolve noise? Ground truth answers this outright.
2. Does anchored upsampling recover real structure? Hold out the true 640x480
   depth and see whether the upsampler reconstructs it — a genuine held-out test.
3. Which metric correlates with truth? Surface area outperformed the error
   figures here; that hypothesis is now testable.
4. Does the 2.9x Phase 1 improvement over ARKit survive against real geometry?

**Known trap:** ICL-NUIM uses a **negative focal length** in its camera model,
and SurfReg expects reconstructions aligned to the first frame at the origin.
Verify by back-projecting against the GT mesh before trusting anything — the way
`scripts/check_convention.py` settled the ARKitScenes question. Conventions have
cost more time on this project than any algorithm.

---

## Remaining forward plan

**Phase 5 — surface & texture.** The biggest perceived-quality jump, and
untouched. Poisson reconstruction from the fused oriented cloud, mesh cleanup
(floating fragments, decimation, hole filling), UV unwrap, project captured RGB
as texture, export OBJ/GLTF/USDZ. Note colour is currently per-vertex, so
texture detail is capped by geometry density — a texture map removes that link.

**Phase 6 — polish.** ARKit plane/semantic labels to flatten walls and floors
(cheap regularisation, high visual impact). Neural implicit surfaces as a
possible replacement for classical TSDF — see External Repos.

**Coverage mapping (unbuilt, recommended).** The one quality complaint about
real output — a lumpy ball — turned out to be a *coverage* problem: matte
surface, in a corner, seen from few angles at oblique incidence. Measuring
observations-per-surface and incidence angle across a scan would separate "noisy
reconstruction" from "you did not look at this properly", and would guide
capture rather than post-processing. Reuses existing poses and raycasting.

---

## External Repos

| Repo | Why | For | Status |
|---|---|---|---|
| [PRBonn/SHINE_mapping](https://github.com/PRBonn/SHINE_mapping) | Sparse octree-encoded neural implicit mapping (ICRA 2023) — beat VDBFusion on accuracy/completeness in recent benchmarks; candidate to replace/augment classical TSDF fusion | Phase 6 | cloned — `work/SHINE_mapping` |
| [ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) | Monocular/multi-view depth. Tested: metric claim does not hold indoors (scale 0.92-1.67 across frames of one room); ~4% shape error after affine fit. Useful as structure, not geometry | Phase 3 | cloned — `work/depth-anything-3` |
| [rvp-group/mrhash](https://github.com/rvp-group/mrhash) | Hash-based GPU TSDF fusion, variance-adaptive voxel grid (2025). **Tested against our exported dataset** (539 frames, anchored-upsampled depth, refined poses) at 10mm voxels: 2,886,403 verts / 2,930,679 faces, ~106s (8.7 fps). Handled 10mm where nvblox could not — **wins visually and on memory** (Pixi env, isolated CUDA 12.8, no conflict with system CUDA 13.3) | fusion candidate | built + run — `work/mrhash` (Win), `~/work/mrhash` (WSL) |
| [nvidia-isaac/nvblox](https://github.com/nvidia-isaac/nvblox) | GPU voxel-hashing TSDF/ESDF fusion (not a true octree — see mrhash comparison in the original tooling survey). **Tested** on the same export: hard CUDA OOM at 10mm voxels (GPU buffer growth exceeds 16GB VRAM); succeeded at 20mm — 571,366 verts / 841,561 faces, 13.4s. Heavier per-voxel GPU footprint than mrhash at matched resolution | fusion candidate, lost to mrhash | built + run — `work/nvblox` (Win), `~/work/nvblox` (WSL) |
| [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio), [ethz-asl/voxblox](https://github.com/ethz-asl/voxblox), [ethz-mrl/supereight2](https://github.com/ethz-mrl/supereight2) | 3DGS trainer (reads COLMAP directly) and two more classical TSDF baselines from the original tooling survey | not yet run | cloned, untested — `work/<name>` |

---

## Layout

**Environments** (deliberately separate — DA3 pins numpy back to 1.26, COLMAP's
Qt collides with Open3D's):

```powershell
conda activate indoor3d   # fusion + evaluation
conda activate colmap     # CUDA COLMAP binary, invoked as a subprocess
conda activate depth      # torch cu128 + Depth Anything 3
```

GPU note: the RTX 5070 Ti is `sm_120`; only `--index-url .../whl/cu128` works.
Earlier builds pass `torch.cuda.is_available()` and still fail at kernel launch,
so verify with an actual matmul.

**Data:**

| path | what |
|---|---|
| `~/data/iphone/RawRecording_2026-08-18_215941/` | 539-frame iPhone capture |
| `~/data/iphone/.../colmap/refined_poses.json` | **seeded** refined poses — the good ones |
| `~/data/iclnuim/` | ICL-NUIM + ground truth, for phase V |
| `~/data/arkitscenes_gt/` | one Faro scan; unusable, see above |
| `~/work/ARKit3d/sample_data/` | ARKitScenes scene 42444946 |

Beware: `~/data/iphone/.../colmap/sparse/` holds several models from different
experiments. `refined_poses_seed.json` (539 poses, 7.5 mm) is the good one;
`refined_poses_priors*.json` are the failed variants.

**Code:** `indoor3d/` is the library (loaders, filters, upsampling, anchoring,
fusion); `scripts/` the entry points. Each script's docstring explains *why* it
does what it does, including the failures that shaped it.

**Licensing** (checked, since commercial use is intended): ICL-NUIM is CC BY
3.0. ARKitScenes grants commercial rights under 700M MAU, and its README
confirms the licence covers the dataset, not just the code. ScanNet++ was ruled
out for this reason. Your own captures are unencumbered. Not legal advice —
worth a review before shipping.

---

## Compute notes

- **Phases 0-3:** realistically portable to on-device (Metal) over time.
- **Phase 4:** heavy, and now known not to pay off for close-range indoor scans.

**WSL2, for building external Linux/CUDA repos (nvblox, mrhash, LichtFeld-Studio).**
Default WSL2 memory cap is 50% of host RAM — 7.7GB on this 15.8GB-RAM machine,
which OOM-killed a LibTorch build mid-compile. Fixed via `C:\Users\user\.wslconfig`:
`memory=12GB`, `swap=16GB` (swap is disk-backed and cheap; physical RAM isn't —
host only has 15.8GB total, so don't push `memory` much higher or Windows itself
starves). Takes a `wsl --shutdown` + relaunch to apply; the filesystem and
already-cloned repos survive the restart, only in-memory state is lost — resume
`ninja`/`cmake --build` rather than reconfiguring.

CUDA Toolkit for WSL-Ubuntu is a **separate install from the Windows one** (the
driver is stubbed through from the Windows host via `libcuda.so`; installing
`cuda`/`cuda-drivers` meta-packages inside WSL is wrong and fights that
passthrough — install `cuda-toolkit-13-3` specifically). GCC 14 and CMake 3.30+
also needed manually (Ubuntu 24.04's apt defaults are 13.3 / 3.28.3, both too
old for these projects); `pip3 install --break-system-packages cmake` gets a
current one, but lands in `~/.local/bin`, which is **not** reliably on `PATH` in
a non-interactive `wsl -- bash -lc` shell — pass it explicitly or the stale
`/usr/bin/cmake` wins silently.

When a repo's manual-install path wants a different CUDA minor than what's
system-installed (mrhash wanted 12.6-12.x against our 13.3), reach for
[Pixi](https://pixi.sh) if the repo offers it — pins its own isolated CUDA
toolkit via conda-forge, sidesteps the conflict entirely instead of juggling
multiple system CUDA installs.
