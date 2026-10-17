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
| 1 — custom TSDF fusion | **done, then superseded** — our Open3D fusion beat ARKit's on-device mesh by 2.9x, but lost a head-to-head against external tools (see External Repos) — **mrhash is now the chosen fusion tool** |
| 2 — pose refinement (COLMAP) | **done** — +8%, and it caps everything downstream |
| 3 — depth upsampling | **done** — anchored upsampling, used in the best config and feeds mrhash too |
| 4 — dense MVS | **done — negative.** Does not help this capture |
| 5 — surface & texture | not started |
| 6 — polish | not started |
| **V — validation on ground truth** | **in progress (ICL-NUIM pass).** Pose ceiling, voxel sweep and 2/8 ablation done; DA3 depth cache built, DVP-MVS on ICL-NUIM staged but not run. See "Phase V status (2026-10-17)" below. ETH3D pass not started |

### Best known configuration

**Chosen fusion tool: mrhash** ([rvp-group/mrhash](https://github.com/rvp-group/mrhash)), picked over supereight2 and nvblox by direct visual comparison (see External Repos). Two steps — export our data to Replica format, then run mrhash against it:

```powershell
# 1. Export (Windows, indoor3d conda env) - anchored upsampling + refined COLMAP poses by default
conda activate indoor3d
python scripts/export_replica.py
# writes <session>/replica/scene/ (results/ + traj.txt) and <session>/replica/cam_params.json
```

```bash
# 2. Run (WSL, mrhash's pixi env) - config at mrhash/configurations/indoor3d.cfg points
# at the exported scene; edit data_path/results_path there for a different session
cd ~/work/mrhash && pixi run python mrhash/apps/rgbd_runner.py mrhash/configurations/indoor3d.cfg
```

10 mm voxel, seeded COLMAP poses, anchored upsampling — same winning recipe as our
own fusion's best config below, just run through mrhash's GPU hash-based fusion
instead of our Open3D pipeline.

**Our own fusion** (`scripts/run_fusion_raw.py`) is kept as a documented fallback —
no external build/env dependency, useful if mrhash's WSL/pixi setup isn't available:

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

## Phase V results (2026-08-24)

**A substantial chunk of Phase V was already run before this was written down**
— `hand_off_original_plan.md` (a more detailed sibling doc, referenced by all
the eval scripts below) and `scripts/{run_colmap_iclnuim,check_convention_iclnuim,
eval_iclnuim_pose_ceiling,eval_iclnuim_voxel_sweep,eval_iclnuim_ablation,
eval_iclnuim_dense_mvs}.py` plus `indoor3d/{data_iclnuim,iclnuim_eval,sensor_sim,
colmap_utils,pose_perturbation}.py` already exist. Re-scored the existing output
meshes against ground truth without re-running fusion (`scripts/
rescore_iclnuim_existing.py` — cheap, reuses the cached ground-truth KD-tree):

**Pose ceiling — decisive, complete, trustworthy.** GT poses: 10.4mm median
accuracy. Perturbed poses (~7.5mm, matching real COLMAP-vs-ARKit disagreement):
15.4mm (1.48x worse). **Pose accuracy is the ceiling** — this is no longer a
guess. Phase 2 (pose refinement) is the real lever; finer voxels help once
poses improve.

**Voxel sweep — complete, confirms an existing hypothesis with real data.**
15mm: 10.5mm | **10mm: 9.7mm (best)** | 5mm: 10.4mm. Non-monotonic, confirming
"you cannot resolve below your pose accuracy" (Gotchas section, above) with
ground truth instead of the distrusted raw-LiDAR proxy.

**Ablation — only 2 of 8 planned depth×pose combinations ran.** `full 640x480
depth + GT poses` (the ceiling question): 6.8mm — right in SOTA range
(BundleFusion 7mm, ElasticFusion 8mm). `degraded + anchored-upsampled + GT
poses`: 27.1mm, ~4x worse than the ceiling. That's also worse than the *plain
degraded, non-upsampled* depth scored 10.4mm in the pose-ceiling run (same GT
poses, same 5mm voxel) — a 2.6x gap suggesting the bilateral-solver anchored
upsampler may be introducing real geometric bias despite looking visually
sharp. **Confirmed real, not a measurement confound** — the user independently
expected this finding. **A separate concurrent agent session is actively
working on replacing/augmenting the bilateral-solver upsampler with a DA3
(Depth Anything 3) prior** — matches the `da3_anchored` mode already
scaffolded in `eval_iclnuim_ablation.py` / `indoor3d/da3_cache.py` /
`scripts/precompute_da3_depth.py`. **Don't duplicate that work.** The missing
`degraded+gt` combination (same script, same everything as `upsampled+gt`
minus the upsample step) would be the cheapest way to fully isolate the
upsampler's cost, if it's ever needed — not run yet.

**Scale note:** COLMAP is currently seeded with ARKit poses for approximate
metric scale (ARKit's VIO is metric via its IMU); exact scale precision is
explicitly not the current priority.

**Dense MVS ablation — superseded, do not run.** No `~/data/iclnuim/colmap_dense/`
output exists despite the script (`eval_iclnuim_dense_mvs.py`) existing and
looking complete. **Decided 2026-08-24: use DVP-MVS instead of this script** —
see the section directly below. Several partial/aborted raw-dense-COLMAP
attempts on the real capture (`colmap/dense*`, `dense_aborted_221547`,
`dense_pilot30_starved` under `C:\Users\user\data\iphone\RawRecording_2026-08-18_215941\colmap\`)
are on disk from before this decision — checked, all empty/incomplete
(`dense/sparse` and `dense/images` both empty), i.e. raw dense COLMAP was
stalling/too slow, which is part of why DVP-MVS was chosen instead. Safe to
delete those `dense*` subfolders once DVP-MVS is confirmed working; not
deleted yet.

### Phase V status (2026-10-17) — verified against files on disk

| Item | State |
|---|---|
| Pose ceiling, voxel sweep | done (results above) |
| Ablation | 2 of 8 combos have output (`full_gt`, `upsampled_gt` in `iclnuim_output/ablation/`) |
| DA3 depth cache | built: `~/data/iclnuim/da3_depth_cache/` (880 files). `da3_anchored` ablation **not yet run** |
| DVP-MVS, iPhone capture | done: `iphone/RawRecording_2026-08-18_215941/dvpmvs/mvsnet_format/APD/APD.ply` + `APD_mesh.ply` |
| DVP-MVS, ICL-NUIM | inputs staged (`~/data/iclnuim/dvpmvs/`), **no `APD.ply` yet**, so not scored |
| ETH3D pass | not started |

**Next, in order:** (1) run DVP-MVS on ICL-NUIM and score `APD.ply` with
`iclnuim_eval.report()` against `living-room.ply`; (2) run the `da3_anchored`
ablation, plus the cheap `degraded+gt` combo to isolate the bilateral-solver
upsampler's cost; (3) view results (Open3D/Plotly for 3D, Matplotlib for plots)
before drawing conclusions; (4) only then start ETH3D. Phase 5/6 stay parked
until the upsampler question is settled.

---

## DVP-MVS setup (2026-08-24) — handoff for next agent

**Decision:** run DVP-MVS (PatchMatch MVS, AAAI 2025,
`github.com/ZhenlongYuan/DVP-MVS`) on the real RawRecording capture and on
ICL-NUIM, using the sparse COLMAP results we already have, and compare its
output to mrhash's. Replaces the dense-COLMAP ablation above — don't run
`eval_iclnuim_dense_mvs.py`.

**Build: done, both clones.** Windows clone `C:\Users\user\work\DVP-MVS` is
for editing/reference; the WSL clone `~/work/DVP-MVS` is what actually builds
and runs (same two-clone pattern as LichtFeld-Studio, see Compute notes above
— a source fix has to land in *both* or the build silently keeps the old
file). Two fixes were needed, applied to both clones:
- `CMakeLists.txt`: `cmake_minimum_required` 2.8→3.10, `OpenCV_DIR` hardcoded
  Windows path wrapped in `if(WIN32)...endif()`, CUDA arch `sm_80`→`sm_120`
  (Blackwell).
- `main.h`: deleted the `#include <cuda_texture_types.h>` line — CUDA 13.3
  dropped that legacy header, and the code only uses the modern
  `cudaTextureObject_t` API (confirmed via grep), not legacy `texture<>`
  references, so it isn't needed.

Executable confirmed working: `~/work/DVP-MVS/build/APD` →
`USAGE: APD dense_folder`.

**Known caveat, accepted by user:** the public DVP-MVS code does **not**
actually contain the paper's claimed Depth-Anything-V2 integration — only
classic `cv::Canny` edge detection. User chose to build/run it anyway.
Wiring in a real monocular prior is explicit future work, not now — user has
both DA2 and DA3 on disk locally and said "either will do" when we get to it.

**Input format — this is the part not yet done.** `APD` takes one
MVSNet-format `dense_folder` (must contain `cams/`, `images/`, `pair.txt`),
produced by `colmap2mvsnet.py` (in the DVP-MVS repo root). That script is
ETH3D-shaped and hardcodes `image_dir = dense_folder/images` and
`model_dir = dense_folder/dslr_calibration_undistorted` — it does **not**
take raw COLMAP sparse output paths directly. Before running it, build a
staging dir per dataset:
```
<staging>/dense_folder/
  images/                          -> symlink to the COLMAP images dir
  dslr_calibration_undistorted/    -> cameras.txt, images.txt, points3D.txt
                                       copied/symlinked from the sparse model
```
then:
```
python3 colmap2mvsnet.py --dense_folder <staging>/dense_folder \
    --save_folder <staging>/mvsnet_format --model_ext .txt
~/work/DVP-MVS/build/APD <staging>/mvsnet_format
# output point cloud: <staging>/mvsnet_format/APD/APD.ply
```
WSL `python3` needed `numpy`/`pillow` for this script — installed via
`pip3 install --user --break-system-packages numpy pillow` (already done).

**Real capture COLMAP data, confirmed ready:**
`C:\Users\user\data\iphone\RawRecording_2026-08-18_215941\colmap\sparse\refined\`
— `cameras.txt` (539 cameras, **PINHOLE model, already undistorted, no
distortion params** — per-frame ARKit-seeded intrinsics, plugs directly into
`colmap2mvsnet.py`'s PINHOLE handling with no extra work), `images.txt`
(1082 lines ≈ 539 posed images), `points3D.txt` (160k lines, well
triangulated). Source images at `colmap/images/` (539 files).

**ICL-NUIM COLMAP data:** `~/data/iclnuim/colmap/sparse/{seed,triangulated,
refined}/` — use `refined` (or `triangulated` if `refined` turns out to be
posterior-only) the same way, images at `~/data/iclnuim/rgb/`.

**Not started yet:**
1. Build the staging dir + run `colmap2mvsnet.py` + run `APD` for the real
   capture.
2. Same for ICL-NUIM.
3. Score the ICL-NUIM DVP-MVS point cloud against `living-room.ply` using
   `indoor3d/iclnuim_eval.py`'s `report()` (it's a point cloud, not a mesh —
   load with `o3d.io.read_point_cloud` instead of `read_triangle_mesh`;
   `report()` should work unchanged on a point set but verify).
4. Compare both results to mrhash's recorded numbers (mrhash mesh outputs are
   in `iclnuim_output/` / `replica/mrhash_mesh_10mm.ply` — see "Best known
   configuration" section above).

---

## Next: validate against ground truth (phase V)

Everything above rests on a metric shown to be unreliable, and on visual
judgement that does not scale. The fix is a dataset where truth is known.

**Two-pass plan, decided 2026-08-24:** ICL-NUIM first (fast, ready-made RGB-D,
answers the pose-vs-noise question below), then a second real-world pass on
**ETH3D** (CC BY-NC-SA 3.0 — viable now that this is a personal/non-commercial
project; real survey-grade laser-scan ground truth across real indoor scenes,
stronger validation than synthetic, but needs a raycast-based pseudo-depth
exporter since ETH3D ships photos + laser scan + COLMAP poses, not RGB-D pairs
— same raycasting technique already used elsewhere in this project for
measurement error). ETH3D not started; the ICL-NUIM pass is under way (see
"Phase V status (2026-10-17)" below).

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
| [ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) | Monocular/multi-view depth. Tested: metric claim does not hold indoors (scale 0.92-1.67 across frames of one room); ~4% shape error after affine fit. Useful as structure, not geometry | Phase 3 | cloned — `work/depth-anything-3` |
| [rvp-group/mrhash](https://github.com/rvp-group/mrhash) | Hash-based GPU TSDF fusion, variance-adaptive voxel grid (2025). **Tested against our exported dataset** (539 frames, anchored-upsampled depth, refined poses) at 10mm voxels: 2,886,403 verts / 2,930,679 faces, ~106s (8.7 fps). Handled 10mm where nvblox could not (Pixi env, isolated CUDA 12.8, no conflict with system CUDA 13.3). **Final ranking, by eye: mrhash > supereight2 > nvblox** (SHINE_mapping parked, see below). mrhash wins on visual quality, speed edge is small (supereight2 was actually ~17s faster), and mrhash's mesh is ~3.7x smaller on disk (169MB vs 621MB, since supereight2's is unwelded — 3 verts/face). **Decision: mrhash chosen as the fusion tool going forward** | **fusion candidate — chosen** | built + run — `work/mrhash` (Win), `~/work/mrhash` (WSL) |
| [nvidia-isaac/nvblox](https://github.com/nvidia-isaac/nvblox) | GPU voxel-hashing TSDF/ESDF fusion (not a true octree — see mrhash comparison in the original tooling survey). **Tested** on the same export: hard CUDA OOM at 10mm voxels (GPU buffer growth exceeds 16GB VRAM); succeeded at 20mm — 571,366 verts / 841,561 faces, 13.4s. Heavier per-voxel GPU footprint than mrhash at matched resolution. **Visual verdict: medium** | fusion candidate, lost to mrhash and supereight2 on looks | built + run — `work/nvblox` (Win), `~/work/nvblox` (WSL) |
| [ethz-mrl/supereight2](https://github.com/ethz-mrl/supereight2) | CPU-only octree TSDF (no CUDA at all). **Tested** on a TUM-format re-export of the same anchored depth/poses (`export_tum.py`): 88.5s total (539 frames, 6.1 fps), 13,428,342 verts / 4,476,114 faces. Note: unwelded mesh (exactly 3 verts/face), so vertex count isn't directly comparable to the others — triangle count (4.48M) is the fair number. No GPU, no OOM risk. **Visual verdict: much better — comparable to mrhash**, confirmed via direct side-by-side (`scripts/view_supereight2_vs_mrhash.py`): "both pretty good, [with] mrhash slightly better" — final ranking: **mrhash > supereight2**, both well ahead of nvblox, despite supereight2 being the only CPU-only entrant | **fusion candidate, runner-up to mrhash** | built + run — `work/supereight2` (Win), `~/work/supereight2` (WSL) |
| [PRBonn/SHINE_mapping](https://github.com/PRBonn/SHINE_mapping) | Sparse octree-encoded neural implicit mapping (ICRA 2023) — beat VDBFusion on accuracy/completeness in published benchmarks. **Tested**: reused the Replica export reformatted to KITTI point-cloud format via the repo's own `rgbd_to_kitti_format.py`. Modern torch 2.8+cu128/kaolin 0.18 stack required (README's stale torch≤1.13.1 predates Blackwell). Batch mode's `every_frame=1` accumulation is genuinely O(n²) (`lidar_dataset.py:263`, repeated `torch.cat` on a growing pool) — needed `every_frame=4` subsampling (~135 effective frames) to finish in reasonable time. 20,000 iterations, 40:09, final mesh 3,580,410 verts / 6,204,034 faces. **Visual verdict: disaster** — good numbers, bad geometry. Leading suspect is the 4x frame subsampling starving the octree's per-cell training views, not the method itself; not yet re-tested at every_frame=1 with the O(n²) fix applied instead of worked around | Phase 6 candidate, currently failing | run — `~/work/SHINE_mapping` (WSL only) |
| [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) | 3DGS trainer (reads COLMAP directly, not tested against our data yet — just the build). **Built successfully** after three separate real upstream linker bugs, in order: (1) GNU `ld`'s reloc-sort limitation on this large a binary — fixed by switching to `-fuse-ld=lld`; (2) WSL inheriting the full Windows `PATH` let CMake's `find_package(Tiff)` pick up a Windows Anaconda `tiff.dll` instead of vcpkg's own — fixed with a clean Linux-only `PATH` + `-UTiff_DIR`; (3) `lld`'s stricter undefined-symbol handling exposed a real gap where `libstdc++exp`'s C++23 `<stacktrace>` symbols never made it into `liblfs_core.so` — patched `src/core/CMakeLists.txt` to force `$<LINK_LIBRARY:WHOLE_ARCHIVE,stdc++exp>`. Executable builds and runs; one loose end: the optional `stubgen.py` Python-typing post-build step still hits the stacktrace symbol via its own strict `dlopen`, even though the real executable doesn't — root cause not found (isolated repro of the same symbol+flags works fine, so something specific to this ~100-library link graph shadows it) and deliberately not chased further since it's cosmetic (IDE autocomplete only) | fusion candidate (not yet run) | **built** — `~/work/LichtFeld-Studio/build/LichtFeld-Studio` (WSL only; not built on Win) |
| [ethz-asl/voxblox](https://github.com/ethz-asl/voxblox) | Classical TSDF baseline from the original tooling survey | not yet run | cloned, untested — `work/voxblox` |

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

**Licensing.** Personal/non-commercial project as of 2026-08-24 — the earlier
commercial-use premise (which ruled out ScanNet++) no longer applies, reopening
NC-licensed datasets like ETH3D (CC BY-NC-SA 3.0) and ScanNet++. ICL-NUIM is
CC BY 3.0 regardless. ARKitScenes grants commercial rights under 700M MAU, and
its README confirms the licence covers the dataset, not just the code. Your own
captures are unencumbered. Not legal advice — worth a review if commercial
intent ever returns.

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

**More WSL2 gotchas, from the LichtFeld-Studio build specifically:**

- **Two clones, one build.** `work/LichtFeld-Studio` (Windows) and
  `~/work/LichtFeld-Studio` (WSL) are separate checkouts. Editing the Windows
  one (natural, since Read/Edit tools default there) does nothing for a build
  that only ever runs against the WSL copy — a source fix has to land in
  *both*, or the build silently keeps using the unpatched file. Cost a full
  extra rebuild cycle here.
- **`ninja` does not reliably auto-reconfigure** on a `CMakeLists.txt` change
  in this setup. Don't trust `cmake --build` alone after a source edit —
  explicitly re-run `cmake -B build` first, or a stale `build.ninja` silently
  keeps the old link command.
- **`sudo` needs a real terminal.** Any `apt-get install` has to be run by the
  user directly in their own WSL terminal (has a cached credential); an
  automated non-interactive shell has no way to answer the password prompt and
  just hangs until killed.
- **Large binaries can exceed GNU `ld`'s reloc-sort capacity.** `unable to sort
  relocs - they are of an unknown size` on a big shared library (CUDA + USD +
  LibTorch + Vulkan all in one `.so`) is a known `bfd`-linker limitation, not
  corruption — switch to `-fuse-ld=lld`.
- **WSL inherits the whole Windows `PATH`.** CMake's `find_package` can pick up
  a Windows-side library over the Linux one you actually want (here: Anaconda's
  `tiff.dll` instead of vcpkg's `libtiff.a`) — silently, since both satisfy the
  same `find_package` call. Strip `/mnt/c/...` entries from `PATH` before
  configuring anything build-critical, and `-U<Var>_DIR` to force CMake to
  re-search once the leak is fixed.
