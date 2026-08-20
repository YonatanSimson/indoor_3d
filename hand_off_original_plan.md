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
| **V — validation on ground truth** | **Stages A & B done.** Pose accuracy confirmed as the ceiling; 10mm voxel is as good as 5mm against real geometry; **anchored confirmed the best upsampler** against true depth; **9.8mm vs published SOTA's 7-9mm** on the same `lr kt2` benchmark sequence (GT poses, isolating fusion quality); COLMAP-on-ICL-NUIM inconclusive for COLMAP-vs-ARKit specifically (too little synthetic-scene texture to lock onto, not a real accuracy verdict). See below |

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

## Validate against ground truth (phase V)

Everything above rests on a metric shown to be unreliable, and on visual
judgement that does not scale. The fix is a dataset where truth is known.

**Data:** `~/data/iclnuim/` — ICL-NUIM living room trajectory 2 (880 posed
frames), **CC BY 3.0, commercial use permitted with attribution**. It ships
what nothing else here has:

- `living-room.ply` — ground-truth surface, ~10M points (points + normals,
  **no faces** — read it as a point cloud, not a mesh; `o3d.io.read_triangle_mesh`
  silently reports zero triangles on it)
- `livingRoom2n.gt.sim` — the ground-truth poses that actually work (see below)
- `livingRoom2.gt.freiburg` — present, but **do not use it** (see below)
- `rgb/` + `depth/` at 640x480, clean (noisy variants also downloadable)

**Simulate our sensor** by degrading it: 640x480 to 256x192, depth-dependent
noise, and a synthetic confidence map — confidence gating does real work in this
pipeline, and a simulation without it would not exercise that path.

### Stage A: done — the pose-ceiling experiment has an answer

Built `indoor3d/data_iclnuim.py` (loader), `indoor3d/sensor_sim.py` (sensor
degradation), `indoor3d/pose_perturbation.py`, `scripts/check_convention_iclnuim.py`,
and `scripts/eval_iclnuim_pose_ceiling.py`. Stage B (re-deriving the metric
questions, plus real pose/rotation accuracy) is done too — see below.

**A second convention trap, worse than the known one.** `livingRoom2.gt.freiburg`
looks exactly like the ground-truth trajectory — it's the file this hand-off
originally pointed at — and it is not usable: back-projecting it against
`living-room.ply` lands 20-50+ cm off under *every* axes/direction/fy-sign
hypothesis tested, and a full RANSAC+ICP registration search doesn't rescue it
either. Its first pose is exactly identity, which suggests it's expressed
relative to frame 1 rather than in the mesh's world frame — but that isn't
documented anywhere shipped with the data, so it's left alone rather than
guessed at further. `livingRoom2n.gt.sim` (880 blank-line-separated 3x4 `[R|t]`
blocks, ICL-NUIM's own native format) is what actually works: used directly as
camera-to-world with **no** change of basis, it back-projects to a **14.7 mm
median** against the ground-truth cloud — confirmed formally by
`scripts/check_convention_iclnuim.py` at **27 mm** median (coarser sampling,
same verdict), against 92 mm+ for every other hypothesis. The one *known* trap
from this doc's original text — ICL-NUIM's negative focal length — held up:
`fy = -480.00` is confirmed the correct sign, same as `.gt.sim`'s rotation
needing no RUB/RDF change of basis. `scripts/check_convention.py`'s pattern
(sweep hypotheses, measure against the reference, let the data decide) is what
caught this — trusting the hand-off doc's own file listing would not have.

**The sensor simulator needed one recalibration.** A first pass at
`indoor3d/sensor_sim.py`'s depth-dependent noise (`sigma = 0.001 + 0.004·z²`)
put sigma at 17-64 mm across this room's 1-4 m depth range — large enough that
the simulated sensor's own noise dominated the experiment regardless of pose
accuracy, and 83% of pixels graded low-confidence. Recalibrated to
`0.0015 + 0.0007·z²` (single-digit mm at typical range, ~13 mm at 4 m) and the
confidence-edge detection switched from a bespoke gradient check to reusing
`depth_filters.flying_pixel_mask` directly (the pipeline's own edge notion, at
two thresholds) — a meaningful minority (17% level 0, 25% level 1) rather than
almost everything.

**The result** (`scripts/eval_iclnuim_pose_ceiling.py`, 880 frames, 5 mm voxel,
simulated-sensor depth identical in both runs):

| poses | accuracy median | accuracy mean | <5cm coverage |
|---|---|---|---|
| ground truth | **10.4 mm** | 40.4 mm | 27.7% |
| perturbed ±7.5mm | 15.4 mm | 41.0 mm | 28.2% |

GT poses land 1.48x more accurate than the perturbed run. **Pose accuracy is
the ceiling** — Phase 2 (pose refinement) is where effort belongs, and finer
voxels would help once poses improve. This also gives the project its first
non-circular accuracy figure: 10.4 mm median against *true* geometry, not
against the same LiDAR that built the mesh. Mean sitting well above median
(40 mm vs 10 mm) is the same long-tail signature the Phase 1 ARKit comparison
showed (11 mm median, 45 mm mean) — a minority of badly-placed geometry, not a
uniformly-noisy surface. Coverage at ~28% reflects the camera trajectory
covering a fraction of the room's full extent, the same "scan sees areas the
capture never visited" caveat as the ARKitScenes laser-scan attempt.

### Stage B: re-derived metrics, plus pose/rotation accuracy

**1. Does 5 mm actually resolve noise?** No further than 10 mm does.
`scripts/eval_iclnuim_voxel_sweep.py`, GT poses, simulated-sensor depth:

| voxel | accuracy median | accuracy mean | <5cm coverage | old-metric median |
|---|---|---|---|---|
| 15 mm | 10.5 mm | 43.3 mm | 27.4% | 1.9 mm |
| **10 mm** | **9.8 mm** | 42.4 mm | 27.6% | 1.9 mm |
| 5 mm | 10.4 mm | 40.4 mm | 27.7% | 1.8 mm |

All three land within 0.7 mm of each other against real geometry — voxel size
stopped mattering once poses are this good, exactly as the pose-ceiling
result predicts (the ceiling is pose accuracy, not voxel resolution).

**3. Does the old metric still mislead?** Yes, the same way, again: ranked by
ground truth the order is 10/5/15 mm; ranked by the old raw-LiDAR-distance
proxy it's 5/10/15 mm. Same disagreement this doc already documented twice
on the real capture, now reproduced against true geometry rather than
inferred from visual judgement. Surface area was not re-tested as a
candidate metric this round (Stage B question 3's second half) — the voxel
range here is too narrow for it to say much; worth revisiting with a wider
sweep.

**2. Does anchored upsampling recover real structure?**
`scripts/eval_iclnuim_upsampling.py` degrades the true 640x480 depth to our
256x192 sensor, upsamples back, and compares against the true depth that was
never shown to any method:

| method | median | mean | p90 | coverage |
|---|---|---|---|---|
| **bilinear** | **2.7 mm** | 8.1 mm | 10.5 mm | 100.0% |
| joint | 2.0 mm | 9.9 mm | 22.2 mm | 100.0% |
| solver | 19.6 mm | 50.8 mm | 131.5 mm | 92.5% |
| anchored | 5.4 mm | 12.7 mm | 19.4 mm | 92.5% |

That first pass was too easy on bilinear/joint: `degrade_frame`'s coarse
depth is fully populated (100% coverage) because it only creates holes from
insufficient downsampling support, whereas real fusion runs `clean_depth`
first, which drops flying pixels and low-confidence pixels *before*
upsampling ever sees the depth (`fusion.fuse`'s own order). Re-run with that
step added (`clean_depth`, same as `fuse`, before `upsample`):

| method | median | mean | p90 | coverage |
|---|---|---|---|---|
| bilinear | 1.7 mm | **32.0 mm** | 8.4 mm | 84.1% |
| joint | 2.1 mm | 28.8 mm | 65.5 mm | 84.1% |
| solver | 27.9 mm | 104.8 mm | 219.6 mm | 92.3% |
| anchored | 3.2 mm | **24.0 mm** | 18.3 mm | 92.3% |

Coverage is not a side note here, it's the point: bilinear and joint cannot
fill a hole at all (`upsample_bilinear`'s own docstring: "cannot invent
depth where there is none"), so 15.9% of true-depth pixels get **no
prediction whatsoever** — real holes in the fused mesh unless another frame
happens to cover the same spot. Median still favours bilinear on the pixels
it *did* fill, but mean tells the opposite story (32.0 mm vs anchored's
24.0 mm): bilinear drags a stale value across real gaps at hole boundaries,
producing a long tail of severe errors that its low median hides. This
confirms Phase 3's original verdict rather than reversing it: **anchored is
the best upsampler**, the only one here that is both accurate on filled
pixels and actually fills most of the frame. Solver alone remains clearly
the worst choice on every axis, and plain bilinear/joint's apparent win in
the first (buggy) pass was an artifact of not feeding them real coverage
gaps to fail on. `--upsample anchored` stays the right default.

**4. Reframed: not "does the 2.9x ARKit gain survive" but "how do we compare
to published SOTA on this exact benchmark?"** An ARKit-equivalent baseline
would only ever be a proxy — ARKit can't run on synthetic frames — and
ICL-NUIM living room already has a real answer: this is `lr kt2`, a
standard SLAM/reconstruction benchmark sequence, with a decade of published
surface-reconstruction-accuracy numbers to compare against directly instead.

From "Robust Keyframe-based Dense SLAM with an RGB-D Camera" (Le & Kim,
[arXiv:1711.05166](https://arxiv.org/abs/1711.05166), Table V), `kt2`
surface accuracy vs. ground truth:

| method | kt2 accuracy |
|---|---|
| BundleFusion | 0.7 cm |
| ElasticFusion | 0.8 cm |
| Kintinuous | 0.9 cm |
| RKD-SLAM | 1.6–1.8 cm |
| RGB-D SLAM | 3.1 cm |
| DVO-SLAM | 11.9 cm |

Our numbers on the same sequence (`eval_iclnuim_voxel_sweep.py`, GT poses):
**9.8 mm (0.98 cm)** at 10 mm voxel, 10.4 mm at 5 mm voxel. That lands
between Kintinuous and RKD-SLAM — respectable, not SOTA, comfortably ahead
of RGB-D-SLAM and DVO-SLAM. With poses perturbed to realistic ~7.5 mm
uncertainty instead of ground truth: 15.4 mm, still same order of magnitude
as the published numbers rather than off by 10x.

GT poses here are a deliberate, correct choice, not a shortcut being
glossed over: pose accuracy already has its own answer from the
pose-ceiling and COLMAP experiments above, so using GT poses for *this*
comparison isolates exactly what's left to check — fusion and depth-cleaning
quality on their own, decoupled from pose estimation. It does mean this
number cannot be compared to the classical systems' *end-to-end* SLAM
pipelines without that caveat: they tracked their own trajectory in real
time and got their poses wrong by some amount our GT-pose run does not pay
for. Whether Table V's figures are median, mean, or RMSE is not confirmed
across sources, so treat 9.8 mm vs. 7–9 mm as a solid ballpark rather than
an exact apples-to-apples number.

### Pose and rotation accuracy: the real COLMAP pipeline on ICL-NUIM

Beyond Stage B's original four questions: does COLMAP's *actual* pose
refinement — not a hypothetical perturbation — recover ground truth?
ICL-NUIM has no onboard VIO of its own, so `scripts/run_colmap_iclnuim.py`
seeds COLMAP the same way a real capture does: perturb the true poses by the
same ~7.5 mm Phase 2 measured between ARKit and COLMAP, treat that as "an
iPhone's approximate, drifted pose," then run the identical
seed/triangulate/bundle-adjust pipeline `run_colmap_raw.py` uses. Metrics are
`evo` (github.com/MichaelGrupp/evo), not hand-rolled: APE (Absolute Pose
Error, i.e. ATE) after Sim3 alignment, and RPE (Relative Pose Error) at
1-frame delta, both for translation and rotation.

220 frames (stride 4), exhaustive matching (sequential hit `NO_CONVERGENCE`
here too — same lesson as the real capture):

| metric | median | mean | rmse |
|---|---|---|---|
| APE translation | 45.0 mm | 58.0 mm | 78.5 mm |
| APE rotation | 1.65° | 1.81° | 2.08° |
| RPE translation | 30.7 mm/frame | 59.1 mm/frame | 93.5 mm/frame |
| RPE rotation | 1.00°/frame | 1.49°/frame | 2.00°/frame |

**COLMAP made the poses worse here**, not better — the seed started at
7.5 mm and refinement moved it to a 45 mm median. Bundle adjustment still
reported `NO_CONVERGENCE` even with exhaustive matching (Residuals: 21114,
Parameters: 9767, terminated at 101 iterations), and per-image track counts
were thin and uneven (many images saw well under 100 of several hundred
candidate points).

**This is not a usable verdict on COLMAP vs. ARKit, and shouldn't be read as
one.** ICL-NUIM's synthetic room has too little exploitable texture for
feature-based SfM to lock onto reliably — flat, low-detail rendered
surfaces starve ORB/SIFT-style matching of the corner-like features it
needs, a known general weakness of synthetic scenes for this class of
method, and exactly what `NO_CONVERGENCE` plus thin per-image track counts
are showing here. The real capture's texture-rich photos are a different
regime entirely, where Phase 2 measured COLMAP genuinely *correcting* drift
(7.5 mm median correction). Conclusion: **this experiment measures ICL-NUIM's
suitability as a COLMAP testbed, not COLMAP's accuracy** — it validates
fusion accuracy well (Stage A/B above) but is confounded for this one
question. Answering it for real would need either a more feature-rich
synthetic dataset or accepting that COLMAP-vs-ARKit can only be judged on
real captures.

**A convention trap distinct from the pose-source one above.** COLMAP's
PINHOLE model has no negative-fy support, so the seed pose cannot simply be
handed over as-is. The first attempt flipped one row of the seed rotation to
compensate — which turns a proper rotation (det +1) into a reflection
(det -1), not a valid camera orientation, and silently wrecked
triangulation (10 total bundle-adjustment residuals across 220 images,
instead of the ~21000 a working reconstruction produces). The fix is to
flip the *images* vertically instead and keep the pose untouched; ICL-NUIM's
principal point sits exactly at the image's vertical centre (239.5 of 480),
so `cy` needs no adjustment to match.

Reproduce with:

```powershell
conda activate indoor3d
python scripts/check_convention_iclnuim.py
python scripts/eval_iclnuim_pose_ceiling.py
python scripts/eval_iclnuim_voxel_sweep.py
python scripts/eval_iclnuim_upsampling.py
python scripts/run_colmap_iclnuim.py --matcher exhaustive
```

---

## Remaining forward plan

**Phase 5 — surface & texture.** The biggest perceived-quality jump, and
untouched. Poisson reconstruction from the fused oriented cloud, mesh cleanup
(floating fragments, decimation, hole filling), UV unwrap, project captured RGB
as texture, export OBJ/GLTF/USDZ. Note colour is currently per-vertex, so
texture detail is capped by geometry density — a texture map removes that link.

**Phase 6 — polish.**
- ARKit plane/semantic labels to flatten walls and floors (cheap
  regularisation, high visual impact).
- Neural implicit surfaces as a possible replacement for classical TSDF —
  see External Repos.
- **Live coverage feedback in `model_3d`** (the iOS capture app, separate
  repo). The one quality complaint about real output — a lumpy ball — turned
  out to be a *coverage* problem: matte surface, in a corner, seen from few
  angles at oblique incidence. Real-time observations-per-surface and
  incidence-angle tracking, shown to the operator while scanning, would let
  them fix it during capture instead of discovering it in post. Needs
  exploring `model_3d`'s architecture first (not started); an offline
  post-hoc version (reusing existing poses + raycasting against a completed
  mesh, no on-device work) is a cheaper Phase-V-adjacent precursor if the
  metric needs validating before committing to on-device engineering.

---

## External Repos

| Repo | Why | For | Status |
|---|---|---|---|
| [PRBonn/SHINE_mapping](https://github.com/PRBonn/SHINE_mapping) | Sparse octree-encoded neural implicit mapping (ICRA 2023) — beat VDBFusion on accuracy/completeness in recent benchmarks; candidate to replace/augment classical TSDF fusion | Phase 6 | cloned — `work/SHINE_mapping` |
| [ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) | Monocular/multi-view depth. Tested: metric claim does not hold indoors (scale 0.92-1.67 across frames of one room); ~4% shape error after affine fit. Useful as structure, not geometry | Phase 3 | cloned — `work/depth-anything-3` |

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
