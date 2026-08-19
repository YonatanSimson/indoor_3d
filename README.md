# indoor_3d

Offline indoor 3D reconstruction from ARKit captures. Phase plan lives in
[hand_off.md](hand_off.md); this file records what is built and what was learned
building it.

Self-contained by design: nothing here imports from `ARKit3d` or `ARKitScenes`.

## Status

| Phase | State |
|---|---|
| 0 — capture | done, by the `model_3d` iOS app (external) |
| 1 — custom TSDF fusion | **done** |
| 2 — pose refinement (COLMAP) | **done**; refined poses feed back into fusion |
| 3 — depth upsampling | **done**; solver for interpolation, DA3 for holes |
| 4 — dense MVS | not started |
| 5 — surface & texture | not started |

## Setup

Three environments, deliberately separate:

```powershell
conda activate indoor3d   # fusion + evaluation: open3d, opencv, scipy, scikit-image
conda activate colmap     # CUDA COLMAP binary, invoked as a subprocess
conda activate depth      # torch cu128 + Depth Anything 3
```

COLMAP is kept apart so its Qt stack cannot collide with Open3D's, and `depth`
is kept apart because installing DA3 pins numpy back to 1.26 and pulls its own
open3d, which would disturb the fusion environment.

### GPU note (Blackwell)

The RTX 5070 Ti is compute capability 12.0 (`sm_120`). Stock torch builds on
cu118/cu124 enumerate the device and then fail at kernel launch. The working
combination is:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

which gives torch 2.11.0+cu128 with `sm_120` in its arch list. Verified by
actually running a matmul, not just by `torch.cuda.is_available()` -- the
earlier builds passed that check and still could not execute.

## Usage

```powershell
# ARKitScenes-format capture
python scripts/run_fusion.py --voxel 0.01 --out output/scene.ply

# a RawRecorder session from the model_3d app
python scripts/run_fusion_raw.py --voxel 0.015 --out output/myroom.ply

# Phase 2: refine poses with COLMAP, seeded from ARKit
python scripts/run_colmap_raw.py --matcher exhaustive

# quality against ARKit's own mesh, and viewers
python scripts/compare_meshes.py --meshes output/scene.ply
python scripts/view_mesh.py output/a.ply output/b.ply
python scripts/browse_frames.py          # page through raw RGB / depth / confidence
```

Two diagnostics, both of which answer "are the poses being interpreted right?":
`scripts/check_convention.py` (needs a reference mesh) and
`run_fusion_raw.py --compare-conventions` (does not).

## Phase 1 results

ARKitScenes scene `42444946`, 925 frames, measured against the on-device
`ARMeshAnchor` mesh. Error is the distance from raw LiDAR measurements to the
reconstructed surface.

| mesh | verts | tris | edge mm | median err | mean err | <2 cm |
|---|---|---|---|---|---|---|
| ARKit (on-device) | 1.08M | 2.00M | 14.6 | 11.1 mm | 45.2 mm | 67.1% |
| ours @ 2 cm | 0.86M | 1.62M | 19.4 | 9.4 mm | 23.1 mm | 72.8% |
| ours @ 1 cm | 4.79M | 9.04M | 9.6 | **8.4 mm** | **17.4 mm** | **77.5%** |

Mean error improves 2.6x. Mean sitting far above median in the ARKit row is the
signature of the artefacts this phase targets: flying pixels and smeared edges
throwing a long tail of badly-placed geometry.

### What "error" means here, and what it does not

There is **no ground truth** in any number above. Error is the distance from the
raw LiDAR measurements to the reconstructed surface, so it measures disagreement
with the sensor, not with reality, and the sensor's own noise sets the floor.
ARKitScenes does ship Faro laser scans (`--download_laser_scanner_point_cloud`,
raw dataset only) -- that would be true ground truth, and is not used yet.

The table above is also mildly circular: those evaluation frames were fused into
our meshes, though not into ARKit's. Re-running with disjoint frames (fuse on
even, evaluate on odd) shows the bias is small and the conclusion survives:

| mesh | median | mean | <2 cm |
|---|---|---|---|
| ARKit (on-device) | 11.6 mm | 50.3 mm | 65.8% |
| ours @ 2 cm, holdout | 9.9 mm | 22.2 mm | 72.2% |
| ours @ 1 cm, holdout | **8.8 mm** | **17.2 mm** | **75.9%** |

Mean improves 2.9x under the fair split, slightly *better* than the circular
figure. TSDF fusion averages hundreds of overlapping views, so no single frame
has much influence to memorise.

Caveat on that split: at 10 fps consecutive frames see almost the same geometry,
so odd/even holdout rules out gross memorisation but not shared-viewpoint bias.
A genuinely independent test needs the laser scans.

#### Laser ground truth: attempted, blocked

One Faro scan for visit 421337 was downloaded (`scripts/fetch_laser_scan.py`,
1.9 GB) and it does not drop straight in:

| | extent | footprint | frame |
|---|---|---|---|
| laser scan `170078` | 31 x 50 x 12 m | 1574 m2 | site-level, z ~ 229 m |
| scene `42444946` | 9 x 10 x 4.4 m | 95 m2 | room-local, z ~ 0 |

The scan covers the whole building, our scene is one room inside it (~6%), and
**the dataset provides no laser-to-ARKit transform** — `_pose.txt` registers the
FARO scans to each other, not to the capture. Blind global registration
(RANSAC on FPFH, then ICP) reaches only 0.076 fitness, so the accuracy numbers
it produces are meaningless and are not reported here.

Making this work needs coarse localisation of the room within the building scan
first — a real task, not a flag. `scripts/eval_vs_laser.py` is written and
correct in structure; it prints registration fitness precisely so a bad
alignment cannot be mistaken for a bad reconstruction.

Reproduce with:

```powershell
python scripts/run_fusion.py --voxel 0.01 --stride 2 --offset 0 --out output/holdout_v10mm.ply
python scripts/compare_meshes.py --holdout 2 --meshes output/holdout_v10mm.ply
```

Note that at 2 cm the mesh is *coarser* than ARKit's yet still more accurate --
ARKit's tessellation is fine but a good part of that detail is noise.

## Phase 3 results

LiDAR depth is 256x192 while RGB is 1920x1440 -- a 56x gap in pixel count, and
all geometry came from the smaller grid. Guided upsampling cannot invent detail
the sensor never measured, but it can put depth edges where the image says the
object boundary is, instead of where a coarse grid guessed.

Measured over 30 frames at 3x (`scripts/eval_upsampling.py`):

| method | alignment px | edge sharpness | flat noise | sharp/noise | filled |
|---|---|---|---|---|---|
| bilinear | 6.02 | 0.2875 | 0.0629 | 4.6 | 76.3% |
| joint bilateral | 4.04 | 0.2893 | 0.0540 | 5.4 | 76.3% |
| **fast bilateral solver** | **1.82** | **0.6011** | **0.0433** | **13.9** | **82.6%** |

The solver wins on every axis: edges land 3.3x closer to the true image
boundary, are 2.1x sharper, and flat regions are 31% *less* noisy. Sharpening
usually costs noise, so getting both is the signature of a global solve that
uses ARKit's confidence to reject bad samples, rather than a local filter
averaging its neighbours.

Enable with `--upsample solver --upsample-scale 3`. It is off by default because
it triples fusion time (52s to 162s over 539 frames).

### Choosing the metric matters here

The first numbers looked at were vertex count and surface area, and they are
useless for this: the solver adds a million vertices and 36 m2 mostly by filling
holes, which says nothing about whether any edge moved to the right place. They
made the solver look suspicious -- possibly inventing surface -- when it was in
fact the best method by a wide margin.

Edge alignment had to be measured directly: threshold the depth gradient, run
Canny on the guide image, and take the distance from each depth edge to the
nearest image edge. Alignment alone is still not enough, since a method that
outputs no edges scores perfectly, so sharpness and flat-region noise are
reported alongside it.

## Depth Anything 3, anchored to LiDAR

A monocular network works at image resolution, so in principle it sees detail
the 256x192 LiDAR cannot. Two things had to be established: whether its depth is
trustworthy, and whether it beats simply interpolating the LiDAR we already
have.

### Is it metric? No.

`DA3METRIC-LARGE` disagrees with LiDAR by 111 mm, and needs a per-frame rescale
between **0.92x and 1.67x** on frames of the same room. It is not merely
miscalibrated by a constant -- it is not self-consistent between frames, so
fusing it unanchored would place the same wall in different spots.

Removing a per-frame affine fit recovers most but not all of it:

| prediction | raw | after affine | correlation | scale range |
|---|---|---|---|---|
| DA3METRIC-LARGE (monocular) | 111.3 mm | **42.3 mm** | **0.983** | 0.92-1.67 |
| DA3-LARGE (pose-conditioned) | **88.2 mm** | 60.0 mm | 0.975 | 0.88-1.69 |

Correlation of 0.98 means the shape is broadly right, but 42-60 mm of residual
is real shape error, not noise -- four times the LiDAR's own agreement with the
fused mesh.

Note the inversion: pose conditioning improves *absolute* placement and worsens
*shape*. Since anchoring removes scale anyway, the "after affine" column is the
one that governs, so the monocular checkpoint is the better input here despite
failing the metric claim that motivated choosing it. Pose conditioning also
needs a multi-view checkpoint at all: the metric variants have no camera encoder
(`cam_enc is None`) and fail deep in the forward pass.

### Does it beat interpolation? Only in holes.

| | near a sample | in a gap |
|---|---|---|
| bilateral solver, extrapolating | -- | 83.2 mm |
| DA3, anchored | -- | **44.4 mm** |

At 256x192 the LiDAR is dense enough that interpolating between samples beats
predicting from scratch. The network earns its place only where there is nothing
to interpolate: **2x better in contiguous holes**, which is exactly where
confidence masking removes the glossy, dark and distant readings.

Getting that answer required fixing the experiment twice. Hiding 25% of samples
*at random* leaves a survivor within a pixel or two of everything, so no gap
ever forms and interpolation wins trivially; real holes are contiguous, so the
holdout has to remove blocks. And the first comparison was rigged: the solver
masks its output to a dilation of its support, so it declined to fill holes at
all and scored no data rather than a fair error.

## Camera conventions

The single biggest source of silent, catastrophic error here. A wrong
convention still produces a plausible-looking point cloud, just mirrored, so
nothing fails loudly. **The two data sources this repo reads disagree**, and
each was verified against data rather than assumed:

| | ARKitScenes `.traj` | model_3d `RawRecorder` |
|---|---|---|
| stored direction | world → camera | camera → world |
| camera axes | **RDF** (OpenCV) | **RUB** (ARKit native) |
| serialisation | one line per pose | per-frame JSON, **column-major** |
| into Open3D | pass through unchanged | `RUB_TO_RDF @ inv(T)` |

Apple converted to RDF when publishing ARKitScenes; a raw
`ARCamera.transform` is RUB. Evidence: for ARKitScenes, RDF back-projects
11 mm from the reference mesh versus 332 mm for RUB. For a RawRecorder session,
RUB fuses to 100 m² of surface at 17.5 mm mean error while RDF scatters into
251 m² at 68.5 mm.

### RawRecorder gotchas

Reading that format correctly needs four corrections, not one:

1. **Transposed matrices.** `simd_float4x4` columns are written as JSON rows,
   so a naive read gives the transpose. Both `transform_matrix` and
   `intrinsic_matrix` are affected. `position` is a built-in check: it matches
   row 3 of the raw array, where row-major would keep translation in column 3.
2. **Direction.** `transform_matrix` is camera-to-world; Open3D wants the
   inverse.
3. **Axes.** RUB, per above.
4. **Intrinsics resolution.** They describe the RGB frame (1920x1440) while
   depth is 256x192 — a 7.5x scale that must be applied before the intrinsics
   touch the depth grid.

`indoor3d/data_rawrecording.py` handles all four and asserts (1) per frame.

## Phase 2 notes

COLMAP is seeded with the ARKit poses rather than run as plain SfM, so the
result keeps metric scale and gravity alignment. Intrinsics are held **fixed**
at ARKit's per-frame values: autofocus moves the focal length ~3% across a
session, so cameras are per-image, and with one camera per image there is
nothing to calibrate — refining focal length would only let pose error hide in
it.

Bundle adjustment leaves seven degrees of freedom free (rotation, translation,
scale), so raw refined poses sit in an arbitrary gauge — up to 3.8x off metric.
`align_to_arkit` removes that with a similarity transform (scikit-image's
`SimilarityTransform`, i.e. Umeyama) and reports the residual, which is the
genuine drift correction. Note that scale carries no quality signal at all: the
run that landed *further* off metric was the more accurate one.

Matcher choice dominates the result, on a 539-frame handheld scan:

| | sequential | exhaustive |
|---|---|---|
| BA termination | `NO_CONVERGENCE` | `CONVERGENCE` |
| median correction | 47.4 mm | **7.5 mm** |
| max correction | 85.3 mm | **33.5 mm** |

Re-fusing at 15 mm with the refined poses, measured against raw LiDAR:

| | ARKit poses | refined |
|---|---|---|
| median error | 11.04 mm | **10.10 mm** |
| mean error | 18.06 mm | **16.90 mm** |
| within 2 cm | 70.6% | **73.4%** |
| surface area | 132.6 m² | **127.6 m²** |

### Pose priors: tried, and worse

COLMAP can take ARKit positions as priors in the database (`pose_prior_mapper`,
`prior_position_std_*`) so the gauge is constrained *during* optimisation
instead of afterwards. That part works. The reconstruction it produces does not:

| poses | median | mean | <2 cm | area m2 |
|---|---|---|---|---|
| ARKit, no refinement | 11.04 mm | 18.06 mm | 70.6% | 132.6 |
| COLMAP seed + align | **10.10 mm** | **16.90 mm** | **73.4%** | **127.6** |
| COLMAP pose priors | 70.41 mm | 89.60 mm | 19.5% | 209.4 |

Tightening the prior was tested too, since a loose one is a weak pull:

| poses | median | mean | <2 cm | area m2 | scale error |
|---|---|---|---|---|---|
| ARKit, no refinement | 11.04 mm | 18.06 mm | 70.6% | 132.6 | — |
| COLMAP seed + align | **10.10 mm** | **16.90 mm** | **73.4%** | **127.6** | 73.8% |
| priors, sigma 100 mm | 70.41 mm | 89.60 mm | 19.5% | 209.4 | 6.4% |
| priors, sigma 20 mm | 47.22 mm | 70.41 mm | 26.5% | 226.5 | **0.85%** |

Priors win their intended fight outright: scale error falls to under 1% and
gauge float to 0.14 m, against 74% and 3.4 m for the seed path. They still lose
the reconstruction, and tightening sigma only halves the gap.

The reason is that **position priors do not constrain rotation**.
`pose_prior_mapper` pulls camera centres toward ARKit's but rebuilds orientation
from scratch by incremental SfM, and orientation is what a fused surface is
most sensitive to -- 1 degree of error at 2 m displaces geometry by 35 mm, far
more than the equivalent position error. ARKit's orientation is its strongest
output, being gravity-aligned and IMU-corrected, and this path discards it.
Seeding keeps all six degrees of freedom and only refines them.

So `--mode seed` (the default) is right whenever an ARKit trajectory exists.
Priors are the tool for the other case: metric scale with no trajectory to align
against.

Also note how easily this went unnoticed: the mappers write into numbered
subdirectories (`sparse/refined/0/`), so reading the parent picked up the
previous run's converted model and reported the seed result under the priors
label. `resolve_model_dir` now prefers sub-models over stale parents.

The gain is real but small, and that is the expected result rather than a
disappointment: a 7.5 mm median correction means ARKit's VIO was already
close to right over a 24 m walk in one room. Phase 2 earns its keep on long or
looping trajectories, where drift accumulates into double walls. Less surface
area for fewer vertices is the tell that duplicated shells collapsed together.

Two traps worth remembering, both of which produced confident nonsense:

- **`camera_centre` needs an orthonormal rotation.** Applying a scaled
  similarity to a pose leaves the linear block non-orthonormal, and `-R^T t`
  then returns garbage — this reported an 8.5 m "drift" that did not exist.
  The aligned poses are rescaled back to orthonormal, and centres are
  cross-checked by transforming them directly.
- **`images.txt` has two lines per image and the second may be empty.** Parsing
  by alternation while skipping blank lines silently drops every other pose
  (270 of 539). Pose lines are identified by shape instead.

Matching matters more than it looks: `sequential` alone gives no loop closure,
which is the entire point of this phase, and leaves the chain weakly enough
constrained that bundle adjustment reports `NO_CONVERGENCE`. Use
`--matcher exhaustive` (O(n²) but fine at ~500 images) or supply a vocabulary
tree.

## Depth cleanup

Applied per frame before integration, since fusion cannot undo these later:

- **confidence gating** — drop ARKit confidence level 0. Only active on
  RawRecorder captures; the ARKitScenes sample data ships no confidence maps.
  It matters: ARKit reports *every* depth pixel as populated, inpainting
  unmeasured regions, so without gating that invented depth is fused as fact.
- **flying-pixel rejection** — drop pixels whose local depth range exceeds a
  distance-scaled threshold. These sit on foreground/background boundaries and
  smear silhouettes into the background. ~2.6% of pixels on ARKitScenes, ~9% on
  the handheld capture.
- **bilateral smoothing** — mask-aware, so holes stay holes. `cv2.bilateralFilter`
  is deliberately not used: it has no notion of invalid pixels and lets zero
  holes bleed into neighbouring surfaces, dragging real depth toward the camera.

Order matters: discontinuities are detected on raw depth, because smoothing
first would soften the very steps used to find them.
