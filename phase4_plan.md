# Phase 4 — Dense MVS: plan

Dense multi-view stereo is the step that can actually close the detail gap,
because it computes geometry from the 1920x1440 image stream rather than
interpolating 49k LiDAR samples. It is also the easiest step so far to get
quietly wrong: MVS fails by producing *plausible* geometry, not by crashing.

Measured facts about this capture that drive every decision below:

| | value |
|---|---|
| median scene depth | 1.09 m |
| baseline, consecutive frames | 4.2 cm (**2.19°** triangulation) |
| frames with <1 cm motion | 10.4% |
| sparse model gauge | 3.8x off metric (scale 0.262) |
| sparse model reprojection | 1.4e152 px mean (outlier observations) |
| GPU | RTX 5070 Ti, 16 GB |

## The three ways this goes wrong

**1. Low baseline -- real, but COLMAP already handles most of it.** Depth
uncertainty scales as `Z^2 / (f * baseline)`, so at 1.09 m a 4.2 cm baseline is
~21 mm per pixel of matching error. But no reference frame is stuck with its
nearest neighbour: `patch-match.cfg` defaults to `__auto__, 20`, selecting the
20 most-covisible source views and aggregating photometric cost across all of
them, and `filter_min_triangulation_angle` then discards pixels whose support is
poorly conditioned. Several narrow-baseline views that jointly span a wide one
triangulate fine.

*So subsampling is a compute decision, not a quality one*, and the two roles
must not be confused:

| | reference frames | source views |
|---|---|---|
| cost | one depth map each -- the whole runtime | reused, nearly free |
| baseline need | none | wide baseline wanted |
| action | subsample to ~150 | keep all 539 available |

*Mitigation:* thin the **reference** set (greedy, >= 15 cm apart) to cut runtime,
while leaving the sparse model complete so source selection keeps its
wide-baseline options. Raise `__auto__, 20` if the pilot shows poor
completeness.

**2. Gauge mismatch.** The sparse model on disk is in bundle adjustment's
arbitrary gauge -- 3.8x off metric. Only the exported `refined_poses.json` was
aligned back to ARKit. MVS output inherits whatever the model says, so fusing it
with a metric LiDAR TSDF would silently misplace everything by a factor.

*Mitigation:* recover the scale from **LiDAR** rather than from the ARKit
alignment: project the sparse points into the images, compare their predicted
depth against the LiDAR reading at those pixels, and fit one global scale. This
is an independent metric reference, whereas reusing the `align_to_arkit`
similarity would inherit any error in the ARKit trajectory -- and it
cross-checks that alignment for free. Verify camera centres against
`refined_poses.json` afterwards; the two should now agree.

**3. Textureless and repetitive surfaces.** Blank walls give MVS nothing to
match; carpet and tiling give it too many equally good matches. Both produce
confident, wrong depth rather than no depth.

*Mitigation:* LiDAR as an **outlier rejector** (see stage 3). This is the right
role for it -- COLMAP accepts only global `depth_min`/`depth_max`, so there is
no way to seed per-image priors regardless.

## Stages

### 0. Prepare a clean metric sparse model
- Recover scale from LiDAR: project the sparse points into the images they are
  observed in, read the LiDAR depth at those pixels, and fit one global scale by
  robust median ratio. Rescale the model by it.
- Filter outlier 3D points (one point at 130 m is dragging the mean
  reprojection error to 1e152; the bulk is sane at ~5.5 m).
- **Gates:** `model_analyzer` reports ~1 px mean reprojection error; the
  LiDAR-derived scale agrees with the `align_to_arkit` similarity to within a
  few percent (they are independent, so a large disagreement means one of them
  is wrong and must be resolved before continuing); camera centres then match
  `refined_poses.json`.

### 1. Reference-frame selection (for runtime)
- Greedy: keep a *reference* frame when its centre is >= `min_baseline`
  (default 15 cm) from the last kept one. This drops the 10.4% near-stationary
  frames for free.
- Leave the sparse model complete: source-view selection should still be able
  to reach every frame.
- **Gate:** ~130-180 references retained, and `patch-match.cfg` still lists 20
  source views for each.

### 2. Undistort
- `image_undistorter --max_image_size 1600`. Our cameras are `PINHOLE` with no
  distortion, so this mainly builds the workspace and fixes resolution.
- **Gate:** workspace contains one image and one camera per selected frame.

### 3. Patch-match stereo — pilot first
Run **30 frames** before committing hours. Settings that differ from default and
why:

| option | default | use | reason |
|---|---|---|---|
| `cache_size` | 32 (GB) | 8 | the default will thrash or OOM |
| `geom_consistency` | 1 | 1 | keep; it is the main quality lever |
| `min_triangulation_angle` | 1 | **30** | see below -- the single most important number |
| `filter_min_triangulation_angle` | 3 | 5 | sources are all >=30°, so this rarely binds |
| `depth_min` / `depth_max` | -1 | from LiDAR | the only prior COLMAP accepts |
| `max_image_size` | -1 | 1600 | runtime and memory |

- **Gate:** measure seconds/image, peak VRAM, completeness, and median
  |MVS - LiDAR|. Extrapolate before running the full set.

#### Why `min_triangulation_angle = 30`

This governs which views are even eligible as sources, and the default of 1°
admits pairs 1.9 cm apart at our ~1 m scene depth -- contributing ~47 mm of
depth noise. Raising it costs nothing here, because the scan revisits the room
and wide-baseline views are plentiful:

| angle | min baseline | depth err @1px | candidates/frame | worst-served frame |
|---|---|---|---|---|
| 1° (default) | 1.9 cm | 46.6 mm | 538 | -- |
| 20° | 36.3 cm | 2.18 mm | 490 | 444 |
| **30°** | **55.2 cm** | **1.43 mm** | **440** | **381** |
| 45° | 85.4 cm | 0.93 mm | 358 | 264 |

Even at 30° the worst-served reference still has 381 candidates, so
`__auto__, 20` is never starved.

**Why 20° is borderline and 30° is not.** The threshold is set against the
*median* depth (1.03 m), but the scene spans 0.63 m to 1.75 m, so one setting
means different things across it:

| setting | parallax at 0.63 m | parallax at 1.75 m |
|---|---|---|
| 20° | 32° | **12°** |
| 30° | 47° | **18°** |

Far surfaces bind. At a 20° setting, anything at 1.75 m gets only 12° of
parallax -- back in the noisy regime, and precisely for the far walls that
already carry the weakest texture. At 30° they get 18°. The price is 47° on
near surfaces, where perspective distortion starts to hurt NCC matching, but
near surfaces have both strong parallax and the best texture, so they can
absorb it.

Two further reasons this threshold matters more than the others:

- **Filter-then-rank targets the threshold.** Candidates below 20° are dropped,
  then survivors are ranked by shared point count, which favours the nearest of
  them. The selected sources therefore cluster just above 20°, instead of
  drifting to 60° extremes where perspective distortion and occlusion break NCC
  matching. The minimum behaves like a target.
- **It changes what success looks like.** At 30° the geometric depth limit is
  1.4 mm, well inside the anchored solver's 4.5 mm held-out error, so MVS can
  compete on accuracy and not only on edge placement. At the 1° default that
  headroom is buried under triangulation noise, and the comparison in stage 6
  would have been unwinnable for reasons having nothing to do with MVS itself.

If the pilot shows thin completeness on near surfaces -- the ones seeing 47° --
fall back to 25°, not to the default.

### 4. LiDAR outlier rejection
For each MVS depth map, compare against LiDAR upsampled to the MVS grid:
- reject where `|MVS - LiDAR| > max(5 cm, 5% of depth)` and LiDAR confidence is
  medium or high -- MVS is wrong there, since LiDAR does not fail on
  textureless surfaces;
- keep MVS unconditionally where LiDAR has no confident reading, which is
  exactly where MVS is supposed to earn its place;
- record the rejection rate per image. A high rate is a warning about that
  view's baseline, not just its texture.

### 5. Fusion
Integrate filtered MVS depth into the existing TSDF alongside LiDAR, with LiDAR
weighted higher where confident. `stereo_fusion` is the alternative but produces
a point cloud, which suits Phase 5 (Poisson) better than Phase 4.

### 6. Evaluation
- **Accuracy:** MVS never sees LiDAR, so every LiDAR sample is a held-out test.
  Report median |MVS - LiDAR| per image, and the distribution across views.
- **Detail:** edge alignment against image edges
  (`scripts/eval_upsampling.py`), versus the anchored-solver baseline.
- **Completeness:** share of pixels with surviving depth, split by whether
  LiDAR had confidence there -- this is what says whether MVS added anything.
- **Gate for keeping Phase 4 at all:** MVS must beat the anchored solver on
  edge alignment *without* losing on held-out accuracy. If it only matches,
  the LiDAR is already dense enough and the compute is not worth it.

## Runtime

539 frames at 1920x1440 with geometric consistency would be several hours. The
selection in stage 1 cuts it to ~130-180 frames, and `max_image_size 1600`
cuts it further. The stage-3 pilot exists to turn this estimate into a
measurement before committing.
