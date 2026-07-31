# Running the ConfCarti pipeline on OAIZIB-CM

This is the end-to-end procedure for the notebook, plus what was wrong with the
run in `ConfCarti_Full_Pipeline_error.ipynb` and what changed.

---

## 1. What actually went wrong

The notebook did not raise. It ran all 95 cells and wrote
`confcarti_results.csv`, four tables, a LaTeX table and a manifest. That is the
problem: two defects made every number meaningless without ever failing.

### 1.1 The cohort was never real

The configuration cell said:

```python
USE_REAL_DATA = True
OAIZIB_ROOT   = "D:\\ishtiaque\\Knee segementation\\OAI-ZIB\\OAIZIB-CM\\OAIZIB-CM"
```

and the very next cell ignored both. It fetched the real *metadata* from
CartiMorph (481 subjects, printed correctly), then did:

```python
knees, metadata = make_phantom_cohort(N_PHANTOM, seed=SEED)
```

`OAIZIB_ROOT` appears nowhere else in the notebook except inside a printed
string. Every downstream number — thickness, curvature, bulge, Dice, coverage,
reliability, all four tables — came from **32 analytic phantoms at
96 x 96 x 64 voxels**. The manifest then recorded:

```json
"used_real_data": true
```

because it read the flag rather than what the flag did.

Corroborating symptoms in the saved outputs:

| Symptom | Explanation |
|---|---|
| `phantom cohort: 32 knees, volume (96, 96, 64) @ 0.5 mm` | the cohort, stated plainly |
| lateral tibial cartilage Dice `0.000`, medial `0.001` | 6 epochs on a 32 x 32 x 24 mm crop of a phantom |
| `joint-centred crop ... clipped ~52% of the plate` on every case | `crop_size=(64, 64, 48)` is far too small |
| `curv_curvedness_median` = 0.032–0.035 flat across KL 0→4 | phantom geometry has no OA curvature signal |
| KL 4 has n=2, Mondrian groups of 3 | 32 subjects split 5 ways |

### 1.2 Curvature was thrown away on 76–87 % of every surface

Repeated through the log:

```
WARNING | curvature is not estimable on 80% of this surface: the patch is narrow
relative to the 6.0 mm boundary margin. Reported curvature summarises only the
834 interior vertices.
```

`compute_curvature` set curvature to NaN at every vertex within
`2 * radius_mm` = **6.0 mm** of an open boundary, and measured that distance
**through space** with a KD-tree.

Two things are wrong with that, and you were right that 6 mm is not valid here:

1. **A cartilage plate is an open sheet, not a closed surface.** The tibial
   plates are roughly 20–25 mm across. Eating a 6 mm rim from every edge of a
   22 x 24 mm patch removes about 80 % of it by area — reproduced exactly in
   `tests/test_curvature_open_patch.py`, where the old rule keeps 19.8 % of a
   plate-sized strip. What survived and got reported as "the compartment's
   curvature" was a small island in the middle of each plate, and it was not a
   consistent island across subjects, because plate shape changes with disease.

2. **Euclidean distance reaches around folds.** Two points on opposite banks of
   the trochlear groove, or across the intercondylar notch, are millimetres
   apart through space and centimetres apart along the cartilage. The margin
   therefore also deleted genuinely interior vertices.

The margin was aimed at something real — a quadric fitted on a one-sided,
clipped neighbourhood *is* badly biased — but it used a proxy (distance to a rim)
rather than the thing itself (is this neighbourhood two-sided?).

It also survived validation because every closed-form check in the notebook is
on a **sphere or a plane**: closed or unbounded surfaces where the boundary rule
never fires. The fixed notebook adds an open-patch check for exactly this reason.

### 1.3 Smaller defects fixed along the way

- **Smoothing ran before masking.** Three diffusion passes spread the biased rim
  fits ~1.5 mm inward, and *then* the mask was applied — so the retained interior
  carried contamination from the very vertices that were about to be deleted.
  The order is now reversed and the smoother preserves NaN.
- **The neighbourhood cap was binding and biased.** `max_neighbours=96` while a
  3 mm ball at 0.5 mm resolution holds ~113 vertices, and the subset was taken by
  `np.linspace` over the KD-tree's *unordered* output — an arbitrary slice that
  can leave a directional hole. Cap raised to 256 and subsampling made
  radius-stratified.
- **`cotangent_curvature` set `K = 0` on boundary vertices** rather than marking
  them undefined, silently reporting a saddle-free plane along every rim.
- **Crop size.** `(64, 64, 48)` at 0.5 mm is 32 x 32 x 24 mm; the log warned it
  clipped ~50 % of the plate on every case. A knee needs ~80 x 80 x 48 mm.
- **Whole-cohort materialisation.** `list(zip(knees, metadata.iterrows()))` is
  fine for 32 in-memory phantoms and fatal for 481 volumes on disk.
- **Risk control held every voxel.** ~2.5 M voxels x ~100 calibration knees as
  float64 is gigabytes.

---

## 2. The corrected curvature

`confcarti/thickness/curvature.py`. The blanket margin is replaced by a
**per-vertex support test** — asking directly whether each fit had the support to
mean anything:

| Test | Rejects |
|---|---|
| **Angular gap** — largest gap between neighbour azimuths in the tangent plane, default 120° | one-sided neighbourhoods, wherever they occur, including at interior holes a distance-to-rim rule never sees |
| **Neighbour count** ≥ 8 | slivers and isolated vertices |
| **Reciprocal condition number** ≥ 1e-4 on the (radius-normalised) 5x5 normal matrix | collinear or degenerate neighbourhoods |
| **Normal agreement** ≥ cos 60°, **out-of-plane offset** ≤ 0.6 r | candidates a KD-tree reached across the joint gap or around a fold |

Two further changes:

- **Geodesic, not Euclidean, boundary distance.** One multi-source Dijkstra over
  the mesh edge graph. Exposed as `geodesic_boundary_distance()` for diagnostics
  and stratification; `boundary_influence_zone(..., metric=...)` keeps the old
  Euclidean behaviour available for reproducing the previous run.
- **Adaptive scale.** A vertex that fails at 3.0 mm is retried at 2.0, then
  1.5 mm (`min_radius_mm`, ≈ 3 marching-cubes edge lengths — below that the fit
  measures voxel staircase, not anatomy). That recovers the plate rim as a
  real measurement at a stated scale instead of a NaN. Set
  `CURVATURE_ADAPTIVE = False` for a strict single-scale field.

Every summary now carries its own coverage: `curv_estimable_fraction`,
`curv_fit_radius_median_mm`, `curv_fit_radius_p05_mm`,
`curv_fit_radius_reduced_fraction`. **A curvature mean without an estimable
fraction beside it is not interpretable** — that is the lesson of the original
run, and it is now impossible to write one to the CSV without the other.

### Validated

`tests/test_curvature_open_patch.py` — 11 tests, all passing:

```
open strip R=22 mm, 22 x 24 mm (tibial-plate sized), true H = 1/(2R) = 0.022727
  support test  : 91.7% of the plate estimable, H = 0.02279  (+0.3%)
  old 6 mm rule : 19.8% of the plate estimable
  restored rim  : median bias < 15% on the outer 3 mm that the old rule deleted
open spherical cap R=25 mm : H and K within 10% / 20%
folded sheet (trochlea)    : both flat arms return H ~ 0 — the fit does not
                             reach across a 1.5 mm gap to a surface 20 mm away
closed sphere R=10         : H = 0.100, K = 0.0100 (unchanged from before)
```

Run them with:

```bash
python -m pytest tests/test_curvature_open_patch.py -q
```

Cost on a 21 k-vertex mesh (about the size of a femoral BCI at 0.5 mm): **2.8 s**.

---

## 3. Generating the fixed notebook

```bash
git clone <this repo> && cd RAVE-Knee2
python notebook_patch/apply_fixes.py \
    --notebook ConfCarti_Full_Pipeline.ipynb \
    --out      ConfCarti_Full_Pipeline_fixed.ipynb
```

`ConfCarti_Full_Pipeline_fixed.ipynb` is already committed here, so you can skip
this unless you have local edits to carry over. Every edit is anchored on an
exact substring and the script **fails loudly** if an anchor is missing, so a
drifted notebook cannot be half-patched in silence.

All stored outputs are cleared: the saved outputs were the phantom run, and
leaving them beside code that now reads real data produces a notebook that looks
like it has real results and does not.

---

## 4. Running it on your data

### 4.1 Layout

The loader expects nnU-Net style under `OAIZIB_ROOT`:

```
OAIZIB-CM/
├── imagesTr/   9001104_0000.nii.gz   ...
├── labelsTr/   9001104.nii.gz        ...
├── imagesTs/   (optional — picked up automatically)
└── labelsTs/
```

`imagesTr` + `labelsTr`, `imagesTs` + `labelsTs`, and `images` + `labels` are all
searched. Subject ids are parsed as the 7-digit OAI number in the filename
(`9001104_V00.nii.gz`, `sub-9001104_....nii.gz`), which is how cases join to the
metadata table. If a mirror names files differently, that join is the first thing
to check — the loader prints how many cases matched and lists the ones that
didn't.

Labels must follow the OAIZIB-CM 5-ROI convention:
`1` femur, `2` femoral cartilage, `3` tibia, `4` medial tibial cartilage,
`5` lateral tibial cartilage. The label-convention check in section 3 verifies
this on the first 20 cases.

### 4.2 Environment

```bash
conda create -n confcarti python=3.10 -y && conda activate confcarti
pip install numpy scipy pandas scikit-learn scikit-image trimesh matplotlib \
            pyyaml nibabel SimpleITK openpyxl rtree plotly
# section 6 only (segmentation training):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install monai
```

`SimpleITK` is what reads the NIfTI headers; `openpyxl` is needed for the
CartiMorph `.xlsx` metadata tables; `rtree` speeds up trimesh ray queries (the
original notebook had a bare `# pip install rtree` cell for this).

### 4.3 Configure

The only cell you edit is the configuration cell in section 2:

```python
USE_REAL_DATA  = True
OAIZIB_ROOT    = r"D:\ishtiaque\Knee segementation\OAI-ZIB\OAIZIB-CM\OAIZIB-CM"
N_SUBJECTS     = None    # None = all; set 8 for a smoke test first
RESAMPLE_TO_MM = 0.5     # isotropic before meshing; None = acquisition grid
```

Note the `r"..."` prefix — a raw string. Without it a Windows path containing
`\t`, `\n` or `\U` is silently mangled.

**Run with `N_SUBJECTS = 8` first.** It exercises the whole notebook end to end
in a few minutes and will surface a path or label-convention problem before you
commit to a multi-hour run.

`RESAMPLE_TO_MM = 0.5` matters for curvature: OAI DESS is acquired at about
0.36 x 0.36 x 0.7 mm, and marching cubes on a grid that anisotropic produces
triangles twice as long in z as in x — the resulting curvature carries the grid's
anisotropy rather than the anatomy's. This is the same isotropic resampling
CartiMorph applies before meshing.

### 4.4 Run order

Run top to bottom. Sections 1 (library code) and 2 (data) must run first;
everything else reads `confcarti_results.csv`.

| § | What | Time, ~480 knees |
|---|---|---|
| 1 | library cells | seconds |
| 2 | metadata fetch + cohort discovery | ~1 min (needs network for the CartiMorph tables) |
| 3 | dataset figures | ~1 min |
| 4 | closed-form geometry validation | ~1 min — **stop if anything says FAIL** |
| 5 | per-knee morphometry | **4–8 h**, progress printed every 10 knees |
| 6 | segmentation training | hours on a GPU; set `RUN_TRAINING = False` to skip |
| 7 | conformal layer | minutes |
| 8 | reliability (`N_RELIABILITY` x `REPEATS` extra runs) | ~30 min |
| 9 | ablations (`ABLATION_N` knees x 9 variants) | ~1 h |
| 10 | tables + manifest | seconds |

Section 5 is the long one and it writes `confcarti_results.csv` at the end. If
you expect to iterate on sections 7–10, run 5 once and then reload rather than
re-running it:

```python
results = pd.read_csv("confcarti_results.csv")
```

Sections 6–10 are independent of section 5 except through that CSV.

### 4.5 Check these before believing any output

1. **The manifest.** `confcarti_manifest.json` now reports the *effective* value:

   ```json
   "used_real_data": true,
   "n_subjects": 481,
   "curvature_radius_mm": 3.0
   ```

   If `n_subjects` is 32, you are still on phantoms.

2. **Curvature coverage.** Section 5 prints an estimable fraction per
   compartment. Expect **> 0.85** for FC and **> 0.75** for MTC/LTC at a 3.0 mm
   radius. If a compartment is much lower, the plate is fragmented — usually a
   segmentation problem, not a curvature problem. Look at the surface before
   changing the radius.

3. **Split sizes.** With ~480 subjects at 0.6/0.2/0.2 you should get about
   288/96/96, and the Mondrian warnings about groups smaller than 9 should be
   gone. Those warnings in the original run were a symptom of the 32-subject
   phantom cohort, not of the conformal code.

4. **No `clipped ... % of the plate` warnings.** If they reappear, `CROP_SIZE` is
   too small for your field of view; raise it.

### 4.6 Section 6, honestly

`MAX_EPOCHS = 500` and `backbone="swinunetr"` are set for real training, but
6 epochs of SegResNet on 32 phantoms was never a model and 500 epochs on 288
knees is a day or more of GPU time. If the segmentation is not the point of your
study, set `RUN_TRAINING = False`: sections 7–10 run on the manual OAIZIB-CM
labels, which is the stronger basis for a morphometry claim anyway. The
conformal layer does not depend on the network.

---

## 5. Known limits that remain

- **Adaptive-scale curvature is a mixed-scale field.** Vertices near a rim are
  measured at 1.5–2.0 mm and the interior at 3.0 mm; a smaller radius admits
  more marching-cubes ripple, so the rim runs slightly noisier. Both the median
  radius and the reduced fraction go into the CSV so you can check the mix and,
  if a reviewer asks, re-run with `CURVATURE_ADAPTIVE = False` and compare.
- **The strip fixtures are cylinders and caps.** They pin accuracy and coverage
  on open patches of the right size and curvature, but a real BCI has holes from
  full-thickness loss. `estimable_fraction` is the number that tells you when
  that is biting.
- **`site` is the OAI image-release prefix**, an acquisition-batch surrogate, not
  the OAI clinical site (`V00SITE`). "Site-conditional coverage" should say which
  variable it means. This was already documented in `from_cartimorph_tables` and
  is unchanged.
- **The bulge amplitude is a relative index**, under-estimated by design because
  a local form fit absorbs part of the feature it isolates. Also unchanged, and
  the notebook already says so.
