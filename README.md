# RAVE-Knee2 — ConfCarti pipeline fixes

Corrections to the ConfCarti knee cartilage morphometry notebook
(`ConfCarti_Full_Pipeline.ipynb`), for running it on the real **OAIZIB-CM**
dataset.

**Start here: [`docs/RUNBOOK_OAIZIB.md`](docs/RUNBOOK_OAIZIB.md)** — what was
wrong, what changed, and the step-by-step run procedure.

## The defects

1. **The pipeline never touched the real data.** `USE_REAL_DATA` and
   `OAIZIB_ROOT` were declared and then ignored; the data cell always called
   `make_phantom_cohort`. Every reported number came from 32 analytic phantoms,
   while the run manifest recorded `"used_real_data": true` because it reported
   the flag rather than what the flag did.

1b. **And it could not have found the files anyway.** OAIZIB-CM ships
   `oaizib_001_0000.nii.gz` / `oaizib_001.nii.gz` — a sequential case number, not
   the 7-digit OAI subject id. The loader's `(?:sub-)?(\d{7})` matches nothing,
   falls back to the stem, and keys image and label differently, so
   `discover_cases` raises *"507 image(s) have no matching label"* on a dataset
   where every image has one. The bridge is the `CMT-ID` column of the shipped
   subject tables; `confcarti/data/oaizib.py` makes and checks that join, and
   reads the metadata locally instead of fetching a 481-row subset over the
   network.

2. **Curvature was discarded on 76–87 % of every surface.** `compute_curvature`
   NaN-ed every vertex within `2 * radius_mm` = **6.0 mm** of an open boundary,
   measured through space with a KD-tree. A cartilage plate is an open sheet
   20–25 mm across, so almost none of it survives — and Euclidean distance also
   reaches around folds like the trochlear groove. It passed validation only
   because every closed-form check was on a sphere or a plane, where the rule
   never fires.

## What is here

| Path | |
|---|---|
| `confcarti/thickness/curvature.py` | corrected curvature module: per-vertex support test in place of the blanket margin, geodesic boundary distance, adaptive scale, coverage reported alongside every summary |
| `confcarti/data/oaizib.py` | the OAIZIB-CM case-id join: subject tables, split manifests, and the checks that catch a release mismatch |
| `tests/test_curvature_open_patch.py` | closed-form validation on **open** patches — the case the original suite never exercised |
| `tests/test_oaizib_metadata.py` | the metadata join, its failure modes, and the split |
| `notebook_patch/apply_fixes.py` | rewrites the notebook; fails loudly if an anchor is missing |
| `notebook_patch/cell_config.py`, `cell_data_real.py`, `cell_splits.py` | replacement cells, readable on their own |
| `ConfCarti_Full_Pipeline_fixed.ipynb` | the patched notebook, outputs cleared |
| `docs/RUNBOOK_OAIZIB.md` | the full write-up and run procedure |

## Quick start

```bash
python -m pytest tests/ -q                      # 30 passed

python notebook_patch/apply_fixes.py \
    --notebook ConfCarti_Full_Pipeline.ipynb \
    --out      ConfCarti_Full_Pipeline_fixed.ipynb
```

Then set `OAIZIB_ROOT` in the notebook's configuration cell and run top to
bottom — with `N_SUBJECTS = 8` first. `OAIZIB_ROOT` must hold both the nnU-Net
directories and the four shipped CSVs (`subInfo_train_1.csv`,
`subInfo_test_1.csv`, `train.csv`, `test.csv`); the run needs no network access.

On the 507-case release that yields **481 analysable subjects** (26 have no KL
grade — the same 26 the public CartiMorph tables drop) split **288 / 95 / 98**,
holding the dataset's own test set out whole.

## Curvature, before and after

On a 22 x 24 mm open strip cut from a cylinder of radius 22 mm, where
`H = 1/(2R) = 0.022727` and `K = 0` exactly:

| | plate estimable | measured H |
|---|---|---|
| old 6 mm Euclidean margin | 19.8 % | — (an island in the middle) |
| per-vertex support test | **91.7 %** | **0.02279** (+0.3 %) |

Closed-sphere accuracy is unchanged; the fit no longer reaches across a fold; and
`curv_estimable_fraction` now travels with every curvature summary, because a
curvature mean without it says nothing about whether it describes the plate or a
fragment of it.
