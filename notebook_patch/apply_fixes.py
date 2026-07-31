#!/usr/bin/env python3
"""Rewrite ConfCarti_Full_Pipeline.ipynb into a version that runs on real data.

Usage
-----
    python notebook_patch/apply_fixes.py \
        --notebook ConfCarti_Full_Pipeline.ipynb \
        --out      ConfCarti_Full_Pipeline_fixed.ipynb

Every edit is anchored on an exact substring of the original notebook and the
script **fails loudly** if an anchor is missing, so a notebook that has already
drifted cannot be half-patched in silence. Run it against the notebook the
errored run came from.

What it changes, and why
------------------------
1. **The cohort was never real.** `USE_REAL_DATA` and `OAIZIB_ROOT` were declared
   and then ignored; the data cell always called `make_phantom_cohort`. The run
   manifest still recorded `"used_real_data": true`, because it reported the flag
   rather than what the flag did. Fixed by replacing the config and data cells,
   and by reporting the *effective* value in the manifest.

2. **Curvature was discarded on 76-87% of every surface.** `compute_curvature`
   NaN-ed every vertex within `2 * radius_mm` = 6.0 mm of an open boundary,
   measured through space with a KD-tree. A cartilage plate is an open sheet
   ~20-25 mm across, so almost none of it survives, and Euclidean distance also
   reaches around folds. Replaced by the corrected module, which tests each fit
   for two-sided support directly.

3. **Whole-cohort loads.** Several cells did `list(zip(knees, ...))`, which is
   harmless for 32 in-memory phantoms and fatal for 481 volumes on disk.
   Rewritten to index.

4. **A crop that cut the joint in half.** `(64, 64, 48)` at 0.5 mm is a
   32 x 32 x 24 mm box; the original run warned on every case that it clipped
   ~50% of the cartilage plate. Driven from `CROP_SIZE` now.

5. **Risk control materialised every voxel.** ~2.5 M voxels x ~100 calibration
   knees x float64 is ~2 GB. Subsampled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class PatchError(RuntimeError):
    """An anchor did not match, so the notebook is not the one this expects."""


def _source(cell) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _set_source(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def find_cell(cells, anchor: str, *, what: str) -> int:
    """Index of the single code cell containing ``anchor``."""
    hits = [
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "code" and anchor in _source(c)
    ]
    if len(hits) != 1:
        raise PatchError(
            f"expected exactly one cell containing {anchor!r} ({what}), found {len(hits)}"
        )
    return hits[0]


def replace_in_cell(cells, index: int, old: str, new: str, *, what: str) -> None:
    text = _source(cells[index])
    if old not in text:
        raise PatchError(f"anchor for {what!r} not found in cell {index}")
    if text.count(old) != 1:
        raise PatchError(f"anchor for {what!r} is ambiguous in cell {index}")
    _set_source(cells[index], text.replace(old, new))


# --------------------------------------------------------------------------- #
# Replacement cell bodies
# --------------------------------------------------------------------------- #

CURVATURE_MODULE = (REPO / "confcarti" / "thickness" / "curvature.py").read_text()
CONFIG_CELL = (HERE / "cell_config.py").read_text()
DATA_CELL = (HERE / "cell_data_real.py").read_text()

MORPHOMETRY_LOOP = '''import time

all_rows, kept_surfaces, all_warnings = [], {}, []
keep_ids = set(metadata.subject_id.iloc[:4])          # keep a few for 3D rendering

start = time.time()
for i, (_, meta_row) in enumerate(metadata.iterrows()):
    knee = knees[i]
    analysis = analyse_knee(
        knee.label, knee.spacing,
        subject_id=meta_row.subject_id,
        kl_grade=int(meta_row.kl_grade),
        site=meta_row.site,
        split=SUBJECT_SPLIT.get(meta_row.subject_id),
        laterality=meta_row.laterality,
        model_name="manual",
        seed=SEED,
        thickness_method="surface_normal",
        curvature_radius_mm=CURVATURE_RADIUS_MM,
        curvature_min_radius_mm=CURVATURE_MIN_RADIUS_MM,
        curvature_adaptive=CURVATURE_ADAPTIVE,
        keep_surfaces=meta_row.subject_id in keep_ids,
    )
    all_rows.append(analysis.rows)
    all_warnings.extend(analysis.warnings)
    if analysis.surfaces:
        kept_surfaces[meta_row.subject_id] = analysis.surfaces
    # A real cohort is 400+ knees at tens of seconds each. Silence for two hours
    # is indistinguishable from a hang, so say where we are.
    if (i + 1) % 10 == 0 or i + 1 == len(metadata):
        rate = (time.time() - start) / (i + 1)
        print(f"  {i + 1}/{len(metadata)} knees  ({rate:.1f}s each, "
              f"~{rate * (len(metadata) - i - 1) / 60:.0f} min left)", flush=True)

results = pd.concat(all_rows, ignore_index=True)
print(f"analysed {len(metadata)} knees in {time.time()-start:.0f}s")
print(f"tidy results: {len(results)} rows, {results.metric.nunique()} distinct metrics, "
      f"{results.subregion.nunique()} subregions")
print("warnings:", len(all_warnings))
for w in all_warnings[:3]:
    print("  ", w)

# How much of each plate the curvature actually describes. Without this the
# curvature columns are uninterpretable: a compartment mean says nothing about
# whether it covers the plate or an island in the middle of it.
est = results[results.metric == "curv_estimable_fraction"]
if len(est):
    print("\\ncurvature estimable fraction by compartment:")
    print(est.pivot_table(index="subregion", values="value",
                          aggfunc=["mean", "min"]).round(3).to_string())

results.to_csv("confcarti_results.csv", index=False)
print("\\nwrote confcarti_results.csv — every table and figure below is built from this file alone")
results.head(8)'''

OPEN_PATCH_CHECK = '''
# --- curvature on an OPEN patch, which is what a cartilage plate is -----------
# The checks above are all on closed or unbounded surfaces -- a sphere, a plane --
# and that is exactly why the boundary bug survived them. A cartilage plate is an
# open sheet ~20-25 mm across; every point of it is close to a rim. This builds
# a strip cut from a cylinder of tibial-plate size, where H = 1/(2R) and K = 0
# exactly, and checks both the accuracy and how much of the patch survives.
def _cylinder_strip(radius_mm=22.0, width_mm=22.0, arc_mm=24.0, edge_mm=0.5):
    n_ax = int(round(width_mm / edge_mm)) + 1
    n_ci = int(round(arc_mm / edge_mm)) + 1
    axial = np.linspace(-width_mm / 2, width_mm / 2, n_ax)
    theta = np.linspace(-arc_mm / (2 * radius_mm), arc_mm / (2 * radius_mm), n_ci)
    tt, aa = np.meshgrid(theta, axial, indexing="ij")
    verts = np.stack([radius_mm * np.cos(tt), radius_mm * np.sin(tt), aa], -1).reshape(-1, 3)
    f = []
    for i_ in range(n_ci - 1):
        for j_ in range(n_ax - 1):
            a_ = i_ * n_ax + j_
            f += [[a_, a_ + n_ax, a_ + 1], [a_ + 1, a_ + n_ax, a_ + n_ax + 1]]
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(f), process=False)
    if np.dot(m.vertex_normals[0], m.vertices[0] * [1, 1, 0]) < 0:
        m.invert()
    return m

_R = 22.0
_strip = _cylinder_strip(radius_mm=_R)
_new = compute_curvature(_strip, radius_mm=CURVATURE_RADIUS_MM,
                         min_radius_mm=CURVATURE_MIN_RADIUS_MM,
                         adaptive_radius=CURVATURE_ADAPTIVE, smoothing_iterations=3)
_old = compute_curvature(_strip, radius_mm=CURVATURE_RADIUS_MM, smoothing_iterations=3,
                         mask_boundary=True, boundary_margin_mm=2 * CURVATURE_RADIUS_MM,
                         boundary_metric="euclidean")
checks.append(("curvature H, open strip R=22 (plate-sized)",
               float(np.nanmedian(_new.mean_curvature)), 1 / (2 * _R), 0.10))
checks.append(("curvature K, open strip (0)",
               float(np.nanmedian(_new.gaussian_curvature)), 0.0, None))
checks.append(("open strip: fraction of plate estimable",
               _new.estimable_fraction, 1.0, 0.20))

print(f"open-patch coverage: {100 * _new.estimable_fraction:5.1f}% with the support test, "
      f"{100 * _old.estimable_fraction:5.1f}% with the old {2 * CURVATURE_RADIUS_MM:.0f} mm "
      f"Euclidean margin")
print("The second number is the defect: on a plate-sized open patch the old rule keeps "
      "an island\\nin the middle and calls it the plate's curvature.\\n")

'''

RISK_CONTROL_HEAD = '''# --- conformal risk control on the segmentation masks -------------------------
# Voxel subsampling: a real knee volume is ~2.5 M voxels, and holding one float64
# probability map per calibration subject is gigabytes. The risk is an expectation
# over voxels, so a fixed random subsample estimates it without bias; the seed
# keeps it reproducible.
RISK_VOXELS = 200_000
rng = np.random.default_rng(SEED)
prob_maps, truth_masks = [], []
for i, (_, m) in enumerate(metadata.iterrows()):
    if SUBJECT_SPLIT.get(m.subject_id) not in ("calibration", "test"):
        continue
    knee = knees[i]
    truth_mask = np.isin(knee.label, CARTILAGE_LABELS).reshape(-1)
    if truth_mask.size > RISK_VOXELS:
        pick = rng.choice(truth_mask.size, RISK_VOXELS, replace=False)
        truth_mask = truth_mask[pick]
    # Stand-in for a network's softmax when §6 was skipped: a noisy but informative score.
    prob = np.clip(np.where(truth_mask, rng.beta(6, 2, truth_mask.shape),
                                        rng.beta(2, 6, truth_mask.shape)), 0, 1)
    prob_maps.append(prob); truth_masks.append(truth_mask)
'''


def patch(notebook: dict) -> dict:
    cells = notebook["cells"]

    # 1. curvature module -- the whole cell is the module source.
    i = find_cell(cells, "confcarti/thickness/curvature.py", what="curvature module")
    _set_source(cells[i], CURVATURE_MODULE)

    # 2/3. config and data cells.
    i = find_cell(cells, "USE_REAL_DATA = True", what="run configuration")
    _set_source(cells[i], CONFIG_CELL)
    i = find_cell(cells, "def fetch_oaizib_metadata", what="cohort metadata")
    _set_source(cells[i], DATA_CELL)

    # 4. pipeline: thread the curvature parameters through analyse_knee.
    i = find_cell(cells, "def analyse_knee(", what="per-knee pipeline")
    replace_in_cell(
        cells, i,
        "    curvature_radius_mm: float = 3.0,\n",
        "    curvature_radius_mm: float = 3.0,\n"
        "    curvature_min_radius_mm: float | None = 1.5,\n"
        "    curvature_adaptive: bool = True,\n"
        "    curvature_mask_boundary: bool = False,\n",
        what="analyse_knee signature",
    )
    replace_in_cell(
        cells, i,
        "    curvature_radius_mm\n        Physical radius of the curvature fit.\n",
        "    curvature_radius_mm\n"
        "        Nominal physical radius of the curvature fit.\n"
        "    curvature_min_radius_mm, curvature_adaptive\n"
        "        Adaptive-scale fallback near a plate rim; see\n"
        "        :func:`confcarti.thickness.curvature.compute_curvature`.\n"
        "    curvature_mask_boundary\n"
        "        Additionally discard a geometric margin around the plate rim.\n"
        "        Off by default -- the old ``2 * radius_mm`` margin removed\n"
        "        76-87 per cent of every surface.\n",
        what="analyse_knee docstring",
    )
    replace_in_cell(
        cells, i,
        "                curvature_result = compute_curvature(\n"
        "                    bci, radius_mm=curvature_radius_mm, smoothing_iterations=3\n"
        "                )",
        "                curvature_result = compute_curvature(\n"
        "                    bci,\n"
        "                    radius_mm=curvature_radius_mm,\n"
        "                    min_radius_mm=curvature_min_radius_mm,\n"
        "                    adaptive_radius=curvature_adaptive,\n"
        "                    mask_boundary=curvature_mask_boundary,\n"
        "                    smoothing_iterations=3,\n"
        "                )\n"
        "                if curvature_result.estimable_fraction < 0.5:\n"
        "                    warnings.append(\n"
        "                        f\"{subject_id}: {key} curvature estimable on only \"\n"
        "                        f\"{100 * curvature_result.estimable_fraction:.0f}% of the \"\n"
        "                        f\"surface at radius {curvature_radius_mm} mm\"\n"
        "                    )",
        what="compute_curvature call",
    )

    # 4b. geometry validation: add the open-surface case. The existing checks are
    # all on a sphere or a plane -- closed or unbounded -- which is precisely why
    # a boundary rule that destroys open patches passed every one of them.
    i = find_cell(cells, "# --- Taubin smoothing preserves volume", what="geometry validation")
    replace_in_cell(
        cells, i,
        "# --- Taubin smoothing preserves volume",
        OPEN_PATCH_CHECK + "# --- Taubin smoothing preserves volume",
        what="open-patch validation",
    )

    # 5. dataset wrapper must not materialise a lazy cohort.
    i = find_cell(cells, "class PhantomDataset(_BaseKneeDataset)", what="dataset module")
    replace_in_cell(
        cells, i,
        "        self.knees = list(knees)\n",
        "        # NOT list(knees): a real cohort is a lazy sequence that reads each\n"
        "        # volume from disk on access, and list() would pull all 481 of them\n"
        "        # into RAM at construction time.\n"
        "        self.knees = knees\n",
        what="PhantomDataset lazy cohort",
    )

    # 6. label-convention sweep must not read every volume.
    i = find_cell(cells, "label convention check on every phantom", what="raw volume view")
    replace_in_cell(
        cells, i,
        'print("label convention check on every phantom:")\n'
        "bad = [m.subject_id for k, (_, m) in zip(knees, metadata.iterrows())\n"
        "       if not validate_label_convention(k.label).ok]",
        "N_CONVENTION_CHECK = min(len(metadata), 20)   # reading 481 volumes to check a\n"
        "                                             # label convention is not worth it\n"
        'print(f"label convention check on {N_CONVENTION_CHECK} case(s):")\n'
        "bad = [metadata.subject_id.iloc[j] for j in range(N_CONVENTION_CHECK)\n"
        "       if not validate_label_convention(knees[j].label).ok]",
        what="label convention sweep",
    )

    # 7. main morphometry loop.
    i = find_cell(cells, "all_rows, kept_surfaces, all_warnings", what="morphometry loop")
    _set_source(cells[i], MORPHOMETRY_LOOP)

    # 8. training / inference crop size.
    i = find_cell(cells, "RUN_TRAINING = HAS_TORCH", what="training cell")
    replace_in_cell(
        cells, i,
        "RUN_TRAINING = HAS_TORCH and True     # set False to skip\n"
        "MAX_EPOCHS   = 6                      # demonstration budget; raise for real training\n",
        "RUN_TRAINING = HAS_TORCH and True     # set False to skip\n"
        "# MAX_EPOCHS and CROP_SIZE come from the configuration cell: 6 epochs on a\n"
        "# 32 x 32 x 24 mm crop is a smoke test, not a model.\n",
        what="training budget",
    )
    replace_in_cell(
        cells, i,
        "dataset = PhantomDataset(knees, metadata, crop_size=(64, 64, 48))",
        "dataset = PhantomDataset(knees, metadata, crop_size=CROP_SIZE)",
        what="training crop size",
    )
    replace_in_cell(
        cells, i,
        "patch_size=(64, 64, 48), batch_size=1,",
        "patch_size=CROP_SIZE, batch_size=1,",
        what="training patch size",
    )
    i = find_cell(cells, "# --- inference + segmentation metrics", what="inference cell")
    replace_in_cell(
        cells, i,
        "roi_size=(64, 64, 48),",
        "roi_size=CROP_SIZE,",
        what="inference roi size",
    )

    # 9. conformal risk control: subsample voxels.
    i = find_cell(
        cells, "# --- conformal risk control on the segmentation masks",
        what="risk control cell",
    )
    text = _source(cells[i])
    head_end = text.index("n_cal = sum(")
    _set_source(cells[i], RISK_CONTROL_HEAD + "\n" + text[head_end:])

    # 10. reliability + ablations: index instead of zipping the whole cohort.
    i = find_cell(cells, "repeat_subjects = list(metadata.subject_id", what="reliability cell")
    replace_in_cell(
        cells, i,
        "for knee, (_, m) in zip(knees, metadata.iterrows()):\n"
        "    if m.subject_id not in repeat_subjects:\n"
        "        continue\n",
        "for j, (_, m) in enumerate(metadata.iterrows()):\n"
        "    if m.subject_id not in repeat_subjects:\n"
        "        continue\n"
        "    knee = knees[j]\n",
        what="reliability loop",
    )
    for anchor, what in (
        ("ABLATION_N = min(12", "ablation A1"),
        ("A2: denuded on", "ablation A2/A6/A8"),
    ):
        i = find_cell(cells, anchor, what=what)
        replace_in_cell(
            cells, i,
            "    for knee, (_, m) in list(zip(knees, metadata.iterrows()))[:ABLATION_N]:\n",
            "    for j in range(ABLATION_N):\n"
            "        m = metadata.iloc[j]\n"
            "        knee = knees[j]\n",
            what=f"{what} loop",
        )

    # 11. the manifest must report what happened, not what was requested.
    i = find_cell(
        cells, "# --- run manifest: what produced these numbers", what="run manifest"
    )
    replace_in_cell(
        cells, i,
        '    "used_real_data": bool(USE_REAL_DATA),\n',
        '    # The *effective* value. The original reported the flag, which is how a\n'
        '    # phantom run came to be recorded as a real-data run.\n'
        '    "used_real_data": bool(USE_REAL_DATA_EFFECTIVE),\n'
        '    "curvature_radius_mm": CURVATURE_RADIUS_MM,\n'
        '    "curvature_min_radius_mm": CURVATURE_MIN_RADIUS_MM,\n'
        '    "curvature_adaptive": bool(CURVATURE_ADAPTIVE),\n',
        what="manifest used_real_data",
    )

    # 12. Drop every stored output. The saved outputs are the phantom run; leaving
    # them beside code that now reads real data produces a notebook that *looks*
    # like it has real results and does not. Re-run it.
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    return notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    try:
        patched = patch(notebook)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nThe notebook does not match what this patch expects. Apply the cell "
            "replacements in notebook_patch/ by hand, or run against the original "
            "ConfCarti_Full_Pipeline.ipynb.",
            file=sys.stderr,
        )
        return 1

    args.out.write_text(json.dumps(patched, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
