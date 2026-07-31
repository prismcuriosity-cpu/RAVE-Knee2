from __future__ import annotations
# ==========================================================================
# confcarti/data/oaizib.py
# ==========================================================================
"""OAIZIB-CM on disk: case ids, the shipped subject tables and the split manifests.

Why this module exists
----------------------
The generic loader in :mod:`confcarti.data.dataset` parses a subject id out of a
filename with ``(?:sub-)?(\\d{7})`` -- the 7-digit OAI identifier. OAIZIB-CM does
not name its files that way. It ships:

    imagesTr/oaizib_001_0000.nii.gz   labelsTr/oaizib_001.nii.gz
    imagesTs/oaizib_405_0000.nii.gz   labelsTs/oaizib_405.nii.gz

with a **sequential case number**, not the OAI subject id, and with nnU-Net's
``_0000`` modality suffix on the image but not on the label. Against that layout
the generic parser fails twice over:

* no run of seven digits exists in ``oaizib_497_0000``, so it falls back to the
  whole stem;
* the image stem is then ``oaizib_497_0000`` and the label stem ``oaizib_497``,
  so the two never join and ``discover_cases(require_labels=True)`` raises
  ``507 image(s) have no matching label`` on a dataset where every image has one.

The bridge between the two worlds is the ``CMT-ID`` column of the subject tables
shipped with the dataset (``subInfo_train_1.csv`` / ``subInfo_test_1.csv``):
``CMT-ID`` *is* the case number, zero-padded to three digits. This module makes
that join explicit and checks it, rather than inferring identity from filenames.

What the tables contain
-----------------------
507 rows, ``CMT-ID`` 1-507, one row per subject, ``SubjectID`` unique. Columns:
``SubjectID, CMT-ID, Path, MRBarCode, KneeSide, KLGrade, Gender, Age, BMI``.

Three properties of the real data that the code below has to handle, all of them
verified against the shipped tables rather than assumed:

1. **26 subjects have no KL grade** (21 in train, 5 in test). 507 - 26 = 481,
   which is exactly the row count of the CartiMorph tables published on GitHub --
   the public release simply drops them. KL grade is not optional here: every
   split is stratified on it and conditional coverage is reported by it, and
   ``validate_metadata`` treats a missing grade as a fatal problem. They are
   dropped by default, and counted out loud.
2. **BMI can be blank as well as zero.** The published ``.xlsx`` encodes missing
   BMI as ``0``; the shipped CSVs leave the cell empty. Both are missing.
3. **``KneeSide`` is 1 (right) for all 507 cases.** The parcellation's
   left/right mirroring therefore never varies on this dataset. Worth knowing
   before reporting it as a handled degree of freedom.

The official split
------------------
``train.csv`` and ``test.csv`` list 404 and 103 cases with no overlap, and they
match the two subject tables case-for-case. The split is stratified on KL to
within half a point in every stratum (82/21, 46/12, 86/22, 111/28, 58/15), so
calibration data drawn from the official train set is exchangeable with the
official test set -- which is what the conformal guarantee needs. Prefer it over
re-splitting the pooled cohort.

Data access
-----------
No OAI data is redistributed with this package. These loaders read tables you
already hold under your own OAI data use agreement.
"""


import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CASE_ID_PATTERN",
    "OAIZIBCohortTables",
    "case_id_from_filename",
    "load_case_manifest",
    "load_subject_info",
    "load_oaizib_tables",
    "to_canonical_metadata",
]

#: Case number in an OAIZIB-CM filename. Three digits in the shipped release;
#: the ``{3,}`` allows a larger release without a code change.
CASE_ID_PATTERN = re.compile(r"(oaizib_(\d{3,}))")

#: Suffixes stripped before matching an image filename to a label filename.
_VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")

#: nnU-Net's modality suffix, present on images and absent on labels.
_MODALITY_SUFFIX = re.compile(r"_\d{4}$")

#: BMI recorded as this value or below means "not measured", not "very thin".
_BMI_MISSING_SENTINEL = 0.0

_SUBINFO_REQUIRED = ("SubjectID", "CMT-ID", "Path", "KneeSide", "KLGrade")


def case_id_from_filename(path: str | Path) -> str:
    """Extract the OAIZIB-CM case id from an image or label filename.

    Strips the volume extension and nnU-Net's ``_0000`` modality suffix, so that
    an image and its label reduce to the same key.

    Parameters
    ----------
    path
        Filename or path.

    Returns
    -------
    str
        ``oaizib_NNN``, or the bare stem when the name does not match the
        OAIZIB-CM convention (so a caller can see what it actually got).

    Examples
    --------
    >>> case_id_from_filename("imagesTr/oaizib_497_0000.nii.gz")
    'oaizib_497'
    >>> case_id_from_filename("labelsTr/oaizib_497.nii.gz")
    'oaizib_497'
    >>> case_id_from_filename("something_else.nii.gz")
    'something_else'
    """
    name = Path(path).name
    for suffix in _VOLUME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = _MODALITY_SUFFIX.sub("", name)
    match = CASE_ID_PATTERN.search(name)
    return match.group(1) if match else name


def case_id_from_cmt(cmt_id: int | str, *, width: int = 3) -> str:
    """Build the case id for a ``CMT-ID`` from the subject tables.

    Parameters
    ----------
    cmt_id
        The ``CMT-ID`` value, as an int or a possibly zero-padded string.
    width
        Zero-padding width; 3 in the shipped release.

    Returns
    -------
    str
        ``oaizib_NNN``.

    Examples
    --------
    >>> case_id_from_cmt("001"), case_id_from_cmt(497)
    ('oaizib_001', 'oaizib_497')
    """
    return f"oaizib_{int(cmt_id):0{width}d}"


def load_case_manifest(path: str | Path, *, split: str | None = None) -> pd.DataFrame:
    """Load a ``train.csv`` / ``test.csv`` image-mask manifest.

    Parameters
    ----------
    path
        CSV with ``image`` and ``mask`` columns of bare filenames.
    split
        Value written into the ``official_split`` column.

    Returns
    -------
    pandas.DataFrame
        ``case_id, image, mask, official_split``.

    Raises
    ------
    ValueError
        If a required column is absent, if a case id is duplicated, or if any
        row's image and mask reduce to different case ids -- which would mean the
        manifest pairs one subject's image with another's segmentation.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = {"image", "mask"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing column(s) {sorted(missing)}")

    frame = frame.dropna(subset=["image", "mask"]).copy()
    frame["case_id"] = frame["image"].map(case_id_from_filename)
    mask_case = frame["mask"].map(case_id_from_filename)

    mismatched = frame.loc[frame["case_id"] != mask_case]
    if len(mismatched):
        example = mismatched.iloc[0]
        raise ValueError(
            f"{path.name}: {len(mismatched)} row(s) pair an image and a mask from "
            f"different cases, e.g. {example['image']!r} with {example['mask']!r}. "
            "Analysing that pair would measure one knee's cartilage against "
            "another knee's bone."
        )

    duplicated = frame["case_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{path.name}: duplicate case id(s) "
            f"{frame.loc[duplicated, 'case_id'].unique()[:5].tolist()}"
        )

    frame["official_split"] = split
    return frame[["case_id", "image", "mask", "official_split"]].reset_index(drop=True)


def load_subject_info(path: str | Path, *, split: str | None = None) -> pd.DataFrame:
    """Load one ``subInfo_*.csv`` subject table, unmodified except for a split tag.

    Parameters
    ----------
    path
        CSV with the OAIZIB-CM subject-table columns.
    split
        Value written into the ``official_split`` column.

    Returns
    -------
    pandas.DataFrame
        The table plus ``case_id`` and ``official_split``.

    Raises
    ------
    ValueError
        If a required column is absent or a ``CMT-ID`` is duplicated.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype={"CMT-ID": str})
    missing = [c for c in _SUBINFO_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing column(s) {missing}. Expected the OAIZIB-CM "
            f"subject table: {list(_SUBINFO_REQUIRED)} (+ Gender, Age, BMI)."
        )

    frame = frame.dropna(subset=["CMT-ID"]).copy()
    frame["case_id"] = frame["CMT-ID"].map(case_id_from_cmt)
    if frame["case_id"].duplicated().any():
        dupes = frame.loc[frame["case_id"].duplicated(), "case_id"].unique()[:5]
        raise ValueError(f"{path.name}: duplicate CMT-ID(s) {dupes.tolist()}")

    frame["official_split"] = split
    return frame.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class OAIZIBCohortTables:
    """The shipped tables, joined and checked.

    Attributes
    ----------
    metadata
        Canonical metadata for the analysable cohort, one row per subject.
    dropped_no_kl
        Subjects excluded for having no KL grade. Not a failure -- a documented
        property of the dataset -- but they cannot be stratified or given
        conditional coverage, so they are set aside explicitly rather than
        silently.
    manifest
        Case-id to filename mapping, or None when no manifest was supplied.
    """

    metadata: pd.DataFrame
    dropped_no_kl: pd.DataFrame
    manifest: pd.DataFrame | None = None

    @property
    def n_subjects(self) -> int:
        """Rows in the analysable cohort."""
        return int(len(self.metadata))

    def summary(self) -> str:
        """One-paragraph description, for printing at the top of a run."""
        kl = self.metadata["kl_grade"].value_counts().sort_index().to_dict()
        site = self.metadata["site"].value_counts().to_dict()
        split = (
            self.metadata["official_split"].value_counts().to_dict()
            if "official_split" in self.metadata
            else {}
        )
        return (
            f"OAIZIB-CM: {self.n_subjects} analysable subject(s)"
            f"{f', {len(self.dropped_no_kl)} dropped for missing KL grade' if len(self.dropped_no_kl) else ''}\n"
            f"  KL grade      : {kl}\n"
            f"  site (release): {site}\n"
            f"  official split: {split}"
        )


def to_canonical_metadata(
    frame: pd.DataFrame, *, site_from_release: bool = True
) -> pd.DataFrame:
    """Convert a loaded subject table to the canonical ConfCarti schema.

    Parameters
    ----------
    frame
        Concatenated output of :func:`load_subject_info`.
    site_from_release
        Use the OAI image-release prefix of ``Path`` as the ``site`` variable.

        .. note::
           This is an **acquisition-batch surrogate, not the OAI clinical
           site**. The true site variable is ``V00SITE`` in the OAI enrollees
           table and is not redistributed here. The surrogate is a legitimate
           stratifier -- it separates two distinct acquisition batches -- but a
           claim of "site-conditional coverage" should say which variable it
           means. Pass False and join a real ``site`` column when you have one.

    Returns
    -------
    pandas.DataFrame
        ``subject_id, kl_grade, site, visit, laterality`` first, then
        ``case_id, official_split, oai_subject_id, cmt_id, age, sex, bmi``.

    Notes
    -----
    ``subject_id`` is the OAI 7-digit identifier, not the case number: splits are
    claimed to be *subject-level*, and that claim should rest on subject identity
    rather than on a file index that happens to be one-to-one with it today.
    ``case_id`` carries the file join separately.
    """
    out = pd.DataFrame(
        {
            "subject_id": frame["SubjectID"].astype(str),
            "kl_grade": pd.to_numeric(frame["KLGrade"], errors="coerce"),
            # OAI-ZIB codes KneeSide 1 = right, 2 = left.
            "laterality": frame["KneeSide"].map({1: "right", 2: "left"}).fillna("unknown"),
            "visit": "V00",
            "case_id": frame["case_id"],
            "official_split": frame.get("official_split"),
            "oai_subject_id": frame["SubjectID"].astype(str),
            "cmt_id": frame["CMT-ID"].astype(str),
        }
    )

    if site_from_release:
        out["site"] = frame["Path"].astype(str).str.split("/").str[0]
    else:
        out["site"] = "unknown"

    for src, dst in (("Age", "age"), ("Gender", "sex"), ("BMI", "bmi"), ("MRBarCode", "mr_barcode")):
        if src in frame.columns:
            out[dst] = frame[src].to_numpy()

    if "sex" in out.columns:
        out["sex"] = out["sex"].map({1: "male", 2: "female"}).fillna("unknown")
    if "bmi" in out.columns:
        bmi = pd.to_numeric(out["bmi"], errors="coerce")
        # Two encodings of the same fact: the published .xlsx writes 0, the
        # shipped .csv leaves the cell empty. `to_numeric` handles the second;
        # the sentinel test handles the first. Treating a 0 as a real BMI would
        # drag any BMI-adjusted analysis toward zero for that subject.
        n_sentinel = int((bmi <= _BMI_MISSING_SENTINEL).sum())
        n_blank = int(bmi.isna().sum())
        if n_sentinel or n_blank:
            logger.warning(
                "BMI missing for %d subject(s): %d recorded as 0, %d blank",
                n_sentinel + n_blank, n_sentinel, n_blank,
            )
        out["bmi"] = bmi.mask(bmi <= _BMI_MISSING_SENTINEL)

    leading = ["subject_id", "kl_grade", "site", "visit", "laterality"]
    ordered = leading + [c for c in out.columns if c not in leading]
    return out[ordered].reset_index(drop=True)


def load_oaizib_tables(
    subinfo_train: str | Path,
    subinfo_test: str | Path | None = None,
    *,
    manifest_train: str | Path | None = None,
    manifest_test: str | Path | None = None,
    drop_missing_kl: bool = True,
    site_from_release: bool = True,
) -> OAIZIBCohortTables:
    """Load, join and check the OAIZIB-CM subject tables and split manifests.

    Parameters
    ----------
    subinfo_train, subinfo_test
        ``subInfo_train_1.csv`` and ``subInfo_test_1.csv``. The test table is
        optional only so that a partial download still loads.
    manifest_train, manifest_test
        ``train.csv`` and ``test.csv``. When given, every case id in the
        manifests must appear in the subject tables and vice versa -- that check
        is the point of passing them.
    drop_missing_kl
        Exclude subjects with no KL grade. Keep them only if you intend to use
        them somewhere that does not stratify (segmentation training, say):
        every split, every Mondrian group and every conditional-coverage table
        needs a grade, and ``validate_metadata`` rejects a table without one.
    site_from_release
        See :func:`to_canonical_metadata`.

    Returns
    -------
    OAIZIBCohortTables

    Raises
    ------
    ValueError
        If a subject id repeats across the two tables, or if the manifests and
        the subject tables disagree about which cases exist.

    Examples
    --------
    >>> tables = load_oaizib_tables(                     # doctest: +SKIP
    ...     root / "subInfo_train_1.csv", root / "subInfo_test_1.csv",
    ...     manifest_train=root / "train.csv", manifest_test=root / "test.csv",
    ... )
    >>> print(tables.summary())                          # doctest: +SKIP
    """
    frames = [load_subject_info(subinfo_train, split="train")]
    if subinfo_test is not None:
        frames.append(load_subject_info(subinfo_test, split="test"))
    combined = pd.concat(frames, ignore_index=True)

    duplicated = combined["SubjectID"].astype(str).duplicated()
    if duplicated.any():
        offenders = combined.loc[duplicated, "SubjectID"].unique()[:5].tolist()
        raise ValueError(
            f"{int(duplicated.sum())} subject(s) appear in both the train and test "
            f"tables, e.g. {offenders}. Two scans of one knee straddling the "
            "train/test boundary breaks the exchangeability the coverage "
            "guarantee rests on."
        )

    manifests = [
        load_case_manifest(path, split=split)
        for path, split in ((manifest_train, "train"), (manifest_test, "test"))
        if path is not None
    ]
    manifest = pd.concat(manifests, ignore_index=True) if manifests else None

    if manifest is not None:
        table_cases = set(combined["case_id"])
        manifest_cases = set(manifest["case_id"])
        only_manifest = sorted(manifest_cases - table_cases)
        only_table = sorted(table_cases - manifest_cases)
        if only_manifest or only_table:
            raise ValueError(
                "the split manifests and the subject tables disagree: "
                f"{len(only_manifest)} case(s) listed in a manifest have no subject "
                f"row (e.g. {only_manifest[:5]}), {len(only_table)} subject(s) have "
                f"no manifest entry (e.g. {only_table[:5]}). The CMT-ID column is "
                "the join key; check that the tables and the manifests come from "
                "the same OAIZIB-CM release."
            )
        # The manifest's split is authoritative -- it is what the files were
        # actually partitioned by.
        split_by_case = manifest.set_index("case_id")["official_split"]
        combined["official_split"] = combined["case_id"].map(split_by_case)

    metadata = to_canonical_metadata(combined, site_from_release=site_from_release)

    missing_kl = metadata["kl_grade"].isna()
    dropped = metadata.loc[missing_kl].reset_index(drop=True)
    if drop_missing_kl and missing_kl.any():
        logger.warning(
            "%d of %d subject(s) have no KL grade and are excluded: they cannot be "
            "stratified, split or given conditional coverage. (507 - 26 = 481 is "
            "exactly the row count of the CartiMorph tables published on GitHub, "
            "which drop the same subjects.)",
            int(missing_kl.sum()), len(metadata),
        )
        metadata = metadata.loc[~missing_kl].reset_index(drop=True)
    metadata["kl_grade"] = metadata["kl_grade"].astype("Int64" if not drop_missing_kl else int)

    if manifest is not None:
        manifest = manifest[manifest["case_id"].isin(set(metadata["case_id"]))].reset_index(
            drop=True
        )

    lateralities = set(metadata["laterality"].unique())
    if lateralities == {"right"}:
        logger.info(
            "every case is a right knee, so the parcellation's left/right mirroring "
            "is never exercised on this dataset"
        )

    return OAIZIBCohortTables(metadata=metadata, dropped_no_kl=dropped, manifest=manifest)


def official_split_assignment(
    metadata: pd.DataFrame,
    *,
    calibration_fraction: float = 0.25,
    seed: int = 0,
    stratify_on: tuple[str, ...] = ("kl_grade", "site"),
) -> dict[str, list[str]]:
    """Split into train / calibration / test, respecting the dataset's own split.

    The official test set is held out whole and the calibration set is carved out
    of the official train set. This is preferable to re-splitting the pooled
    cohort for two reasons: the published split is already stratified on KL to
    within half a point in every stratum, so calibration and test scores are
    exchangeable; and any comparison against a published OAIZIB-CM number is
    otherwise being made on a different test set.

    Parameters
    ----------
    metadata
        Canonical metadata carrying ``official_split``.
    calibration_fraction
        Share of the official train set reserved for conformal calibration.
    seed
        RNG seed for the within-stratum shuffle.
    stratify_on
        Columns whose joint distribution the calibration split preserves.

    Returns
    -------
    dict
        ``{"train": [...], "calibration": [...], "test": [...]}`` of subject ids.

    Raises
    ------
    ValueError
        If ``official_split`` is absent, or the official train set is empty.
    """
    if "official_split" not in metadata.columns:
        raise ValueError(
            "metadata has no 'official_split' column; pass the manifests to "
            "load_oaizib_tables, or use make_splits() to build your own split"
        )
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )

    train_pool = metadata.loc[metadata["official_split"] == "train"]
    test_pool = metadata.loc[metadata["official_split"] == "test"]
    if train_pool.empty:
        raise ValueError("the official train set is empty; check the manifests")

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    calibration_ids: list[str] = []
    for _key, group in train_pool.groupby(list(stratify_on), dropna=False, sort=True):
        ids = group["subject_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        n_cal = int(round(calibration_fraction * len(ids)))
        # Never take a whole stratum for calibration, and never leave one with
        # nothing: a stratum of 1 stays in train.
        n_cal = min(max(n_cal, 1 if len(ids) > 1 else 0), len(ids) - 1 if len(ids) else 0)
        calibration_ids.extend(ids[:n_cal].tolist())
        train_ids.extend(ids[n_cal:].tolist())

    return {
        "train": sorted(train_ids),
        "calibration": sorted(calibration_ids),
        "test": sorted(test_pool["subject_id"].astype(str).tolist()),
    }
