"""OAIZIB-CM subject tables, split manifests and the case-id join.

The fixtures below are synthetic -- no OAI data is redistributed here -- but they
reproduce the schema and, deliberately, the three quirks of the shipped tables
that the loader has to survive: subjects with no KL grade, BMI recorded both as a
blank and as a zero, and filenames that carry a sequential case number rather than
the OAI subject id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confcarti.data.oaizib import (
    case_id_from_cmt,
    case_id_from_filename,
    load_case_manifest,
    load_oaizib_tables,
    load_subject_info,
    official_split_assignment,
    to_canonical_metadata,
)

SUBINFO_COLUMNS = [
    "SubjectID", "CMT-ID", "Path", "MRBarCode", "KneeSide",
    "KLGrade", "Gender", "Age", "BMI",
]


def _subinfo_rows(start_cmt, specs):
    """Build subject-table rows. ``specs`` is a list of (kl, site, bmi) triples."""
    rows = []
    for offset, (kl, site, bmi) in enumerate(specs):
        cmt = start_cmt + offset
        subject = 9000000 + cmt
        rows.append(
            {
                "SubjectID": subject,
                "CMT-ID": f"{cmt:03d}",
                "Path": f"{site}/{subject}/20050101/1000{cmt:04d}",
                "MRBarCode": f"0166100{cmt:05d}",
                "KneeSide": 1,
                "KLGrade": kl,
                "Gender": 1 + (offset % 2),
                "Age": 50 + (offset % 30),
                "BMI": bmi,
            }
        )
    return pd.DataFrame(rows, columns=SUBINFO_COLUMNS)


@pytest.fixture
def cohort(tmp_path):
    """A miniature OAIZIB-CM: 27 train + 8 test, 3 of them lacking a KL grade."""
    train_specs = [(kl, "0.C.2" if i % 2 else "0.E.1", 25.0 + i)
                   for i, kl in enumerate([0, 1, 2, 3, 4] * 4 + [0, 1, 2, 3])]
    train_specs += [(np.nan, "0.C.2", 30.0)] * 3          # no KL grade
    train_specs[2] = (train_specs[2][0], train_specs[2][1], np.nan)   # blank BMI
    train_specs[5] = (train_specs[5][0], train_specs[5][1], 0.0)      # BMI sentinel

    test_specs = [(kl, "0.E.1" if i % 2 else "0.C.2", 27.0 + i)
                  for i, kl in enumerate([0, 1, 2, 3, 4, 0, 2, 3])]

    train = _subinfo_rows(1, train_specs)
    test = _subinfo_rows(1 + len(train_specs), test_specs)

    paths = {}
    for name, frame in (("subInfo_train_1", train), ("subInfo_test_1", test)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    for name, frame in (("train", train), ("test", test)):
        cases = [case_id_from_cmt(c) for c in frame["CMT-ID"]]
        manifest = pd.DataFrame(
            {
                "image": [f"{c}_0000.nii.gz" for c in cases],
                "mask": [f"{c}.nii.gz" for c in cases],
            }
        )
        path = tmp_path / f"{name}.csv"
        manifest.to_csv(path, index=False)
        paths[name] = path

    return paths


# --------------------------------------------------------------------------- #
# The join that the generic loader could not make
# --------------------------------------------------------------------------- #


def test_case_id_reduces_image_and_label_to_the_same_key():
    """``oaizib_497_0000.nii.gz`` and ``oaizib_497.nii.gz`` are one case.

    This is the regression. The generic parser looks for seven consecutive
    digits, finds none in ``oaizib_497_0000``, falls back to the whole stem, and
    then keys the image as ``oaizib_497_0000`` and the label as ``oaizib_497`` --
    so nothing joins and ``discover_cases(require_labels=True)`` reports that all
    507 images are unlabelled.
    """
    assert case_id_from_filename("imagesTr/oaizib_497_0000.nii.gz") == "oaizib_497"
    assert case_id_from_filename("labelsTr/oaizib_497.nii.gz") == "oaizib_497"
    assert case_id_from_filename("oaizib_001_0000.nii") == "oaizib_001"

    import re
    seven_digits = re.compile(r"(?:sub-)?(\d{7})")
    assert seven_digits.search("oaizib_497_0000") is None


def test_case_id_falls_back_to_the_stem():
    """An unrecognised name reduces to its stem, so a mismatch is visible."""
    assert case_id_from_filename("something_else.nii.gz") == "something_else"


def test_case_id_from_cmt_zero_pads():
    assert case_id_from_cmt("001") == "oaizib_001"
    assert case_id_from_cmt(7) == "oaizib_007"
    assert case_id_from_cmt(497) == "oaizib_497"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_load_tables_joins_and_reports(cohort):
    tables = load_oaizib_tables(
        cohort["subInfo_train_1"], cohort["subInfo_test_1"],
        manifest_train=cohort["train"], manifest_test=cohort["test"],
    )

    assert tables.n_subjects == 35 - 3          # 3 dropped for a missing KL grade
    assert len(tables.dropped_no_kl) == 3
    assert tables.metadata["kl_grade"].notna().all()
    assert set(tables.metadata["official_split"]) == {"train", "test"}
    assert len(tables.manifest) == tables.n_subjects
    assert "OAIZIB-CM" in tables.summary()


def test_subject_id_is_the_oai_id_not_the_case_number(cohort):
    """Splits are claimed to be subject-level; that must rest on subject identity."""
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"])
    assert tables.metadata["subject_id"].str.match(r"^9\d{6}$").all()
    assert tables.metadata["case_id"].str.match(r"^oaizib_\d{3}$").all()
    assert tables.metadata["subject_id"].is_unique


def test_missing_kl_can_be_kept(cohort):
    """Keeping them is allowed for uses that do not stratify, e.g. training."""
    tables = load_oaizib_tables(
        cohort["subInfo_train_1"], cohort["subInfo_test_1"], drop_missing_kl=False
    )
    assert tables.n_subjects == 35
    assert tables.metadata["kl_grade"].isna().sum() == 3


def test_blank_and_zero_bmi_both_become_nan(cohort):
    """The .xlsx writes 0 for missing BMI, the .csv leaves it blank."""
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"])
    bmi = tables.metadata["bmi"]
    assert bmi.isna().sum() == 2
    assert not (bmi.dropna() <= 0).any()


def test_site_comes_from_the_release_prefix(cohort):
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"])
    assert set(tables.metadata["site"]) <= {"0.C.2", "0.E.1"}

    unknown = load_oaizib_tables(
        cohort["subInfo_train_1"], cohort["subInfo_test_1"], site_from_release=False
    )
    assert set(unknown.metadata["site"]) == {"unknown"}


# --------------------------------------------------------------------------- #
# The checks that are the point of passing the manifests
# --------------------------------------------------------------------------- #


def test_manifest_table_disagreement_raises(cohort, tmp_path):
    """A manifest listing a case with no subject row is a release mismatch."""
    manifest = pd.read_csv(cohort["train"])
    manifest.loc[len(manifest)] = ["oaizib_999_0000.nii.gz", "oaizib_999.nii.gz"]
    bad = tmp_path / "train_bad.csv"
    manifest.to_csv(bad, index=False)

    with pytest.raises(ValueError, match="disagree"):
        load_oaizib_tables(
            cohort["subInfo_train_1"], cohort["subInfo_test_1"],
            manifest_train=bad, manifest_test=cohort["test"],
        )


def test_mismatched_image_and_mask_raises(tmp_path):
    """Pairing one knee's image with another's mask must not load quietly."""
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {"image": ["oaizib_001_0000.nii.gz"], "mask": ["oaizib_002.nii.gz"]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="different cases"):
        load_case_manifest(path)


def test_duplicate_case_in_manifest_raises(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "image": ["oaizib_001_0000.nii.gz", "oaizib_001_0000.nii.gz"],
            "mask": ["oaizib_001.nii.gz", "oaizib_001.nii.gz"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate case id"):
        load_case_manifest(path)


def test_subject_in_both_splits_raises(tmp_path, cohort):
    """Two scans of one knee straddling train/test breaks exchangeability."""
    train = pd.read_csv(cohort["subInfo_train_1"], dtype={"CMT-ID": str})
    test = pd.read_csv(cohort["subInfo_test_1"], dtype={"CMT-ID": str})
    test.loc[0, "SubjectID"] = train.loc[0, "SubjectID"]
    leaky = tmp_path / "subInfo_test_leaky.csv"
    test.to_csv(leaky, index=False)

    with pytest.raises(ValueError, match="both the train and test tables"):
        load_oaizib_tables(cohort["subInfo_train_1"], leaky)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "subInfo_bad.csv"
    pd.DataFrame({"SubjectID": [9000001], "CMT-ID": ["001"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing column"):
        load_subject_info(path)


def test_duplicate_cmt_id_raises(tmp_path):
    frame = _subinfo_rows(1, [(0, "0.C.2", 25.0), (1, "0.E.1", 26.0)])
    frame.loc[1, "CMT-ID"] = "001"
    path = tmp_path / "subInfo_dupe.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate CMT-ID"):
        load_subject_info(path)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_official_split_holds_the_test_set_whole(cohort):
    """Calibration is carved out of official train; official test is untouched."""
    tables = load_oaizib_tables(
        cohort["subInfo_train_1"], cohort["subInfo_test_1"],
        manifest_train=cohort["train"], manifest_test=cohort["test"],
    )
    meta = tables.metadata
    splits = official_split_assignment(meta, calibration_fraction=0.25, seed=0)

    official_test = set(meta.loc[meta.official_split == "test", "subject_id"])
    assert set(splits["test"]) == official_test

    assert not set(splits["train"]) & set(splits["calibration"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["calibration"]) & set(splits["test"])
    assert sum(len(v) for v in splits.values()) == len(meta)


def test_official_split_never_empties_a_stratum(cohort):
    """A stratum must not be taken whole for calibration, nor left with nothing."""
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"],
                                manifest_train=cohort["train"], manifest_test=cohort["test"])
    splits = official_split_assignment(tables.metadata, calibration_fraction=0.25, seed=0)

    meta = tables.metadata.set_index(tables.metadata.subject_id.astype(str))
    train_pool = meta.loc[meta.official_split == "train"]
    for _key, group in train_pool.groupby(["kl_grade", "site"], dropna=False):
        ids = set(group.index)
        assert ids & set(splits["train"]), "a stratum was taken whole for calibration"


def test_official_split_is_deterministic(cohort):
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"],
                                manifest_train=cohort["train"], manifest_test=cohort["test"])
    a = official_split_assignment(tables.metadata, seed=7)
    b = official_split_assignment(tables.metadata, seed=7)
    assert a == b


def test_official_split_requires_the_column(cohort):
    tables = load_oaizib_tables(cohort["subInfo_train_1"], cohort["subInfo_test_1"])
    stripped = tables.metadata.drop(columns=["official_split"])
    with pytest.raises(ValueError, match="official_split"):
        official_split_assignment(stripped)


def test_canonical_schema_leads_with_the_required_columns(cohort):
    frame = load_subject_info(cohort["subInfo_train_1"], split="train")
    canonical = to_canonical_metadata(frame)
    assert list(canonical.columns[:5]) == [
        "subject_id", "kl_grade", "site", "visit", "laterality",
    ]
