# --- cohort: real OAIZIB-CM when USE_REAL_DATA, phantoms otherwise -------------
#
# TWO BUGS THIS REPLACES
# ----------------------
# 1. The original cell declared USE_REAL_DATA and OAIZIB_ROOT and then ignored
#    both: it fetched the real *metadata* over the network and immediately
#    overwrote the cohort with `make_phantom_cohort(N_PHANTOM, seed=SEED)`. Every
#    number in the notebook came from 32 analytic phantoms, while the run
#    manifest recorded "used_real_data": true because it read the flag rather
#    than what the flag did.
#
# 2. Even with the flag honoured, the generic loader could not have found the
#    files. OAIZIB-CM ships `imagesTr/oaizib_001_0000.nii.gz` and
#    `labelsTr/oaizib_001.nii.gz` — a sequential case number, not the 7-digit OAI
#    subject id, and nnU-Net's `_0000` modality suffix on the image but not the
#    label. `discover_cases` parses `(?:sub-)?(\d{7})`, finds no seven-digit run,
#    falls back to the whole stem, and then keys the image as `oaizib_001_0000`
#    and the label as `oaizib_001` — so nothing joins and it raises
#    "507 image(s) have no matching label" on a dataset where every image has one.
#
#    The bridge is the CMT-ID column of the shipped subject tables: CMT-ID *is*
#    the case number. That join is what this cell makes, and checks.
#
# Everything downstream is untouched. `knees` still yields objects with
# .image / .label / .spacing and still lines up row-for-row with `metadata`.
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _under_root(name, root):
    """Resolve a configured table name against OAIZIB_ROOT unless it is absolute."""
    if name is None:
        return None
    path = Path(name)
    return path if path.is_absolute() else Path(root) / path


# --------------------------------------------------------------------------- #
# A lazy cohort. 481 knees at 0.5 mm isotropic is ~15 GB of image+label if held
# in RAM; volumes are therefore read on access and one is kept.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class KneeCase:
    """One knee, with the attribute names the rest of the notebook expects."""

    subject_id: str
    case_id: str
    image: np.ndarray
    label: np.ndarray
    spacing: tuple


class OAIZIBCohort(Sequence):
    """Lazily-loaded OAIZIB-CM cohort that quacks like the phantom cohort.

    Parameters
    ----------
    records
        ``(subject_id, case_id, image_path, label_path)`` tuples, in the same
        order as the metadata table.
    resample_mm
        Resample every volume to this isotropic spacing before analysis, or None
        to analyse on the acquisition grid. OAI DESS is acquired at roughly
        0.36 x 0.36 x 0.7 mm; marching cubes on a grid that anisotropic produces
        triangles twice as long in z as in x, and the resulting curvature carries
        the grid's anisotropy rather than the anatomy's. This is the same
        isotropic resampling CartiMorph applies before meshing.
    """

    def __init__(self, records, resample_mm: float | None = 0.5) -> None:
        self._records = list(records)
        self._resample_mm = resample_mm
        self._cached_index: int | None = None
        self._cached: KneeCase | None = None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def subject_ids(self) -> list[str]:
        return [r[0] for r in self._records]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if self._cached_index == index and self._cached is not None:
            return self._cached

        subject_id, case_id, image_path, label_path = self._records[index]
        image, spacing = load_volume(image_path)
        label, label_spacing = load_volume(label_path)
        if not np.allclose(label_spacing, spacing, atol=1e-3):
            raise ValueError(
                f"{case_id}: image spacing {spacing} != label spacing {label_spacing}. "
                "Resampling them independently would misalign the cartilage plate "
                "from the bone it sits on."
            )

        if self._resample_mm is not None and not np.allclose(
            spacing, self._resample_mm, atol=1e-3
        ):
            target = (self._resample_mm,) * 3
            image = resample_volume(image, spacing, target, is_label=False)
            label = resample_volume(label, spacing, target, is_label=True)
            spacing = target

        case = KneeCase(
            subject_id=str(subject_id),
            case_id=str(case_id),
            image=np.asarray(image, dtype=np.float32),
            label=np.asarray(label, dtype=np.int16),
            spacing=tuple(float(s) for s in spacing),
        )
        self._cached_index, self._cached = index, case
        return case


def _locate_volumes(root, case_ids, manifest):
    """Find the image and label file for every case id, wherever the release put them.

    OAIZIB-CM ships nnU-Net style, but mirrors differ on whether the test split
    lives in imagesTs/labelsTs or is folded into imagesTr/labelsTr. Rather than
    guess, index every candidate directory by case id and look each case up.
    """
    root = Path(root)
    image_index, label_index = {}, {}
    searched = []
    for images_dir, labels_dir in (
        ("imagesTr", "labelsTr"), ("imagesTs", "labelsTs"), ("images", "labels")
    ):
        for directory, index in ((images_dir, image_index), (labels_dir, label_index)):
            path = root / directory
            if not path.is_dir():
                continue
            searched.append(directory)
            for volume in path.glob("*.nii*"):
                index.setdefault(case_id_from_filename(volume), volume)

    if not image_index:
        present = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        raise FileNotFoundError(
            f"no image directory under {root}. Sub-directories present: {present}. "
            "Expected an nnU-Net layout with imagesTr/ and labelsTr/."
        )

    records, missing = [], []
    for case_id in case_ids:
        image, label = image_index.get(case_id), label_index.get(case_id)
        if image is None or label is None:
            missing.append(case_id)
            continue
        records.append((case_id, image, label))

    print(f"  searched {sorted(set(searched))}: {len(image_index)} image(s), "
          f"{len(label_index)} label(s) on disk")
    if missing:
        print(f"  {len(missing)} case(s) in the tables have no image/label pair on disk "
              f"(e.g. {missing[:5]})")
    del manifest      # the manifests are used for the split, not for path resolution
    return records, missing


USE_REAL_DATA_EFFECTIVE = False
knees = metadata = oaizib_tables = None
real_metadata = None

if USE_REAL_DATA:
    subinfo_train = _under_root(SUBINFO_TRAIN, OAIZIB_ROOT)
    if subinfo_train is None or not subinfo_train.is_file():
        raise FileNotFoundError(
            f"subject table not found: {subinfo_train}\n"
            "SUBINFO_TRAIN must point at subInfo_train_1.csv — it carries the CMT-ID "
            "column, which is the only link between the oaizib_NNN filenames and the "
            "OAI subject ids, KL grades and sites."
        )

    oaizib_tables = load_oaizib_tables(
        subinfo_train,
        _under_root(SUBINFO_TEST, OAIZIB_ROOT),
        manifest_train=_under_root(MANIFEST_TRAIN, OAIZIB_ROOT),
        manifest_test=_under_root(MANIFEST_TEST, OAIZIB_ROOT),
        drop_missing_kl=DROP_MISSING_KL,
    )
    print(oaizib_tables.summary())
    real_metadata = oaizib_tables.metadata

    records, _missing = _locate_volumes(
        OAIZIB_ROOT, real_metadata["case_id"].tolist(), oaizib_tables.manifest
    )
    if not records:
        raise RuntimeError(
            "no case in the subject tables has a matching image/label pair on disk. "
            "Check OAIZIB_ROOT, and that the volumes are named oaizib_NNN_0000.nii.gz "
            "(images) and oaizib_NNN.nii.gz (labels)."
        )

    found = {case_id for case_id, _i, _l in records}
    metadata = (
        real_metadata[real_metadata["case_id"].isin(found)]
        .reset_index(drop=True)
    )
    if N_SUBJECTS is not None:
        metadata = metadata.iloc[:N_SUBJECTS].reset_index(drop=True)
        print(f"  restricted to the first {len(metadata)} case(s) (N_SUBJECTS)")

    # `knees` must line up row-for-row with `metadata`: every later cell pairs
    # them by index.
    by_case = {case_id: (image, label) for case_id, image, label in records}
    knees = OAIZIBCohort(
        [
            (row.subject_id, row.case_id, *by_case[row.case_id])
            for row in metadata.itertuples()
        ],
        resample_mm=RESAMPLE_TO_MM,
    )
    USE_REAL_DATA_EFFECTIVE = True

    probe = knees[0]
    print(f"\nOAIZIB-CM cohort: {len(knees)} knees, volume {probe.label.shape} @ "
          f"{probe.spacing[0]:.3f} mm"
          + ("" if RESAMPLE_TO_MM is None else f" (resampled to {RESAMPLE_TO_MM} mm isotropic)"))
    print(f"first case: {probe.case_id} = OAI subject {probe.subject_id}")
    print("labels present:", np.unique(probe.label).tolist(), "(expect 0-5)")
else:
    knees, metadata = make_phantom_cohort(N_PHANTOM, seed=SEED)
    print(f"\nphantom cohort: {len(knees)} knees, volume {knees[0].label.shape} @ "
          f"{knees[0].spacing[0]} mm")
    print("NOTE: these are analytic phantoms. Nothing below is a statement about "
          "real knees until USE_REAL_DATA is True and OAIZIB_ROOT points at data.")

report = validate_metadata(metadata)
print("\nmetadata valid:", report.ok, "| KL:", report.kl_counts, "| sites:", report.site_counts)
for w in report.warnings[:3]:
    print("  warning:", w)
