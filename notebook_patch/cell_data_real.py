# --- cohort: real OAIZIB-CM when USE_REAL_DATA, phantoms otherwise -------------
#
# THE BUG THIS REPLACES
# ---------------------
# The original cell declared USE_REAL_DATA and OAIZIB_ROOT and then ignored both:
# it fetched the real *metadata* over the network and immediately overwrote the
# cohort with `make_phantom_cohort(N_PHANTOM, seed=SEED)`. Every number in the
# notebook -- thickness, curvature, bulge, Dice, coverage -- came from 32
# analytic phantoms, while the run manifest recorded "used_real_data": true
# because it read the flag rather than what the flag did. That is the single
# most consequential defect in the run: not a crash, a silent substitution.
#
# Everything downstream is untouched. `knees` still yields objects with
# .image / .label / .spacing and still lines up row-for-row with `metadata`, so
# sections 3-10 need no changes beyond the curvature parameters.
import io
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def fetch_oaizib_metadata(timeout=30):
    """Fetch the public OAI-ZIB subject tables published with CartiMorph."""
    base = "https://raw.githubusercontent.com/YongchengYAO/CartiMorph/main/Dataset/OAIZIB"
    tables = {}
    for name in ("CartiMorph_dataset2", "CartiMorph_dataset3"):   # train + test = 481 subjects
        with urllib.request.urlopen(f"{base}/{name}.xlsx", timeout=timeout) as fh:
            tables[name] = pd.read_excel(io.BytesIO(fh.read()))
    return from_cartimorph_tables(tables)


real_metadata = None
try:
    real_metadata = fetch_oaizib_metadata()
    print(f"fetched real OAI-ZIB metadata: {len(real_metadata)} subjects")
    print("KL distribution:", real_metadata.kl_grade.value_counts().sort_index().to_dict())
    print("sites (OAI image release):", real_metadata.site.value_counts().to_dict())
except Exception as exc:
    print(f"could not fetch OAI-ZIB metadata ({type(exc).__name__}: {exc}).")
    print("Continuing with the phantom cohort — every later section still runs.")


# --------------------------------------------------------------------------- #
# A lazy cohort. 500 knees at 0.5 mm isotropic is ~15 GB of image+label if held
# in RAM; volumes are therefore read on access and one is kept.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class KneeCase:
    """One knee, with the attribute names the rest of the notebook expects."""

    subject_id: str
    image: np.ndarray
    label: np.ndarray
    spacing: tuple


class OAIZIBCohort(Sequence):
    """Lazily-loaded OAIZIB-CM cohort that quacks like the phantom cohort.

    Parameters
    ----------
    cases
        ``CaseRecord``s from :func:`discover_cases`.
    resample_mm
        Resample every volume to this isotropic spacing before analysis, or
        None to analyse on the acquisition grid. OAI DESS is acquired at
        roughly 0.36 x 0.36 x 0.7 mm; marching cubes on a grid that anisotropic
        produces triangles twice as long in z as in x, and the resulting
        curvature carries the grid's anisotropy rather than the anatomy's. This
        is the same isotropic resampling CartiMorph applies before meshing.
    """

    def __init__(self, cases, resample_mm: float | None = 0.5) -> None:
        self._cases = list(cases)
        self._resample_mm = resample_mm
        self._cached_index: int | None = None
        self._cached: KneeCase | None = None

    def __len__(self) -> int:
        return len(self._cases)

    @property
    def subject_ids(self) -> list[str]:
        return [c.subject_id for c in self._cases]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if self._cached_index == index and self._cached is not None:
            return self._cached

        record = self._cases[index]
        image, spacing = load_volume(record.image_path)
        label, label_spacing = load_volume(record.label_path)
        if not np.allclose(label_spacing, spacing, atol=1e-3):
            raise ValueError(
                f"{record.subject_id}: image spacing {spacing} != label spacing "
                f"{label_spacing}. Resampling them independently would misalign the "
                "cartilage plate from the bone it sits on."
            )

        if self._resample_mm is not None and not np.allclose(
            spacing, self._resample_mm, atol=1e-3
        ):
            target = (self._resample_mm,) * 3
            image = resample_volume(image, spacing, target, is_label=False)
            label = resample_volume(label, spacing, target, is_label=True)
            spacing = target

        case = KneeCase(
            subject_id=record.subject_id,
            image=np.asarray(image, dtype=np.float32),
            label=np.asarray(label, dtype=np.int16),
            spacing=tuple(float(s) for s in spacing),
        )
        self._cached_index, self._cached = index, case
        return case


def _find_oaizib_layout(root):
    """Locate the image/label directories, whatever the release called them.

    OAIZIB-CM ships nnU-Net style, but different mirrors split it differently
    (imagesTr/labelsTr only, or plus imagesTs/labelsTs). Rather than fail on a
    naming mismatch, look for every pair that exists and say what was found.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"OAIZIB_ROOT does not exist: {root}\n"
            "Set it to the directory that contains imagesTr/ and labelsTr/."
        )
    pairs = [
        (i, l)
        for i, l in (("imagesTr", "labelsTr"), ("imagesTs", "labelsTs"),
                     ("images", "labels"))
        if (root / i).is_dir() and (root / l).is_dir()
    ]
    if not pairs:
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"no image/label directory pair under {root}. Sub-directories present: "
            f"{found}. Expected imagesTr/ + labelsTr/ (nnU-Net layout)."
        )
    return pairs


USE_REAL_DATA_EFFECTIVE = False
knees = metadata = None

if USE_REAL_DATA:
    if real_metadata is None:
        raise RuntimeError(
            "USE_REAL_DATA=True but the OAI-ZIB metadata could not be fetched. Every "
            "subject needs a KL grade and a site or it cannot be stratified, split or "
            "used for conditional coverage. Download CartiMorph_dataset2.xlsx and "
            "CartiMorph_dataset3.xlsx by hand and load them with "
            "from_cartimorph_tables({'d2': pd.read_excel(...), 'd3': pd.read_excel(...)})."
        )

    records = []
    for images_dir, labels_dir in _find_oaizib_layout(OAIZIB_ROOT):
        found = discover_cases(
            OAIZIB_ROOT, images_dir=images_dir, labels_dir=labels_dir,
            require_labels=True,
        )
        print(f"  {images_dir}/ + {labels_dir}/: {len(found)} case(s)")
        records.extend(found)

    # Join on subject id. The image set and the published subject tables do not
    # have to agree exactly (OAI-ZIB is 507 scans, the CartiMorph tables cover
    # 481), and a case without a KL grade cannot enter any split -- so the
    # intersection is the cohort, and the losses are reported rather than
    # dropped in silence.
    have_metadata = set(real_metadata.subject_id.astype(str))
    kept = [r for r in records if r.subject_id in have_metadata]
    dropped = sorted({r.subject_id for r in records} - have_metadata)
    if dropped:
        print(f"  {len(dropped)} case(s) on disk have no metadata row and are excluded "
              f"(e.g. {dropped[:5]})")
    missing_images = sorted(have_metadata - {r.subject_id for r in kept})
    if missing_images:
        print(f"  {len(missing_images)} metadata subject(s) have no image on disk "
              f"(e.g. {missing_images[:5]})")
    if not kept:
        raise RuntimeError(
            "no case on disk matched a metadata subject id. discover_cases parses the "
            "7-digit OAI id out of the filename; check that your files look like "
            "'9001104_V00.nii.gz' or 'sub-9001104_....nii.gz'."
        )

    if N_SUBJECTS is not None:
        kept = kept[:N_SUBJECTS]
        print(f"  restricted to the first {len(kept)} case(s) (N_SUBJECTS)")

    knees = OAIZIBCohort(kept, resample_mm=RESAMPLE_TO_MM)
    # metadata must be row-for-row aligned with `knees`: every later cell pairs
    # them with zip(knees, metadata.iterrows()).
    metadata = (
        real_metadata.set_index(real_metadata.subject_id.astype(str))
        .loc[[c.subject_id for c in kept]]
        .reset_index(drop=True)
    )
    USE_REAL_DATA_EFFECTIVE = True

    probe = knees[0]
    print(f"\nOAIZIB-CM cohort: {len(knees)} knees, volume {probe.label.shape} @ "
          f"{probe.spacing[0]:.3f} mm"
          + ("" if RESAMPLE_TO_MM is None else f" (resampled to {RESAMPLE_TO_MM} mm isotropic)"))
    print("labels present in the first case:", np.unique(probe.label).tolist())
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
