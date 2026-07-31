# --- configuration for this run ------------------------------------------------
USE_REAL_DATA = True                  # True if you hold OAI-ZIB / OAIZIB-CM locally
OAIZIB_ROOT   = r"D:\ishtiaque\Knee segementation\OAI-ZIB\OAIZIB-CM\OAIZIB-CM"
N_SUBJECTS    = None                  # None = every case found; set an int for a smoke test
RESAMPLE_TO_MM = 0.5                  # isotropic resample before meshing; None = acquisition grid

# --- metadata -------------------------------------------------------------------
# The subject tables and split manifests shipped inside OAIZIB_ROOT. These are the
# authoritative metadata: they carry the CMT-ID column, which is the only link
# between a filename (oaizib_497_0000.nii.gz) and an OAI subject (9695686) — the
# filenames contain a sequential case number, not the 7-digit OAI id, so nothing
# joins without them.
#
# Paths are resolved relative to OAIZIB_ROOT when they are not absolute. Set a
# name to None to skip it; without the manifests the official train/test split is
# unavailable and SPLIT_STRATEGY must be "restratify".
SUBINFO_TRAIN = "subInfo_train_1.csv"
SUBINFO_TEST  = "subInfo_test_1.csv"
MANIFEST_TRAIN = "train.csv"
MANIFEST_TEST  = "test.csv"

# "official"    — hold out the shipped test set (103 cases) whole and carve the
#                 calibration split out of the shipped train set (404). Preferred:
#                 the shipped split is stratified on KL to within half a point in
#                 every stratum, so calibration and test scores are exchangeable,
#                 and any comparison against a published OAIZIB-CM number is then
#                 made on the same test set.
# "restratify"  — ignore the shipped split and re-split the pooled cohort.
SPLIT_STRATEGY = "official"
CALIBRATION_FRACTION = 0.25           # share of the shipped train set held for calibration

# 26 of the 507 subjects have no KL grade (21 train, 5 test). 507 - 26 = 481, which
# is exactly the row count of the CartiMorph tables published on GitHub — the public
# release drops the same subjects. They cannot be stratified, split, or given
# conditional coverage, so they are excluded here too.
DROP_MISSING_KL = True

N_PHANTOM     = 32                    # synthetic cohort size when USE_REAL_DATA is False
N_RELIABILITY = 8                     # subjects re-measured for precision/agreement
REPEATS       = 3                     # repeat measurements per subject
SEED          = 20240617
ALPHA         = 0.10                  # target miscoverage -> 90% intervals

# --- curvature ------------------------------------------------------------------
# CURVATURE_RADIUS_MM is the spatial scale the curvature is measured at. The
# original run also carried an implicit `2 * radius` = 6.0 mm boundary margin
# that discarded 76-87% of every cartilage plate; that margin is gone, replaced
# by a per-vertex support test (see confcarti/thickness/curvature.py). What
# remains here is the scale itself, plus the floor the adaptive fallback may
# shrink to near a rim.
CURVATURE_RADIUS_MM      = 3.0
CURVATURE_MIN_RADIUS_MM  = 1.5        # ~3 marching-cubes edge lengths at 0.5 mm
CURVATURE_ADAPTIVE       = True       # False = strict single-scale field

# --- segmentation ----------------------------------------------------------------
# 64 x 64 x 48 at 0.5 mm is a 32 x 32 x 24 mm box: it clipped ~50% of the
# cartilage plate on every case in the first run ("joint-centred crop ... clipped
# N cartilage voxel(s) (52.00% of the plate)"). A knee needs ~80 x 80 x 48 mm.
CROP_SIZE     = (160, 160, 96) if USE_REAL_DATA else (64, 64, 48)
MAX_EPOCHS    = 500 if USE_REAL_DATA else 6

seed_report = set_all_seeds(SEED)
print(seed_report)
print(f"cohort: {'OAIZIB-CM at ' + str(OAIZIB_ROOT) if USE_REAL_DATA else str(N_PHANTOM) + ' phantoms'}")
