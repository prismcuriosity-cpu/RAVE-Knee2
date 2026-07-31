# --- configuration for this run ------------------------------------------------
USE_REAL_DATA = True                  # True if you hold OAI-ZIB / OAIZIB-CM locally
OAIZIB_ROOT   = r"D:\ishtiaque\Knee segementation\OAI-ZIB\OAIZIB-CM\OAIZIB-CM"
N_SUBJECTS    = None                  # None = every case found; set an int for a smoke test
RESAMPLE_TO_MM = 0.5                  # isotropic resample before meshing; None = acquisition grid
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
