# --- subject-level splits ------------------------------------------------------
# On real OAIZIB-CM the default is to respect the split the dataset ships with:
# hold its 103-case test set out whole and carve the conformal calibration split
# out of its 404-case train set. Two reasons, and the second is the one that
# matters for the guarantee:
#
#   * The shipped split is already stratified on KL to within half a point in
#     every stratum (82/21, 46/12, 86/22, 111/28, 58/15), so calibration scores
#     drawn from shipped-train are exchangeable with shipped-test — which is
#     exactly what split conformal needs.
#   * Any comparison against a published OAIZIB-CM number (nnU-Net Dice, say) is
#     otherwise being made on a different test set, and is not a comparison.
#
# Set SPLIT_STRATEGY = "restratify" to ignore the shipped split and re-split the
# pooled cohort; that is the only option for the phantom cohort, which has none.
if USE_REAL_DATA_EFFECTIVE and SPLIT_STRATEGY == "official":
    splits = official_split_assignment(
        metadata,
        calibration_fraction=CALIBRATION_FRACTION,
        seed=SEED,
        stratify_on=("kl_grade", "site"),
    )
    print(f"using the shipped OAIZIB-CM split; calibration is "
          f"{CALIBRATION_FRACTION:.0%} of the shipped train set")
else:
    if USE_REAL_DATA_EFFECTIVE and SPLIT_STRATEGY != "restratify":
        raise ValueError(
            f"SPLIT_STRATEGY must be 'official' or 'restratify', got {SPLIT_STRATEGY!r}"
        )
    splits = make_splits(metadata, fractions=(0.6, 0.2, 0.2), seed=SEED,
                         stratify_on=("kl_grade", "site"))
    print("re-splitting the pooled cohort 60/20/20")

split_report = stratification_report(metadata, splits)

print("\nsplit sizes:", split_report.sizes)
print("realised fractions:", {k: round(v, 3) for k, v in split_report.fractions.items()})
print(f"max |realised - population| proportion deviation: {split_report.max_abs_deviation:.4f}")
print("\nKL x split:\n", split_report.kl_by_split)

assert_disjoint(splits)   # raises on any subject appearing in two splits
SUBJECT_SPLIT = {s: name for name, ids in splits.items() for s in ids}
print("\nno subject leakage between splits ✓")

# Mondrian conformal needs ceil((n+1)(1-alpha)) <= n in every group, i.e.
# n >= 1/alpha - 1. Check it here rather than discovering it as a wall of
# warnings in §7 — with a 32-subject phantom cohort every group failed this, and
# that was a property of the cohort, not of the conformal code.
_min_group = int(np.ceil(1.0 / ALPHA)) - 1
_cal = metadata[metadata.subject_id.isin(splits["calibration"])]
_by_kl = _cal.kl_grade.value_counts().sort_index()
print(f"\ncalibration group sizes by KL (need >= {_min_group} for alpha={ALPHA}):")
for grade, n in _by_kl.items():
    print(f"  KL {grade}: {n:4d}" + ("" if n >= _min_group else "   << too small, pooled threshold"))
