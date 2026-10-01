import json

for nm, d in (("A", "results_tests/t4_armA_base"),
              ("B", "results_tests/t4_armB_calib")):
    j = json.load(open(d + "/game_day_holdout_metrics.json"))
    au = j.get("audit", {}) or {}
    m = j.get("metrics", {}) or {}
    lrt = au.get("exposure_lrt") or {}
    print("=== arm %s (%s) ===" % (nm, d))
    print("  exposure_lrt keys :", sorted(lrt.keys()))
    print("  lr_stat           :", lrt.get("lr_stat"))
    print("  p_value           :", lrt.get("p_value"))
    print("  verdict           :", lrt.get("verdict"))
    print("  error             :", lrt.get("error"))
    print("  metrics keys      :", sorted(m.keys()))
    for k in ("bin_spearman", "chi2", "lrt_chi2", "rmse", "mae", "n_future_rows"):
        if k in m:
            print("    %-16s = %s" % (k, m[k]))
    print("  exposure_shape_cols:", au.get("exposure_shape_cols"))
    print()
