"""Full-spec MH test: can SearchLibrium MH fit the entire larch destination
spec (all flat UEC terms) on the cached work estimation frame?

Runs MH with the full feature set, evaluates the larch coefficients on the
same frame for an apples-to-apples LL, and compares posterior means vs larch.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["STAGE5_DEST_MH_DRAWS"] = os.environ.get("STAGE5_DEST_MH_DRAWS", "4000")
os.environ["STAGE5_DEST_MH_BURN_IN"] = os.environ.get("STAGE5_DEST_MH_BURN_IN", "1000")

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pandas as pd
import cloudpickle


def patch_larch_flow():
    try:
        from larch.dataset import flow as _flow

        def _safe_getattr(self, name):
            if name == "_obj":
                raise AttributeError(name)
            obj = self.__dict__.get("_obj")
            if obj is None:
                raise AttributeError(name)
            result = getattr(obj, name)
            try:
                result = self.transfer_dimension_attrs(result)
            except Exception:
                pass
            return result

        for _cls in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
            _c = getattr(_flow, _cls, None)
            if _c is not None:
                _c.__getattr__ = _safe_getattr
    except Exception as e:
        print(f"larch flow patch skipped: {e!r}")


SEG = os.environ.get("SEG", "work")
CACHE = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_dest_cache.pkl")
OUT = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_mh_full_spec{os.environ.get('MH_TAG','')}.json")
LARCH_RESULTS = os.path.join(
    CODE, "runs", "_s5_larch_perturb",
    f"{SEG}_perturb_results{os.environ.get('LARCH_TAG', '')}.json")

patch_larch_flow()
t0 = time.time()
with open(CACHE, "rb") as fh:
    cache = cloudpickle.load(fh)
print(f"cache loaded in {time.time()-t0:.0f}s; keys={sorted(cache.keys())}")

if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    print("cache has no dt_dest - rebuilding from cached inputs (crash fix live)")
    _dt, _skims = _rebuild_dt(cache)
    try:
        _skims.close()
    except Exception:
        pass
    cache["dt_dest"] = _dt

import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import ChoiceSetFrame

t1 = time.time()
coef, meta = S5._estimate_adaptive_dest_choice(
    seg_tours=cache["seg_acdc"],
    sampled_alts=cache["sampled_alts"],
    landuse=cache["lu_with_st"],
    logsums=cache["logsums"],
    skim_df=None,
    method="mh",
    seed=42,
    dt_dest=cache["dt_dest"],
    segment=SEG,
    full_spec=True,
)
print(f"\nfull-spec MH finished in {time.time()-t1:.0f}s")
print(f"features={meta['n_features']} rows={meta['n_rows']} "
      f"accept={meta['acceptance_rate']:.4f} ll={meta['loglike']:.3f}")
diag = meta.get("diagnostics") or {}
print(f"ess_min={diag.get('ess_min')} burn_in={diag.get('burn_in')}")
inference = meta.get("inference") or {}

# ── evaluate the larch coefficients on the SAME frame ──────────────────────
feature_cols = list(meta["feature_names"])
feature_map = dict(meta["coefficient_map"])          # col -> coef name
coef_to_col = {v: k for k, v in feature_map.items()}

frame = S5._adaptive_dest_choice_frame_full(
    cache["dt_dest"], cache["sampled_alts"], SEG)[0]
csf = ChoiceSetFrame.from_long(frame, feature_cols)

larch_coefs = {}
if os.path.isfile(LARCH_RESULTS):
    larch_coefs = json.load(open(LARCH_RESULTS))["runs"][0]["coefs"]

beta_larch = np.zeros(len(feature_cols))
matched = 0
for i, col in enumerate(feature_cols):
    c = feature_map[col]
    if c in larch_coefs:
        beta_larch[i] = float(larch_coefs[c])
        matched += 1
ll_larch = csf.loglike(beta_larch)
print(f"larch coefficients matched {matched}/{len(feature_cols)} features; "
      f"larch LL on MH frame = {ll_larch:.3f}")

# ── coefficient comparison ─────────────────────────────────────────────────
rows = []
for col in feature_cols:
    c = feature_map[col]
    mh = inference.get(c, {})
    rows.append({
        "coef": c,
        "column": col,
        "mh_posterior_mean": mh.get("value"),
        "mh_std_err": mh.get("std_err"),
        "mh_t": mh.get("tstat"),
        "mh_p": mh.get("pval"),
        "larch": larch_coefs.get(c),
    })
cmp = pd.DataFrame(rows)
cmp["diff"] = cmp["mh_posterior_mean"] - cmp["larch"]
cmp.to_csv(OUT.replace(".json", "_coefs.csv"), index=False)

valid = cmp.dropna(subset=["mh_posterior_mean", "larch"])
r = float(valid["mh_posterior_mean"].corr(valid["larch"])) if len(valid) > 2 else None
sign = float(np.mean(np.sign(valid["mh_posterior_mean"]) ==
                     np.sign(valid["larch"]))) if len(valid) else None
n_sig = int((cmp["mh_t"].abs() >= 1.96).sum())

out = {
    "segment": SEG,
    "full_spec": meta.get("full_spec"),
    "n_features": meta["n_features"],
    "n_rows": meta["n_rows"],
    "acceptance_rate": meta["acceptance_rate"],
    "mh_ll": meta["loglike"],
    "mh_ess_min": diag.get("ess_min"),
    "larch_ll_on_mh_frame": ll_larch,
    "larch_cached_ll": cache.get("dest_ll"),
    "n_larch_coefs_matched": matched,
    "coef_pearson_r": r,
    "sign_agreement": sign,
    "n_mh_terms_sig_1p96": n_sig,
    "draws": int(os.environ["STAGE5_DEST_MH_DRAWS"]),
    "burn_in": int(os.environ["STAGE5_DEST_MH_BURN_IN"]),
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\n=== summary ===")
print(json.dumps(out, indent=2, default=str))
print("\ntop 15 |mh - larch|:")
print(cmp.reindex(cmp["diff"].abs().sort_values(ascending=False).index)
      .head(15)[["coef", "mh_posterior_mean", "mh_std_err", "mh_t",
                 "larch", "diff"]].to_string(index=False))
