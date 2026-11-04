"""
synth_hhts_cvae.py — Conditional augmentation of QLD QTS-HHTS trips.

Retains individual characteristics by learning p(x | c), not p(x):

  c (fixed per person, from data_connect.py linkage):
    AGEGROUP, SEX, WORKSTATUS, MAINACT, CARLICENCE, HHSIZE, HHVEH, OTAZ(home)
  x (synthesised per trip — user-requested 7):
    purpose (OVERALL coded), mode (MAINMODE coded), STARTIME,
    TRAVTIME, cumdist, tour_primary_purpose, DTAZ

Two generators in one file:
  1. empirical (default, no torch): stratified hot-deck — copies a donor
     trip's full x-vector within the same individual stratum, then jitters
     continuous cols. Preserves joints exactly. Runs anywhere.
  2. cvae: DISABLED (raises). The previous version silently trained a CVAE
     then discarded it and sampled empirical while writing
     synth_trips_cvae_*.csv — mislabelled output. Re-enable only with a real
     mixed-type decoder head. NOTE: this file is dormant/experimental — the
     pipeline estimates on fully real HHTS and never reads its outputs.

Linkage mirrors code/data_connect.py:
  saved_data/trips_processed.csv (person_id=PERSID, HHID)
    x ../2_QTS_PERSONS.xlsx  x ../1_QTS_HOUSEHOLDS.xlsx

Usage:
  python synth_hhts_cvae.py --mode empirical --k 3 --max-rows 5000
  python synth_hhts_cvae.py --mode cvae --epochs 30 --z-dim 32 --k 3
  Outputs: saved_data/synth_hhts/synth_trips_<mode>_x<k>.csv + metrics to stdout.

PIPELINE ISOLATION (estimation stays fully real HHTS):
  - pipeline_logger.load_data() rebuilds HHTS fresh via
    Synth_Connect.process_survey(save=False) from the raw QTS xlsx files.
    It never reads saved_data/trips_processed.csv and never globs
    saved_data/*.csv, so this script's outputs cannot leak into estimation.
  - This script is import-safe (all work under `if __name__ == "__main__"`),
    imports only pandas/numpy (+torch lazily for --mode cvae), never imports
    or patches data_connect/pipeline_logger, and never writes
    trips_processed.csv / tours_processed.csv / persons_processed.csv.
  - Synthetic output lives in isolated saved_data/synth_hhts/ (not the
    pipeline's synth_population / synth_persons paths) and is opt-in only:
    no .pbs job, stage, or load_data() references it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from itertools import combinations

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_ROOT = os.path.normpath(os.path.join(CODE_DIR, ".."))
SAVED = os.path.join(CODE_DIR, "saved_data")
# Isolated opt-in output dir — deliberately NOT scanned by the pipeline.
SYNTH_OUT_DIR = os.path.join(SAVED, "synth_hhts")

# ---------------------------------------------------------------- targets
X_CAT = ["purpose", "mode", "tour_primary_purpose", "DTAZ_bin"]
X_CONT = ["STARTIME", "TRAVTIME", "cumdist"]
C_COLS_RAW = ["AGEGROUP", "SEX", "WORKSTATUS", "MAINACT",
              "CARLICENCE", "HHSIZE", "HHVEH", "OTAZ"]

TOP_DTAZ = 30  # keep top-30 DTAZ + OTHER (313 zones -> tractable)


# ---------------------------------------------------------------- loading
def load_hhts(max_rows=None):
    """Join trips_processed to QTS persons+households (data_connect keys)."""
    tr = pd.read_csv(os.path.join(SAVED, "trips_processed.csv"),
                     usecols=["person_id", "HHID", "STARTIME", "TRAVTIME",
                              "cumdist", "purpose", "mode",
                              "tour_primary_purpose", "DTAZ", "OTAZ"],
                     nrows=max_rows)
    pp = pd.read_excel(os.path.join(TRT_ROOT, "2_QTS_PERSONS.xlsx"),
                       engine="openpyxl",
                       usecols=["HHID", "PERSID", "AGEGROUP", "SEX",
                                "WORKSTATUS", "MAINACT", "CARLICENCE"])
    hh = pd.read_excel(os.path.join(TRT_ROOT, "1_QTS_HOUSEHOLDS.xlsx"),
                       engine="openpyxl",
                       usecols=["HHID", "HHSIZE", "HHVEH"])
    df = tr.merge(pp, left_on=["person_id", "HHID"],
                  right_on=["PERSID", "HHID"], how="left")
    df = df.merge(hh, on="HHID", how="left")

    # ---- clean x ----
    # purpose NaN (~18% in trips_processed: OVERALL_PURPOSE values that missed
    # the data_connect map, e.g. 'Go home' legs) -> explicit 'home' level so
    # the generator learns it instead of dropping rows.
    df["purpose"] = df["purpose"].fillna("home")
    df["tour_primary_purpose"] = df["tour_primary_purpose"].fillna("Other Purpose")
    df["mode"] = df["mode"].fillna("DA")
    df["STARTIME"] = pd.to_numeric(df["STARTIME"], errors="coerce").fillna(510).clip(0, 1439)
    df["TRAVTIME"] = pd.to_numeric(df["TRAVTIME"], errors="coerce").fillna(15).clip(1, 180)
    df["cumdist"] = pd.to_numeric(df["cumdist"], errors="coerce").fillna(5.3).clip(0, 100)
    df["DTAZ"] = pd.to_numeric(df["DTAZ"], errors="coerce").fillna(999)
    df["OTAZ"] = pd.to_numeric(df["OTAZ"], errors="coerce").fillna(999)

    top = df["DTAZ"].value_counts().head(TOP_DTAZ).index
    df["DTAZ_bin"] = df["DTAZ"].where(df["DTAZ"].isin(top), "OTHER")
    df["DTAZ_bin"] = df["DTAZ_bin"].astype(str)

    # ---- clean c (individual, kept fixed at generation) ----
    df["AGEGROUP"] = pd.to_numeric(df["AGEGROUP"], errors="coerce").fillna(10).astype(int).clip(1, 21)
    df["SEX"] = df["SEX"].fillna("male").astype(str)
    df["WORKSTATUS"] = df["WORKSTATUS"].fillna("otherReason").astype(str)
    df["MAINACT"] = df["MAINACT"].fillna("Other Activity").astype(str)
    car = df["CARLICENCE"]
    if car.dtype == object:
        car = car.map({True: 1, False: 0, "True": 1, "False": 0,
                       "TRUE": 1, "FALSE": 0}).fillna(0)
    df["CARLICENCE"] = pd.to_numeric(car, errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["HHSIZE"] = pd.to_numeric(df["HHSIZE"], errors="coerce").fillna(2).astype(int).clip(1, 8)
    df["HHVEH"] = pd.to_numeric(df["HHVEH"], errors="coerce").fillna(1).astype(int).clip(0, 4)
    return df


def make_stratum(df):
    """Coarse individual stratum: age x work x veh x sex (for empirical mode)."""
    age = pd.cut(df["AGEGROUP"], bins=[0, 4, 7, 10, 14, 22],
                 labels=["child", "youth", "young", "mid", "older"]).astype(str)
    veh = df["HHVEH"].clip(0, 2).astype(str)
    return (age + "|" + df["WORKSTATUS"].astype(str) + "|v" + veh
            + "|" + df["SEX"].astype(str))


# ---------------------------------------------------------------- metrics
def _dist(s):
    v = s.value_counts(normalize=True).sort_index()
    return v


def kld(p, q, eps=1e-10):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def srmse_2way(real, gen, cols):
    # Joint-fit metric in the spirit of UMN utils.py. NOTE nonstandard
    # denominator: rmse is scaled by 1/n_cells (uniform-share baseline),
    # so the value grows with the number of joint cells — compare SRMSE
    # values only between runs over the SAME column sets, never across
    # different combos. Lower = better within a combo.
    rc = real.groupby(list(cols)).size()
    gc = gen.groupby(list(cols)).size()
    idx = rc.index.union(gc.index)
    r = rc.reindex(idx, fill_value=0).values.astype(float)
    g = gc.reindex(idx, fill_value=0).values.astype(float)
    r /= r.sum()
    g /= g.sum()
    nb = len(r)
    rmse = np.sqrt(np.sum((r - g) ** 2) / nb)
    return float(rmse / (r.sum() / nb))


def evaluate(real, gen, cols):
    print("\n== marginal fit (real vs synthetic) ==")
    klds = {}
    for c in cols:
        cats = sorted(set(real[c].astype(str)) | set(gen[c].astype(str)))
        p = real[c].astype(str).value_counts().reindex(cats, fill_value=0).values.astype(float)
        q = gen[c].astype(str).value_counts().reindex(cats, fill_value=0).values.astype(float)
        p /= p.sum()
        q /= q.sum()
        mse = float(np.mean((p - q) ** 2))
        k = kld(p, q)
        klds[c] = k
        print(f"  {c:22s} MSE={mse:.6f} KLD={k:.5f} (levels={len(cats)})")
    combos = list(combinations([c for c in cols if c in
                                ["purpose", "mode", "tour_primary_purpose", "DTAZ_bin"]], 2))
    print("\n== joint SRMSE (2-way, lower=better) ==")
    sr = {}
    for cb in combos:
        v = srmse_2way(real, gen, list(cb))
        sr[cb] = v
        print(f"  {cb}: {v:.4f}")
    print(f"\n  mean KLD={np.mean(list(klds.values())):.5f} "
          f"mean SRMSE={np.mean(list(sr.values())):.4f}")
    return klds, sr


# ------------------------------------------------------- empirical sampler
class EmpiricalConditional:
    """p(x|c_stratum) via stratified hot-deck: copy a donor trip's full x
    vector within the same individual stratum, then jitter continuous cols.
    Preserves joint structure (purpose|tour|mode|DTAZ) exactly, unlike
    independent per-column sampling."""

    def __init__(self, seed=42, jitter=0.05):
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter
        self.strata = {}

    def fit(self, df):
        df = df.copy()
        df["_s"] = make_stratum(df)
        # donor pools per stratum for hot-deck joint sampling (nothing else
        # is fitted — earlier Laplace/Gaussian bookkeeping was dead code)
        self.fallback_pool = df[X_CAT + ["TRAVTIME", "cumdist", "STARTIME", "DTAZ"]].copy()
        for s, g in df.groupby("_s"):
            self.strata[s] = {"n": len(g),
                              "pool": g[X_CAT + ["TRAVTIME", "cumdist",
                                                "STARTIME", "DTAZ"]].copy()}
        print(f"[empirical] fitted {len(self.strata)} strata from {len(df)} trips")

    def sample_for(self, crow):
        s = make_stratum(crow.to_frame().T).iloc[0]
        st = self.strata.get(s)
        pool = st["pool"] if (st is not None and st["n"] >= 5) else self.fallback_pool
        donor = pool.iloc[int(self.rng.integers(0, len(pool)))]
        out = {c: str(donor[c]) for c in X_CAT}
        # jitter continuous donor values (keeps joint, adds variety)
        out["TRAVTIME"] = float(np.clip(
            float(donor["TRAVTIME"]) * float(self.rng.lognormal(0, self.jitter)), 1, 180))
        out["cumdist"] = float(np.clip(
            float(donor["cumdist"]) * float(self.rng.lognormal(0, self.jitter)), 0, 100))
        out["STARTIME"] = int(np.clip(
            int(donor["STARTIME"]) + int(self.rng.normal(0, 15)), 0, 1439))
        out["_donor_DTAZ"] = donor["DTAZ"]
        return out


def generate_empirical(df, model, k=3):
    df = df.copy()
    df["_s"] = make_stratum(df)
    df["_i"] = np.arange(len(df))
    out_frames = []
    rep_ids = []
    for s, idx in df.groupby("_s").indices.items():
        st = model.strata.get(s)
        pool = st["pool"] if (st is not None and st["n"] >= 5) else model.fallback_pool
        n_need = len(idx) * k
        pick = model.rng.integers(0, len(pool), size=n_need)
        donors = pool.iloc[pick].reset_index(drop=True)
        base = df.iloc[np.repeat(idx, k)].reset_index(drop=True)
        g = pd.DataFrame({
            "person_id": base["person_id"], "HHID": base["HHID"],
            "AGEGROUP": base["AGEGROUP"], "SEX": base["SEX"],
            "WORKSTATUS": base["WORKSTATUS"], "MAINACT": base["MAINACT"],
            "CARLICENCE": base["CARLICENCE"], "HHSIZE": base["HHSIZE"],
            "HHVEH": base["HHVEH"], "OTAZ": base["OTAZ"],
            "purpose": donors["purpose"].astype(str).values,
            "mode": donors["mode"].astype(str).values,
            "tour_primary_purpose": donors["tour_primary_purpose"].astype(str).values,
            "DTAZ_bin": donors["DTAZ_bin"].astype(str).values,
            "DTAZ": pd.to_numeric(donors["DTAZ"], errors="coerce").fillna(999).values,
            "STARTIME": pd.to_numeric(donors["STARTIME"], errors="coerce").fillna(510).values,
            "TRAVTIME": pd.to_numeric(donors["TRAVTIME"], errors="coerce").fillna(15).values,
            "cumdist": pd.to_numeric(donors["cumdist"], errors="coerce").fillna(5.3).values,
            "rep": np.tile(np.arange(k), len(idx)),
        })
        # jitter continuous (vectorised)
        g["TRAVTIME"] = np.clip(
            g["TRAVTIME"].values * model.rng.lognormal(0, model.jitter, size=len(g)), 1, 180)
        g["cumdist"] = np.clip(
            g["cumdist"].values * model.rng.lognormal(0, model.jitter, size=len(g)), 0, 100)
        g["STARTIME"] = np.clip(
            g["STARTIME"].values + model.rng.normal(0, 15, size=len(g)), 0, 1439).round().astype(int)
        out_frames.append(g)
    gen = pd.concat(out_frames, ignore_index=True)
    gen["DTAZ"] = pd.to_numeric(gen["DTAZ"], errors="coerce").fillna(999).astype(int)
    return gen


# ------------------------------------------------------------- CVAE (torch)
def run_cvae(df, epochs=30, z_dim=32, hidden=256, seed=0):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch not installed -> cannot run --mode cvae. "
              "Use --mode empirical, or pip install torch (CPU).")
        sys.exit(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_levels = {c: sorted(df[c].astype(str).unique()) for c in X_CAT}
    cat_idx = {c: {v: i for i, v in enumerate(vs)} for c, vs in cat_levels.items()}
    # condition coding: numeric + one-hot for low-card strings
    c_num = ["AGEGROUP", "CARLICENCE", "HHSIZE", "HHVEH", "OTAZ"]
    c_cat_cols = ["SEX", "WORKSTATUS"]
    c_levels = {c: sorted(df[c].astype(str).unique()) for c in c_cat_cols}

    def encode_x(d):
        parts = []
        for c in X_CAT:
            oh = np.zeros((len(d), len(cat_levels[c])), np.float32)
            ii = d[c].astype(str).map(cat_idx[c]).fillna(0).astype(int).values
            oh[np.arange(len(d)), ii] = 1.0
            parts.append(oh)
        mu_, sd_ = df[X_CONT].mean(), df[X_CONT].std().replace(0, 1)
        parts.append(((d[X_CONT] - mu_) / sd_).values.astype(np.float32))
        return np.concatenate(parts, axis=1), (mu_, sd_)

    def encode_c(d):
        parts = [d[c_num].astype(float).values.astype(np.float32)]
        for c in c_cat_cols:
            lv = c_levels[c]
            m = {v: i for i, v in enumerate(lv)}
            oh = np.zeros((len(d), len(lv)), np.float32)
            ii = d[c].astype(str).map(m).fillna(0).astype(int).values
            oh[np.arange(len(d)), ii] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1)

    X, _ = encode_x(df)
    C = encode_c(df)
    # dequantize one-hot blocks (UMN utils.py trick) with small uniform noise
    X = X + np.random.uniform(0, 0.1, size=X.shape).astype(np.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)

    class CVAE(nn.Module):
        def __init__(self, xd, cd, hd, zd):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(xd + cd, hd), nn.ReLU(),
                                     nn.Linear(hd, hd), nn.ReLU())
            self.mu = nn.Linear(hd, zd)
            self.lv = nn.Linear(hd, zd)
            self.dec = nn.Sequential(nn.Linear(zd + cd, hd), nn.ReLU(),
                                     nn.Linear(hd, hd), nn.ReLU(),
                                     nn.Linear(hd, xd))

        def forward(self, x, c):
            h = self.enc(torch.cat([x, c], 1))
            mu, lv = self.mu(h), self.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            return self.dec(torch.cat([z, c], 1)), mu, lv

    model = CVAE(X.shape[1], C.shape[1], hidden, z_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Ct), batch_size=512, shuffle=True)
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for xb, cb in loader:
            xb, cb = xb.to(dev), cb.to(dev)
            opt.zero_grad()
            recon, mu, lv = model(xb, cb)
            rl = nn.functional.mse_loss(recon, xb, reduction="sum")
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            (rl + kl).backward()
            opt.step()
            tot += float(rl + kl)
        if ep == 1 or ep % 5 == 0:
            print(f"[cvae] epoch {ep}/{epochs} loss={tot:.1f}")
    return model, (cat_levels, c_levels, c_num, c_cat_cols)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["empirical", "cvae"], default="empirical")
    ap.add_argument("--k", type=int, default=3, help="synthetic replicates per trip")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.mode == "cvae":
        raise SystemExit(
            "ERROR: --mode cvae is disabled. The previous implementation trained "
            "a CVAE then silently discarded it and sampled the empirical donor "
            "pool while labelling output 'cvae' — mislabelled results. Re-enable "
            "only after implementing a real mixed-type decoder head. "
            "Use --mode empirical (labelled honestly) or leave this "
            "experimental file unused: estimation runs on fully real HHTS.")

    df = load_hhts(a.max_rows)
    print(f"loaded {len(df)} trips | persons={df.person_id.nunique()} "
          f"| purpose levels={df.purpose.nunique()} mode={df['mode'].unique().tolist()}")

    if a.mode == "empirical":
        model = EmpiricalConditional(seed=a.seed)
        model.fit(df)
        gen = generate_empirical(df, model, k=a.k)
    else:  # unreachable: cvae exits above; kept so a future mode errors loudly
        raise SystemExit(f"ERROR: unknown mode '{a.mode}'")

    eval_cols = ["purpose", "mode", "tour_primary_purpose", "DTAZ_bin"]
    evaluate(df, gen, eval_cols)
    # conditional check: does each synthetic row keep its person's real c?
    person_c = df.drop_duplicates("person_id")[["person_id"] + C_COLS_RAW]
    ck = gen.merge(person_c, on="person_id", suffixes=("", "_real"))
    print(f"\nindividual-retention check: {len(ck)}/{len(gen)} rows joined to real c "
          f"({gen.person_id.nunique()} persons x{gen.rep.max() + 1})")

    out = os.path.join(SYNTH_OUT_DIR, f"synth_trips_{a.mode}_x{a.k}.csv")
    os.makedirs(SYNTH_OUT_DIR, exist_ok=True)
    gen.to_csv(out, index=False)
    print(f"wrote {out} ({len(gen)} rows)")
    print("pipeline untouched: estimation still uses fully real HHTS via "
          "pipeline_logger.load_data() -> process_survey(save=False); "
          "to experiment with synthetic, pass this file explicitly.")


if __name__ == "__main__":
    main()
