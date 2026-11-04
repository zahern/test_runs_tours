#!/usr/bin/env python
"""
sd_hybrid_371.py — system-dynamics (SD) layer for the SEQ 371-zone ABM
=======================================================================
Paper anchor: Nguyen, Howick & Megiddo (2024), "A framework for
conceptualising hybrid system dynamics and agent-based simulation models",
EJOR 315(3):1153-1166.  Their core requirement is an explicit aggregation /
disaggregation interface: *what* crosses the ABM<->SD boundary, in *which
direction*, at *what resolution*, and *how* the feedback closes.

THE HYBRID LOOP
---------------
The ABM is the fast, agent-level engine (tours -> trips -> mode/destination
choice -> VMT/VHT/fuel cost).  This module is the slow aggregate layer that
rides on top of it at a 5-year step.  There are exactly two stocks and two
disaggregated signals:

    scale   variable                      lives in        role
    ------  ----------------------------  --------------  --------------------
    agent   annual VMT (veh-km / yr)      ABM             aggregation (sum)
    stock   congestion lag  raw_s(t)      SD              state, 5-yr step
    stock   EV share        ev(t)         SD              Bass cumulative
    signal  speed factor    sf(t)         SD -> ABM       scales VHT
    signal  cost multiplier cm(t)         SD -> ABM       scales fuel cost

  ┌──────────────────────── ABM (fast, agent level) ─────────────────────────┐
  │  synthetic pop → tours → trips → mode/destination choice → VMT·VHT·cost  │
  └──────┬──────────────────┬──────────────────────────────────────▲─────────┘
         │ I1 annual VMT    │ I3 policy q (TELE/MBUF)              │ I2 speed factor
         ▼                  ▼                                      │ I4 cost multiplier
  ┌──────────────────────── SD layer (slow, 5-year step) ───────────┴─────────┐
  │  v/c = VMT/cap ──BPR──▶ raw(v/c) ──delay──▶ raw_s ──▶ sf = raw_s/raw(0.7) │
  │  scenario q ──Bass──▶ F(t) ──▶ ev(t) ──▶ cm = 1−ev(1−r)                   │
  └───────────────────────────────────────────────────────────────────────────┘

  I1  aggregation   : Σ over agents × 365, annual VMT per scenario-year.
  I2  disaggregation: same factor to every auto trip (hours / sf).
  I3  ABM -> SD     : TELE / MBUF shift the Bass imitation parameter q.
  I4  disaggregation: same factor to every auto trip (fuel cost × cm).

  PARTIAL LOOP: VMT and trip counts are intentionally NOT adjusted by the
  speed factor (no induced demand), so congestion feeds back into VHT and
  operating cost only.  See the interface table emitted to
  ``sd_interfaces.csv`` for the machine-readable version of this contract.

EQUATIONS (all variables per scenario-year)
-------------------------------------------
1. BPR inverse — speed as a fraction of free-flow speed (i.e. t0 / t):

       raw(v/c)  = 1 / (1 + ALPHA * (v/c) ** BETA)
       v/c       = VMT / capacity,   capacity = VMT_2025 / 0.7
       raw_base  = raw(0.7)                         (2025 anchor, v/c = 0.7)

   The BPR *time* multiplier is 1 + ALPHA*(v/c)^BETA; inverting it gives the
   speed multiplier.  ALPHA = 0.15, BETA = 4.0 are the classic Bureau of
   Public Roads values; override via env to test stronger congestion
   responses (e.g. SD_BPR_ALPHA=1.0, SD_BPR_BETA=2.0).

2. Congestion delay — first-order lag, representing a 5-year response delay:

       raw_s(t)  = DELAY * raw(t) + (1 - DELAY) * raw_s(t-1)
       sf(t)     = raw_s(t) / raw_base
       sf(2025)  = 1.0                              (base-year anchor)

   NOTE: this is exponential smoothing (a lag), *not* a method of successive
   averages (MSA).  The parameter was historically named ``SD_MSA_LAMBDA``;
   that env var is still honoured as an alias of ``SD_DELAY_LAMBDA``.

3. Bass diffusion of the EV fleet:

       F(t)      = (1 - e^{-(p+q)t}) / (1 + (q/p) e^{-(p+q)t}),  t = year - 2025
       ev(t)     = EV_SHARE_2025 + (1 - EV_SHARE_2025) * F(t),  ev <= 0.95
       q_scenario= q * (1.10 TELE, 0.95 MBUF, 1.00 BASE)

   F(t) is the standard Bass cumulative-adoption curve.  The observed 2025
   share (2%) anchors the start; the remaining 98% of the fleet adopts
   according to F.  p and q are SEQ placeholders (override via SD_BASS_P/Q).

4. Fleet fuel-cost multiplier (fuel/energy only; tolls are untouched):

       cm(t)     = 1 - ev(t) * (1 - EV_COST_RATIO)
       cm_rel(t) = cm(t) / cm(2025)

   EV_COST_RATIO = 0.3 is the EV energy cost per km relative to petrol.

Pure pandas/numpy.  Deterministic.  Offline.  The flow-chart export needs
matplotlib and is optional.

Run:  python sd_hybrid_371.py [--travel saved_data/luti/scenario_travel.csv]
Env:  SD_BASS_P, SD_BASS_Q, SD_BPR_ALPHA, SD_BPR_BETA,
      SD_DELAY_LAMBDA (alias SD_MSA_LAMBDA), EV_COST_RATIO, LUTI_OUTDIR
Outputs: saved_data/luti/sd_stocks.csv, saved_data/luti/sd_interfaces.csv,
         saved_data/luti/sd_flow_diagram.png, sd_flow_diagram.mmd
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("sdhybrid")

_HERE = os.path.dirname(os.path.abspath(__file__))
LUTI = os.environ.get("LUTI_OUTDIR", os.path.join(_HERE, "saved_data", "luti"))

# ── parameters (documented in the module docstring) ──────────────────────────
P = float(os.environ.get("SD_BASS_P", 0.003))          # Bass innovation
Q = float(os.environ.get("SD_BASS_Q", 0.38))           # Bass imitation
ALPHA = float(os.environ.get("SD_BPR_ALPHA", 0.15))    # BPR alpha (BPR 1964)
BETA = float(os.environ.get("SD_BPR_BETA", 4.0))       # BPR beta (BPR 1964)
# First-order delay weight.  Historical name was SD_MSA_LAMBDA; keep it as a
# backward-compatible alias.
DELAY = float(os.environ.get("SD_DELAY_LAMBDA",
                             os.environ.get("SD_MSA_LAMBDA", 0.5)))
EV_COST_RATIO = float(os.environ.get("EV_COST_RATIO", 0.3))  # EV vs petrol /km
EV_SHARE_2025 = 0.02
VC_BASE_2025 = 0.7        # documented 2025 v/c anchor
EV_MAX_SHARE = 0.95
SCEN_Q_FACTOR = {"TELE": 1.10, "MBUF": 0.95}


def interface_table():
    """Nguyen et al. hybrid-interface spec: what crosses ABM<->SD, how.

    One row per interface (I1-I4 = closed loop, I5 = reserved extension).
    ``feedback_closed`` is deliberately honest: the congestion loop is
    *partial* because VMT/trips are held exogenous (no induced demand).
    """
    rows = [
        {"interface": "I1 congestion-in", "from_model": "ABM (tour pipeline)",
         "to_model": "SD congestion delay", "variable": "annual VMT (veh-km/yr)",
         "equation": "v/c = VMT / (VMT_2025 / 0.7)",
         "aggregation": "sum over agents x 365", "resolution": "5-year step",
         "direction": "ABM->SD", "feedback_closed": "partial (VMT exogenous)"},
        {"interface": "I2 speed-out", "from_model": "SD congestion delay",
         "to_model": "ABM (travel totals)", "variable": "relative speed factor",
         "equation": "sf(t) = raw_s(t) / raw(0.7)",
         "aggregation": "BPR inverse + first-order lag",
         "resolution": "5-year step", "direction": "SD->ABM",
         "feedback_closed": "yes (scales VHT)"},
        {"interface": "I3 policy-in", "from_model": "ABM scenario (TELE/MBUF)",
         "to_model": "SD EV stock", "variable": "Bass q multiplier",
         "equation": "q_scen = q x {TELE: 1.10, MBUF: 0.95}",
         "aggregation": "scenario constant", "resolution": "scenario",
         "direction": "ABM->SD", "feedback_closed": "yes (adoption responds)"},
        {"interface": "I4 cost-out", "from_model": "SD EV stock",
         "to_model": "ABM (travel totals)", "variable": "fuel-cost multiplier",
         "equation": "cm_rel = (1-ev(1-r)) / (1-ev_2025(1-r))",
         "aggregation": "Bass share x cost ratio", "resolution": "5-year step",
         "direction": "SD->ABM", "feedback_closed": "yes (scales fuel cost)"},
        {"interface": "I5 reserved", "from_model": "ABM (synth pop)",
         "to_model": "SD (extension)", "variable": "car ownership / PT crowding",
         "equation": "-", "aggregation": "-", "resolution": "-",
         "direction": "-", "feedback_closed": "no (documented extension)"},
    ]
    return pd.DataFrame(rows)


def _bass_cumulative(t, p=P, q=Q):
    """Bass cumulative adoption F(t) = (1-e^{-(p+q)t})/(1+(q/p)e^{-(p+q)t})."""
    e = np.exp(-(p + q) * t)
    return (1.0 - e) / (1.0 + (q / max(p, 1e-9)) * e)


# Backward-compatible alias for the original private helper name.
_basscup = _bass_cumulative


def sd_factors(scen, year, vmt, cap, sf_raw_prev=None):
    """Evaluate the SD equations for one scenario-year.

    Parameters
    ----------
    scen : str            scenario label ('BASE', 'TELE', 'MBUF')
    year : int            horizon year (2025, 2030, ...)
    vmt : float           annual VMT for the scenario-year (veh-km / yr)
    cap : float           network capacity (VMT at v/c = 1; 2025 VMT / 0.7)
    sf_raw_prev : float   raw speed factor from the previous step (None at the
                          start of a trajectory)

    Returns
    -------
    (speed_factor, cost_mult, ev_share, sf_raw_new) — all factors relative to
    the 2025 base, so observed base-year times/costs are never rescaled.
    """
    t = float(year) - 2025.0
    q = Q * SCEN_Q_FACTOR.get(scen, 1.0)

    # (1) BPR inverse: speed as a fraction of free-flow speed.
    vc = float(vmt) / max(cap, 1e-9)
    raw = 1.0 / (1.0 + ALPHA * vc ** BETA)
    base_raw = 1.0 / (1.0 + ALPHA * VC_BASE_2025 ** BETA)

    # (2) first-order congestion delay, then express relative to the 2025 anchor.
    if sf_raw_prev is None:
        sf_raw_new, sf_rel = raw, 1.0
    else:
        sf_raw_new = DELAY * raw + (1.0 - DELAY) * sf_raw_prev
        sf_rel = sf_raw_new / base_raw

    # (3) Bass diffusion of the EV fleet (anchored at the 2025 observed share).
    ev = EV_SHARE_2025 + (1.0 - EV_SHARE_2025) * _bass_cumulative(t, q=q)
    ev = float(np.clip(ev, 0.0, EV_MAX_SHARE))

    # (4) fleet fuel-cost multiplier relative to 2025.
    cm = 1.0 - ev * (1.0 - EV_COST_RATIO)
    cm0 = 1.0 - EV_SHARE_2025 * (1.0 - EV_COST_RATIO)
    return round(sf_rel, 4), round(cm / cm0, 4), round(ev, 4), sf_raw_new


def run_sd(travel):
    """Advance the SD stocks along the scenario_travel trajectory.

    travel: DataFrame with scenario/year/vmt (annual).  Returns a stocks
    DataFrame (scenario/year/vc_ratio/speed_factor/ev_share/cost_mult).
    """
    out = []
    # Network capacity proxy: 2025 VMT at the documented v/c = 0.7 anchor.
    base25 = travel[(travel["scenario"] == "BASE") & (travel["year"] == 2025)]
    cap = float(base25["vmt"].iloc[0] / VC_BASE_2025) if len(base25) else 1e12
    for scen, g in travel.groupby("scenario"):
        g = g.sort_values("year").reset_index(drop=True)
        sf_raw_prev = None
        for _, r in g.iterrows():
            sf, cm, ev, sf_raw_prev = sd_factors(
                scen, int(r["year"]), float(r["vmt"]), cap, sf_raw_prev)
            out.append({"scenario": scen, "year": int(r["year"]),
                        "vc_ratio": round(float(r["vmt"]) / cap, 3),
                        "speed_factor": sf, "ev_share": ev, "cost_mult": cm})
    return pd.DataFrame(out)


def flow_diagram_mermaid():
    """Return the hybrid ABM<->SD loop as a Mermaid flowchart string."""
    return """flowchart LR
  subgraph ABM["ABM (fast, agent level)"]
    A1["synthetic population"] --> A2["tours / trips / modes / destinations"]
    A2 --> A3["VMT - VHT - fuel cost"]
  end
  subgraph SD["SD layer (slow, 5-year step)"]
    S1["v/c = VMT / capacity"] --> S2["BPR inverse raw(v/c)"]
    S2 --> S3["first-order delay raw_s(t)"]
    S4["Bass F(t)"] --> S5["EV share ev(t)"]
  end
  A3 -- "I1 annual VMT" --> S1
  S3 -- "I2 speed factor" --> A3
  A2 -- "I3 policy q (TELE/MBUF)" --> S4
  S5 -- "I4 cost multiplier" --> A3
  S3 -. "partial loop: VMT not fed back" .-> A3
"""


def draw_flow_diagram(path=None, title=True):
    """Render the hybrid loop as a PNG flow chart (matplotlib, optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("draw_flow_diagram skipped (matplotlib unavailable: %s)", exc)
        return None

    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h / 2 - 0.16, body, ha="center", va="center",
                fontsize=8.6, color="#222", linespacing=1.45)

    def arrow(x1, y1, x2, y2, label="", color="#444", ls="-"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>,head_width=4,head_length=7",
            lw=1.6, color=color, linestyle=ls, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label,
                    ha="center", va="bottom", fontsize=8.2, color=color,
                    fontweight="bold")

    box(0.30, 2.30, 3.30, 4.30, "ABM — agent level (fast)",
        "synthetic population\n→ tours → trips\n→ mode + destination choice\n"
        "→ VMT · VHT · fuel cost", "#e8f0fe", "#2f5d9e")
    box(5.95, 4.70, 3.60, 1.90, "SD — congestion stock",
        f"v/c = VMT / capacity\nraw = 1/(1+{ALPHA:g}·(v/c)^{BETA:g})\n"
        f"raw_s = {DELAY:g}·raw + (1−{DELAY:g})·raw_s(t−1)", "#fff4e5", "#b26a12")
    box(5.95, 2.30, 3.60, 1.90, "SD — EV fleet stock",
        "Bass F(t), q ← scenario\nEV share ev(t)\n"
        "cm = 1 − ev·(1−r)", "#e9f7ef", "#257a4a")

    arrow(3.60, 5.95, 5.95, 5.95, "I1  annual VMT", "#2f5d9e")
    arrow(5.95, 5.05, 3.60, 5.05, "I2  speed factor", "#b26a12")
    arrow(3.60, 3.60, 5.95, 3.60, "I3  policy q", "#257a4a")
    arrow(5.95, 2.75, 3.60, 2.75, "I4  cost multiplier", "#b26a12")

    ax.text(0.30, 1.90,
            "Partial loop (documented): congestion scales VHT and fuel cost only —\n"
            "VMT and trip counts are held exogenous (no induced-demand feedback).",
            fontsize=8.8, color="#a33", ha="left", va="top")
    ax.text(0.30, 1.15,
            "I1–I4 cross the ABM↔SD boundary once per 5-year step.  "
            "Aggregation: Σ over agents × 365.\n"
            "Disaggregation: the same factor applies to every auto trip "
            "(hours ÷ sf; fuel cost × cm).",
            fontsize=8.8, color="#444", ha="left", va="top")
    if title:
        ax.set_title("Hybrid system-dynamics ↔ agent-based model loop "
                     "(371-zone SEQ)", fontsize=12.5, pad=12)
    fig.tight_layout()
    if path is None:
        path = os.path.join(LUTI, "sd_flow_diagram.png")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--travel", default=os.path.join(LUTI, "scenario_travel.csv"))
    args = ap.parse_args()
    if not os.path.exists(args.travel):
        raise SystemExit(f"FATAL: {args.travel} missing — run run_luti_scenarios_371.py first")
    stocks = run_sd(pd.read_csv(args.travel))
    os.makedirs(LUTI, exist_ok=True)
    stocks.to_csv(os.path.join(LUTI, "sd_stocks.csv"), index=False)
    interface_table().to_csv(os.path.join(LUTI, "sd_interfaces.csv"), index=False)
    try:
        draw_flow_diagram(os.path.join(LUTI, "sd_flow_diagram.png"))
        with open(os.path.join(LUTI, "sd_flow_diagram.mmd"), "w",
                  encoding="utf-8") as f:
            f.write(flow_diagram_mermaid())
    except Exception as _fd_e:
        log.warning("flow diagram export failed (non-fatal): %s", _fd_e)
    for scen in ("TELE", "MBUF"):
        s45 = stocks[(stocks["scenario"] == scen) & (stocks["year"] == 2045)]
        b45 = stocks[(stocks["scenario"] == "BASE") & (stocks["year"] == 2045)]
        if len(s45) and len(b45):
            log.info("%s 2045 vs BASE: speed_factor %.3f vs %.3f | ev_share %.1f%% | cost_mult %.3f",
                     scen, s45["speed_factor"].iloc[0], b45["speed_factor"].iloc[0],
                     100 * s45["ev_share"].iloc[0], s45["cost_mult"].iloc[0])
    print("Wrote sd_stocks.csv + sd_interfaces.csv (+ flow diagram when matplotlib is present)")


if __name__ == "__main__":
    main()
