#!/usr/bin/env python3
"""69_sensitivity.py - reviewer-requested sensitivity grid for the game-day
deployment MIP.

Runs scripts/60_game_day_mip.py once per parameter setting into
results/sens/<tag>/, then evaluates each output with 69_transfer_eval.py so
both predicted and realised (observed) coverage are reported. Idempotent:
existing outputs are reused, so a walltime kill can be resumed by resubmitting
the PBS job.

Grid: officer reach (obs_m 100/300/500 m), minimum separation (300/800 m),
max relocation distance (4/8 km), relocations per officer (0/1/2), and the
marginal-coverage floor (relative 5/10/20%).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
RESULTS = ROOT / "results"
SENS = RESULTS / "sens"

BASE = {
    "--min-marg": "0.5",
    "--max-segments": "250",
    "--squads": "all",
    "--travel-speed-kmh": "43.2",
    "--detour-factor": "1.3",
    "--setup-min": "5",
    "--max-travel-m": "8000",
    "--obs-m": "100",
    "--max-relocations": "1",
    "--min-move-gap-bins": "1",
    "--min-separation-m": "300",
    "--mobility-ab": "off",
}

CONFIGS = [
    ("base", {}),
    ("obs300", {"--obs-m": "300"}),
    ("obs500", {"--obs-m": "500"}),
    ("sep800", {"--min-separation-m": "800"}),
    ("travel4000", {"--max-travel-m": "4000"}),
    ("reloc0", {"--max-relocations": "0"}),
    ("reloc2", {"--max-relocations": "2"}),
    ("frac005", {"--min-marg-frac": "0.05"}),
    ("frac020", {"--min-marg-frac": "0.20"}),
]

FIELDS = ["tag", "args", "mean_coverage", "mean_coverage_network",
          "mean_squad", "squad_lo", "squad_hi", "mean_relocations",
          "distinct_site_sets", "mean_n_sites", "realised_obsfrac",
          "realised_raw", "oracle_top5", "random5_mean", "transfer_mean",
          "candidate_obs_share", "elapsed_s"]


def run_cmd(cmd: list[str], cwd: Path, log: Path) -> bool:
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(cwd), stdout=fh,
                             stderr=subprocess.STDOUT)
    print(f"  rc={rc} elapsed={time.time() - t0:.0f}s (log {log})", flush=True)
    return rc == 0


def eval_row(tag: str, mip_dir: Path, elapsed: float, args_desc: str) -> dict:
    ev = mip_dir / "transfer_eval.json"
    out = json.load(open(ev))
    pool = out["pooled"]
    sm = json.load(open(mip_dir / "game_day_mip_summary.json"))["summary"]
    row = {
        "tag": tag,
        "args": args_desc,
        "mean_coverage": sm.get("mean_coverage"),
        "mean_coverage_network": sm.get("mean_coverage_network"),
        "mean_squad": sm.get("mean_squad"),
        "squad_lo": min(sm.get("squad_range") or [0]),
        "squad_hi": max(sm.get("squad_range") or [0]),
        "mean_relocations": sm.get("mean_relocations"),
        "distinct_site_sets": sm.get("distinct_site_sets"),
        "mean_n_sites": pool.get("n_sites"),
        "realised_obsfrac": pool.get("realised_obsfrac"),
        "realised_raw": pool.get("realised_raw"),
        "oracle_top5": pool.get("oracle_top5"),
        "random5_mean": pool.get("random5_mean"),
        "transfer_mean": pool.get("transfer_mean"),
        "candidate_obs_share": pool.get("candidate_obs_share"),
        "elapsed_s": round(elapsed, 1),
    }
    return row


def write_rows(rows: list[dict]) -> None:
    SENS.mkdir(parents=True, exist_ok=True)
    with open(SENS / "sens_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-deployed", action="store_true")
    args = ap.parse_args()

    SENS.mkdir(parents=True, exist_ok=True)
    log = SENS / "sens.log"
    rows: list[dict] = []

    if args.dry_run:
        for tag, over in CONFIGS:
            print(tag, over)
        return

    if not args.skip_deployed:
        print("\n=== [deployed] evaluating shipped MIP on observed counts ===",
              flush=True)
        ok = run_cmd([sys.executable, str(SCRIPT_DIR / "69_transfer_eval.py"),
                      "--mip-dir", str(RESULTS),
                      "--out", str(RESULTS / "transfer_eval.json")],
                     SCRIPT_DIR, log)
        if ok:
            rows.append(eval_row("deployed", RESULTS, 0.0, "shipped 27008291"))
            write_rows(rows)

    for tag, over in CONFIGS:
        if args.only and tag not in args.only.split(","):
            continue
        out = SENS / tag
        fig = SENS / "figs" / tag
        out.mkdir(parents=True, exist_ok=True)
        fig.mkdir(parents=True, exist_ok=True)
        cfg = dict(BASE)
        cfg.update(over)
        argv: list[str] = []
        for k, v in cfg.items():
            argv.extend([k, v])
        path_args = ["--pred", str(RESULTS / "game_day_pred_15min.parquet"),
                     "--out-dir", str(out), "--fig-dir", str(fig)]
        t0 = time.time()
        if (out / "game_day_mip_summary.json").exists():
            print(f"\n=== [{tag}] existing outputs, skipping solve ===",
                  flush=True)
            ok = True
        else:
            print(f"\n=== [{tag}] solving ===", flush=True)
            ok = run_cmd([sys.executable, str(SCRIPT_DIR
                                           / "60_game_day_mip.py"),
                          *path_args, *argv], SCRIPT_DIR, log)
        if not ok:
            print(f"  {tag}: solve FAILED, continuing", flush=True)
            continue
        if not (out / "game_day_mip_summary.json").exists():
            print(f"  {tag}: no summary written, continuing", flush=True)
            continue
        if not (out / "transfer_eval.json").exists():
            run_cmd([sys.executable, str(SCRIPT_DIR / "69_transfer_eval.py"),
                     "--mip-dir", str(out),
                     "--out", str(out / "transfer_eval.json")],
                    SCRIPT_DIR, log)
        try:
            row = eval_row(tag, out, time.time() - t0, json.dumps(over))
            rows.append(row)
            write_rows(rows)
            print(f"  [{tag}] pred={row['mean_coverage']} "
                  f"network={row['mean_coverage_network']} "
                  f"P={row['mean_squad']} reloc={row['mean_relocations']} "
                  f"realised={row['realised_obsfrac']} "
                  f"oracle={row['oracle_top5']} "
                  f"random={row['random5_mean']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: eval aggregation failed ({exc})", flush=True)

    if rows:
        write_rows(rows)
    print("\n=== sensitivity complete ===", flush=True)
    for r in rows:
        print(f"   {r['tag']:12s} cov={r['mean_coverage']} "
              f"net={r['mean_coverage_network']} P={r['mean_squad']} "
              f"moves={r['mean_relocations']} "
              f"realised={r['realised_obsfrac']}", flush=True)


if __name__ == "__main__":
    main()
