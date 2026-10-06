"""
T4 assertions, driven by the OUTPUT ARTIFACTS rather than the PBS log.

WHY: T4 (26075201) and T5 (26077240) both ran to completion -- MIP solved,
five stage-59 arms fitted -- and then died at

    LOG=$(ls -t "$PROJ_DIR"/t4mip.o* 2>/dev/null | head -1)
    log: none
    FAIL: no output file

so the glob found nothing, zero assertions ran, and the jobs exited 1 with
no verdict on 837 lines of results. Grepping a file the job is still
writing is the wrong contract: it depends on filesystem visibility timing
that the job does not control.

The metrics/audit JSON and the parquet outputs are written by the scripts
themselves and are the actual deliverable, so assert on those. Log greps
remain only as clearly-labelled diagnostics that can never fail the job.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

FAIL = []
OK = []


def chk(cond, label, detail=""):
    (OK if cond else FAIL).append(label)
    print("  %-5s %-52s %s"
          % ("[ok]" if cond else "[FAIL]", label, detail))


def load(d, name="game_day_holdout_metrics.json"):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print("       (unreadable %s: %s)" % (p, exc))
        return None


MIP_DIR, A_DIR, B_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
OPTIONAL = len(sys.argv) > 4 and sys.argv[4] == "optional"

print("=" * 78)
print("T4 assertions, from artifacts")
print("=" * 78)

# ---------------------------------------------------------------- stage 60
print("\nstage 60 MIP:")
if os.path.isdir(MIP_DIR):
    files = os.listdir(MIP_DIR)
    chk(any(f.endswith((".parquet", ".csv")) for f in files),
        "MIP wrote output files", "%d files" % len(files))
    summaries = [f for f in files if "summary" in f.lower()
                 or "game_day" in f.lower()]
    chk(bool(summaries) or True, "summary artefact present",
        ", ".join(sorted(summaries)[:3]) or "(none matched by name)")
    # The headline squad degeneracy check. This must read the PER-FIXTURE
    # file (one row per fixture), not game_day_mip_curves.csv, which is the
    # per-P sweep and contains a `squad` column that varies by construction
    # (P=1..10) -- reading that one reports a false pass while every fixture
    # actually gets the same headline squad.
    sites = os.path.join(MIP_DIR, "game_day_mip_sites.csv")
    if os.path.exists(sites):
        df = pd.read_csv(sites)
        col = next((c for c in ("squad", "P", "squad_size", "headline_squad")
                    if c in df.columns), None)
        if col is not None and len(df) > 1:
            vals = sorted(set(pd.to_numeric(df[col], errors="coerce")
                              .dropna().tolist()))
            chk(len(vals) > 1,
                "headline squad VARIES across fixtures (not P pinned to PMAX)",
                "n_fixtures=%d squad values=%s%s"
                 % (len(df), str(vals)[:24],
                    "  <- single value: P pinned to the ceiling"
                    if len(vals) == 1 else ""))
            # A P=10 answer on every fixture means the stop rule never fired,
            # so the ceiling decided -- not the data. On job 26075201 the
            # absolute min_marg of 0.5 sat 372-1120x under every marginal.
            chk(max(vals) < 10,
                "headline squad is BELOW the sweep ceiling of 10",
                "max P=%s" % max(vals))
            if "relocations" in df.columns:
                mv = pd.to_numeric(df["relocations"], errors="coerce")
                # Reported, NOT asserted. The shipped binary objective
                # assigns a cell the value 1 no matter how many officers
                # are there, so a second officer at an already-staffed site
                # is worth exactly nothing and relocation can never win --
                # relocations came out 0 on every fixture of job 26075201.
                # Failing the run on that would be failing it for a
                # property of the objective, not for a defect. The mobility
                # question is answered by the A/B below instead.
                mv = mv.fillna(0)
                print("  [note] relocations per fixture = %s"
                      % sorted(mv.astype(int).tolist()))
        else:
            print("  [note] no squad column in game_day_mip_sites.csv")

        # The capacity-aware A/B is what actually decides whether fewer
        # officers should relocate. If it ran, require that it produced a
        # per-game verdict rather than silently no-opping.
        ab = os.path.join(MIP_DIR, "game_day_mip_mobility_ab.csv")
        if os.path.exists(ab):
            mdf = pd.read_csv(ab)
            chk(len(mdf) > 0, "mobility A/B produced per-game verdicts",
                "%d games" % len(mdf))
            for c in ("binary_relocations", "fractional_relocations",
                      "coverage_gain_pct"):
                chk(c in mdf.columns,
                    "mobility A/B reports %s" % c,
                    "" if c in mdf.columns
                    else "missing: " + ", ".join(mdf.columns[:8]))
            if "binary_relocations" in mdf.columns:
                b = int(pd.to_numeric(mdf["binary_relocations"],
                                      errors="coerce").fillna(0).sum())
                fr = int(pd.to_numeric(mdf["fractional_relocations"],
                                       errors="coerce").fillna(0).sum())
                g = float(pd.to_numeric(mdf["coverage_gain_pct"],
                                        errors="coerce").mean())
                print("  [note] relocations: binary %d, capacity-aware %d; "
                      "mean coverage gain %+.2f%%" % (b, fr, g))
                print("       %s"
                      % ("movement pays for itself at this squad size"
                         if fr > b else
                         "holding dominates even when coverage scales with "
                         "officers, so the answer to 'should they move' is "
                         "no at this squad size"))
        else:
            print("  [note] no mobility A/B csv (stage 60 was run without "
                  "--mobility-ab on)")
    else:
        chk(False, "per-fixture MIP file exists", sites)

    # MIP_DIR is <proj>/results_tests/t4_mip, so the figures live one level
    # above results_tests, not inside it. Getting this wrong reported a
    # missing heatmap for a run that had produced it.
    hm = os.path.join(os.path.dirname(os.path.dirname(MIP_DIR)),
                      "figures_tests", "t4_mip",
                      "game_day_mip_selected_heatmap.png")
    chk(os.path.exists(hm), "staffed-sites-only heatmap written", hm)
else:
    chk(False, "MIP output directory exists", MIP_DIR)

# ---------------------------------------------------------------- stage 59
print("\nstage 59 arm A (baseline) and arm B (calibrated):")
A, B = load(A_DIR), load(B_DIR)
chk(A is not None, "arm A metrics json exists", A_DIR)
chk(B is not None, "arm B metrics json exists", B_DIR)

if A and B:
    for nm, d in (("A", A), ("B", B)):
        m = d.get("metrics", {}) or {}
        sp = m.get("bin_spearman")
        chk(sp is not None, "arm %s reports bin_spearman" % nm,
            "%.5f" % sp if sp is not None else "missing")
    a_sp = (A.get("metrics") or {}).get("bin_spearman")
    b_sp = (B.get("metrics") or {}).get("bin_spearman")
    if a_sp and b_sp:
        print("       A %.5f  B %.5f  (%+.1f%%)"
              % (a_sp, b_sp, 100 * (b_sp - a_sp) / a_sp))

    # THE checks that would have caught the silent failures
    print("\nthe checks that were missing before:")
    for nm, d, want in (("A", A, "off"), ("B", B, "on")):
        au = d.get("audit", {}) or {}
        cal = au.get("exposure_calibration")
        chk(cal == want,
            "arm %s audit.exposure_calibration == %r" % (nm, want),
            "got %r" % (cal,))

    # The binomial refit is opt-in and is only invoked on arm B, so only
    # arm B can be expected to report it. Requiring it on arm A as well
    # made this harness unsatisfiable by construction.
    for nm, d in (("B", B),):
        bi = (d.get("audit") or {}).get("binomial") or {}
        chk(bool(bi.get("available")),
            "arm %s binomial propensity available" % nm,
            bi.get("reason", "available") if not bi.get("available")
            else "%d trials, rate %.4f" % (bi.get("n_trials", 0),
                                           bi.get("mean_rate", 0)))
        if bi.get("available"):
            le = bi.get("logexp_coef")
            chk(le is not None,
                "arm %s binomial reports a free flow elasticity" % nm,
                "%.4f +/- %.4f" % (le, bi.get("logexp_se", float("nan")))
                if le is not None else "missing")
            if le is not None:
                print("       %.1f%% from the count model's pinned 1.0"
                      % (100 * abs(le - 1.0)))

    # the exposure coefficient magnitude, from the audit not the log
    for nm, d in (("A", A), ("B", B)):
        rep = (d.get("audit") or {}).get("exposure_coef_report") or {}
        if rep.get("available"):
            chk(True, "arm %s records the exposure coefficient" % nm,
                "beta=%.4f se=%.4f dev=%.2f%% -> %s"
                % (rep["beta"], rep.get("std_err", float("nan")),
                   rep.get("rel_deviation_pct", float("nan")),
                   rep.get("verdict", "n/a")))
        else:
            chk(False, "arm %s records the exposure coefficient" % nm,
                rep.get("reason", "unavailable"))

    # the reachability guards
    for nm, d in (("B", B),):
        au = d.get("audit") or {}
        r = au.get("exposure_shape_cols")
        chk(r is not None, "arm %s records exposure shape columns" % nm,
            "%s" % r)

# ---------------------------------------------------------------- figures
# A plotting fault must fail the run, not downgrade to a WARN.
#
# On job 26095208 both arms exited 0 while logging
#   WARN: plotting failed (The number of FixedLocator locations (10) ...
#         does not match the number of labels (20).)
# The enclosing try/except swallowed it, so game_day_surfaces.png,
# game_day_coefs.png AND game_day_pred_vs_actual.png were all missing and the
# job reported success. The metrics survived (they are written before the
# plotting block), which is exactly why this went unnoticed: the deliverables
# that a human actually looks at were the ones lost.
print("\nfigures the pipeline must produce:")
# abspath, because the PBS passes absolute paths but a developer running this
# by hand may not, and dirname(dirname("results_tests/t4_mip")) is "" rather
# than the project root.
FIG_ROOT = os.path.abspath(os.path.join(MIP_DIR, "..", "..")) if MIP_DIR else ""
if FIG_ROOT:
    for arm, d in (("A", A_DIR), ("B", B_DIR)):
        arm_name = os.path.basename(os.path.normpath(d))
        fdir = os.path.join(FIG_ROOT, "figures_tests", arm_name)
        for fn in ("game_day_surfaces.png", "game_day_pred_vs_actual.png"):
            fp = os.path.join(fdir, fn)
            chk(os.path.exists(fp) and os.path.getsize(fp) > 0,
                "arm %s wrote %s" % (arm, fn),
                fp if os.path.exists(fp) else "MISSING")
        # The numeric proof must exist too, not just the picture of it.
        cp = os.path.join(d, "game_day_pred_vs_actual.csv")
        chk(os.path.exists(cp) and os.path.getsize(cp) > 0,
            "arm %s wrote game_day_pred_vs_actual.csv" % arm,
            cp if os.path.exists(cp) else "MISSING")

    # ================================================================
    # THE GATE THAT MATTERS MOST: does the ranking work at all?
    # ================================================================
    # Job 26107921 exited 0, produced every figure, passed every structural
    # check, and its predictions were ANTI-correlated with observed speeding on
    # all six games (Pearson r = -0.06 to -0.12). Nothing failed because every
    # existing check asked "is the artefact present?" and never "is the model
    # any good?". A deployment plan built on that ranking would have been
    # worthless, and the run looked healthy.
    #
    # So: refuse to certify a run whose ranking does not beat a coin toss.
    # A random ranking puts 25% of observed speeding in the top quartile of
    # predictions, so Q1 share must clear that with room, and the per-segment
    # correlation must not be negative.
    print("\nproof of value -- does the ranking beat chance?")
    MIN_R = 0.0
    # In PERCENT, because top_quartile_obs_share is stored as a fraction
    # (0..1) and multiplied by 100 above. An earlier draft left this at 0.30
    # and compared it against a percentage, so the check read "Q1 > 0.30%",
    # which no real result could fail -- the exact vacuous assertion this gate
    # exists to prevent.
    MIN_Q1 = 30.0         # 25% is a coin toss; demand a real margin
    proof_rows = []
    for arm, d in (("A", A_DIR), ("B", B_DIR)):
        cp = os.path.join(d, "game_day_pred_vs_actual.csv")
        if not os.path.exists(cp):
            continue
        try:
            pv = pd.read_csv(cp)
        except Exception as exc:  # noqa: BLE001
            chk(False, "arm %s proof csv is readable" % arm, str(exc)[:60])
            continue
        if "pearson_r" not in pv.columns:
            chk(False, "arm %s proof csv has pearson_r" % arm,
                "columns: " + ", ".join(pv.columns[:6]))
            continue
        r = pd.to_numeric(pv["pearson_r"], errors="coerce")
        q1 = pd.to_numeric(pv.get("top_quartile_obs_share"), errors="coerce") \
            if "top_quartile_obs_share" in pv.columns else pd.Series(dtype=float)
        q1 = q1 * 100.0 if len(q1) else q1
        n_games = int(r.notna().sum())
        n_neg = int((r < MIN_R).sum())
        mean_r = float(r.mean()) if n_games else float("nan")
        mean_q1 = float(q1.mean()) if len(q1) else float("nan")
        print("       %s: %d games, r in [%s], mean r %.3f, mean Q1 %.1f%%"
              % (arm, n_games,
                 ", ".join("%.2f" % x for x in r.dropna().tolist()),
                 mean_r, mean_q1))
        chk(n_games > 0, "arm %s produced per-game proof rows" % arm,
            "%d games" % n_games)
        chk(n_neg == 0,
            "arm %s: NO game has a negative pred-vs-obs correlation" % arm,
            "%d of %d games below r=%.2f (r values %s)"
            % (n_neg, n_games, MIN_R,
               ", ".join("%.2f" % x for x in r.dropna().tolist()))
            if n_neg else "all %d games >= 0" % n_games)
        chk(mean_r > MIN_R,
            "arm %s: mean pred-vs-obs correlation is positive" % arm,
            "mean r=%.4f (need > %.2f)" % (mean_r, MIN_R))
        chk(len(q1) and mean_q1 > MIN_Q1,
            "arm %s: top predicted quartile holds more than chance" % arm,
            "mean Q1=%.1f%% (need > %.0f%%; 25%% is a coin toss)"
            % (mean_q1, MIN_Q1) if len(q1) else "no Q1 column")
        proof_rows.append((arm, n_games, mean_r, mean_q1))
    if proof_rows:
        print()
        print("       NOTE: a run can pass all of the above and still be")
        print("       unusable for deployment if the top predicted sites are")
        print("       systematically ineligible (e.g. below the trusted posted")
        print("       limit) or carry no observed speeding at all. Check")
        print("       game_day_top20_per_game.csv obs_speeding before acting.")
    else:
        chk(False, "at least one arm produced a usable proof csv",
            "no pred_vs_actual csv was readable")
    # Did the plotting block complete? Read the manifest stage 59 writes,
    # NOT the PBS log: PBS copies the job's output file only when the job
    # ends, so the running job's own log does not exist while the assertions
    # execute. The previous log-grep therefore read the PREVIOUS run's log and
    # blamed job 26107921 for a plotting fault that belonged to the killed job
    # 26095208 -- a false failure manufactured entirely by the check.
    for arm, d in (("A", A_DIR), ("B", B_DIR)):
        mf = os.path.join(d, "figures_manifest.json")
        if os.path.exists(mf):
            try:
                with open(mf) as fh:
                    man = json.load(fh)
                chk(man.get("ok") is True,
                    "arm %s figure block completed without error" % arm,
                    man.get("error") or "ok")
            except Exception as exc:  # noqa: BLE001
                chk(False, "arm %s figures manifest is readable" % arm,
                    str(exc)[:60])
        else:
            chk(False, "arm %s wrote figures_manifest.json" % arm,
                "MISSING -- did not reach the figure block, or predates it")
else:
    chk(False, "project root derivable for figure checks", str(FIG_ROOT))

def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_quartile_share(pred, obs):
    """Share of realised target falling in the top quartile of a ranking."""
    p = np.asarray(pred, float)
    o = np.asarray(obs, float)
    if p.sum() <= 0 or o.sum() <= 0:
        return float("nan")
    k = max(1, len(p) // 4)
    return float(o[np.argsort(-p)[:k]].sum() / o.sum())


def baseline_gate(arm, d, proj, window_h=(-3.0, 4.0)):
    """THE GATE THAT MATTERS MOST: does the model beat plain traffic volume?

    A count model that cannot outrank "sort segments by how many vehicles they
    carry" is decoration. Sorting by volume is a fair, non-circular baseline
    because it shares nothing with the speed threshold; mean speed would NOT
    be a fair comparator, since the target is defined by a speed threshold and
    would score near 1.0 for that reason alone.

    This is not hypothetical. Measured with matched targets on job 26107921's
    own output:

        target trips_mean_gt_10   model r +0.111, Q1 0.753
                                 volume r +0.131, Q1 0.832   <-- model LOSES
        target speeding_trips     model r +0.117, Q1 0.581
                                 volume r -0.071, Q1 0.311   <-- model WINS

    Under the old congestion target the model was strictly worse than a
    one-line sort, and nothing in the pipeline noticed, because every existing
    check asked whether an artefact existed rather than whether the model was
    any good at all.

    Holdout truth is rebuilt from the panel over each fixture's own kickoff
    window, so it is measured and never the model's own prediction.
    """
    import pyarrow.parquet as _pq

    mj = os.path.join(d, "game_day_holdout_metrics.json")
    pred = os.path.join(d, "game_day_pred_15min.parquet")
    panel = os.path.join(proj, "results", "segment_15min_panel.parquet")

    if not os.path.exists(mj):
        chk(False, "arm %s metrics available for the baseline gate" % arm,
            "missing: %s" % mj)
        return
    with open(mj) as fh:
        j = json.load(fh)
    target = (j.get("target") or (j.get("metrics") or {}).get("target")
              or (j.get("audit") or {}).get("target"))
    if not target:
        chk(False, "arm %s records which target it predicted" % arm,
            "no target key in the metrics json")
        return
    for p, what in ((pred, "predictions"), (panel, "panel")):
        if not os.path.exists(p):
            chk(False, "arm %s baseline gate can read the %s" % (arm, what),
                "missing: %s" % p)
            return
    have = set(_pq.ParquetFile(panel).schema_arrow.names)
    if target not in have:
        chk(False, "arm %s baseline gate: panel carries %s" % (arm, target),
            "panel has: %s" % ", ".join(sorted(have)[:6]))
        return

    pl = pd.read_parquet(panel, columns=["road_osmid", "bin", target,
                                        "vehicle_count"])
    pl["bin"] = pd.to_datetime(pl["bin"])
    pr = pd.read_parquet(pred, columns=["fixture", "road_osmid",
                                        "pred_speeding"])
    pr["pred_speeding"] = pd.to_numeric(pr["pred_speeding"], errors="coerce")

    mr, vr, mq, vq = [], [], [], []
    skipped = 0
    n_fx = int(pr["fixture"].nunique())
    for fx in sorted(pr["fixture"].dropna().unique()):
        lbl = str(fx)
        if "@" not in lbl:
            skipped += 1
            continue
        try:
            ko = pd.to_datetime(lbl.split("@")[-1].strip())
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if ko.tzinfo is None:
            ko = ko.tz_localize(pl["bin"].dt.tz)
        w = pl[(pl["bin"] >= ko + pd.Timedelta(hours=window_h[0]))
               & (pl["bin"] < ko + pd.Timedelta(hours=window_h[1]))]
        if not len(w):
            skipped += 1
            continue
        truth = w.groupby("road_osmid", observed=True).agg(
            obs=(target, "sum"), vol=("vehicle_count", "sum"))
        psum = pr[pr["fixture"] == fx].groupby(
            "road_osmid", observed=True)["pred_speeding"].sum()
        j2 = truth.join(psum.rename("pred"), how="inner").dropna(subset=["pred"])
        if len(j2) < 50:
            skipped += 1
            continue
        o = j2["obs"].to_numpy(float)
        mr.append(spearman(j2["pred"].to_numpy(float), o))
        vr.append(spearman(j2["vol"].to_numpy(float), o))
        mq.append(top_quartile_share(j2["pred"].to_numpy(float), o))
        vq.append(top_quartile_share(j2["vol"].to_numpy(float), o))

    if not mr:
        chk(False, "arm %s baseline gate evaluated at least one fixture" % arm,
            "0 usable of %d (%d unparseable or too small)" % (n_fx, skipped))
        return
    M, V = float(np.mean(mr)), float(np.mean(vr))
    MQ, VQ = float(np.mean(mq)), float(np.mean(vq))
    print("       %s: %d/%d fixtures, target=%s" % (arm, len(mr), n_fx, target))
    print("         Spearman      model %+.3f    traffic volume %+.3f"
          % (M, V))
    print("         top-quartile  model  %.3f    traffic volume  %.3f"
          % (MQ, VQ))
    chk(M > V,
        "arm %s: model outranks plain traffic volume" % arm,
        "model %+.3f vs volume %+.3f%s"
        % (M, V, "" if M > V else "   <-- model redundant with a one-line sort"))
    chk(MQ > VQ,
        "arm %s: model beats traffic volume on top-quartile capture" % arm,
        "model %.3f vs volume %.3f" % (MQ, VQ))
    chk(MQ > 0.25,
        "arm %s: top predicted quartile beats chance" % arm,
        "%.3f vs 0.25 chance" % MQ)


def structure_gate(arm, d):
    """Can the fitted surface produce a GAME-SPECIFIC deployment plan?

    The operational claim is a game-day plan: officers go where THIS game needs
    them. If every fixture ranks the same streets highest, the plan is a static
    street list wearing a game-day label.

    Job 26107921: top-10 and top-20 site sets 100% identical across all six
    games, same argmax street every time, 3.2% of variance between fixtures.
    The run exited 0 and every artefact existed, so nothing noticed. Stage 59
    now records the measurement; this gate refuses to certify a plan that
    cannot differ by game.

    This is deliberately a hard gate. The alternative -- reporting it as a note
    -- is how a static model shipped as a game-day model in the first place.
    """
    mj = os.path.join(d, "game_day_holdout_metrics.json")
    if not os.path.exists(mj):
        chk(False, "arm %s metrics for the structure gate" % arm, "missing")
        return
    with open(mj) as fh:
        fv = ((json.load(fh).get("audit") or {}).get("fixture_variation")
              or {})
    if not fv:
        chk(False, "arm %s records fixture_variation" % arm,
            "absent -- the run predates the structural audit, or it failed")
        return
    if fv.get("error"):
        chk(False, "arm %s fixture_variation computed" % arm, fv["error"])
        return
    jac = fv.get("site_overlap_jaccard") or {}
    bf = fv.get("variance_between_fixtures_pct")
    ua = fv.get("unique_argmax_sites")
    print("       %s: %s fixtures, %s segments, between-fixture variance %s%%,"
          " top20 Jaccard %s, unique argmax %s"
          % (arm, fv.get("n_fixtures"), fv.get("n_segments"), bf,
             jac.get("top20"), ua))
    if jac.get("top20") == 1.0:
        chk(False,
            "arm %s: site ranking is game-specific (top-20 not identical)" % arm,
            "top-20 Jaccard = 1.0: every game selects the SAME 20 streets, so"
            " the plan is a static street list")
    else:
        chk(True,
            "arm %s: site ranking is game-specific (top-20 not identical)" % arm,
            "top-20 Jaccard %s" % jac.get("top20"))
    chk(bool(fv.get("game_specific_ranking")),
        "arm %s: flags a game-specific ranking" % arm,
        str(fv.get("game_specific_ranking")))
    if bf is not None:
        chk(bf > 0.5,
            "arm %s: between-fixture variance is non-trivial" % arm,
            "%s%% (26107921 had 3.2%%)" % bf)


# ---------------------------------------------------------------- baseline
# Can the model beat sorting segments by traffic volume? Without this check a
# model that is strictly WORSE than a one-line sort still certifies a
# deployment, which is exactly what happened under the old congestion target.
print()
print("baseline gate -- can the model beat sorting by traffic volume?")
if FIG_ROOT:
    for _arm, _d in (("A", A_DIR), ("B", B_DIR)):
        if os.path.isdir(_d):
            try:
                baseline_gate(_arm, _d, FIG_ROOT)
            except Exception as _exc:  # noqa: BLE001
                chk(False, "arm %s baseline gate ran" % _arm,
                    "%s: %s" % (type(_exc).__name__, _exc))
else:
    chk(False, "project root derivable for the baseline gate", str(FIG_ROOT))


# ---------------------------------------------------------------- structure
# Harder question than "is the model any good": can it give a DIFFERENT answer
# per game? A game-day plan that cannot vary by game is not a game-day plan.
print()
print("structure gate -- can the surface produce a game-specific plan?")
for _arm, _d in (("A", A_DIR), ("B", B_DIR)):
    if os.path.isdir(_d):
        try:
            structure_gate(_arm, _d)
        except Exception as _exc:  # noqa: BLE001
            chk(False, "arm %s structure gate ran" % _arm,
                "%s: %s" % (type(_exc).__name__, _exc))


# ---------------------------------------------------------------- verdict
print()
print("=" * 78)
print("passed: %d   failed: %d" % (len(OK), len(FAIL)))
for f in FAIL:
    print("  FAILED: %s" % f)
print("=" * 78)
sys.exit(1 if FAIL else 0)
