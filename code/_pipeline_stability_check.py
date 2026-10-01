"""
_pipeline_stability_check.py
============================
Start-to-end stability audit for the tours pipeline. Fast (< ~3 min):
no real HHTS data needed — every stage contract is exercised on tiny
synthetic frames, every session fix is re-verified, and the live chain
state is reported.

Sections
--------
  A. environment + imports (HPC env, Larch 6.x + patches, SearchLibrium, JAX)
  B. cross-stage contracts  (dict overrides, enrichment, ridge inference,
     Larch ridge-aware covariance, joint household CDAP, salvage, merge)
  C. chain wiring           (afterany deps, GA_CHECKPOINT_DIR, MATSim export)
  D. live run state         (queued/running jobs, stage artifacts, markers)

Usage (on HPC):
    python -X utf8 _pipeline_stability_check.py [--skip-live]
Exit code 0 iff no FAIL.
"""
from __future__ import annotations

import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS: list[tuple[str, str, str]] = []          # (section, name, status/detail)


def check(section: str, name: str, fn, skip_if=None):
    if skip_if is not None:
        reason = skip_if()
        if reason:
            RESULTS.append((section, name, "SKIP"))
            print(f"  [SKIP] {section}/{name} — {reason}")
            return None
    try:
        out = fn()
        RESULTS.append((section, name, "PASS"))
        print(f"  [PASS] {section}/{name}"
              + (f" — {out}" if out else ""))
        return out
    except Exception as exc:                       # noqa: BLE001
        traceback.print_exc()
        RESULTS.append((section, name, f"FAIL: {type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {section}/{name} — {type(exc).__name__}: {exc}")
        return None


def section(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# --------------------------------------------------------------------------- #
def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    sys.path.insert(0, root)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-live", action="store_true",
                    help="don't query PBS / runs dir (e.g. off-HPC)")
    args = ap.parse_args()

    # ── A. environment ────────────────────────────────────────────────────
    section("A. environment + imports")
    import py_compile
    import tempfile  # noqa: F401  (used by the merge / MATSim round-trip checks)
    import pandas as pd  # noqa: F401  (used by several checks below)
    core = ["pipeline_logger.py", "ga_hypothesis_testing.py", "cdap.py",
            "cdap_households.py", "cdap_households_estimation.py",
            "stage1_activity_frequency.py", "stage2_day_duration.py",
            "stage3_activity_duration.py", "stage4_schedule_placement.py",
            "stage5_destination_choice.py", "stage6_plan_finalisation.py",
            "larch_patches.py", "larch_regularization.py",
            "_run_ga_single_stage.py", "merge_ga_results.py",
            "_smoke_pipeline_matsim.py"]

    def _compile():
        for f in core:
            py_compile.compile(f, doraise=True)
        src = open("job_submit_ga_parallel_chain.pbs", encoding="utf-8").read()
        assert "depend=afterany:" in src and "STAGE_MEM" in src
        return f"{len(core)} .py files + chain pbs"
    check("A", "py_compile all core files", _compile)

    def _imports():
        import pipeline_logger  # noqa: F401
        import stage1_activity_frequency  # noqa: F401
        import stage2_day_duration  # noqa: F401
        import stage3_activity_duration  # noqa: F401
        import stage4_schedule_placement  # noqa: F401
        import cdap_households_estimation  # noqa: F401
        from ga_hypothesis_testing import (  # noqa: F401
            ensure_ga_search_columns, _save_best_ckpt)
        return "pipeline + ga + cdap"
    check("A", "import core modules", _imports)

    def _larch_ver():
        import larch
        v = str(larch.__version__)
        assert v.startswith("6."), f"Larch {v} is not the 6.x line"
        return f"larch {v}"
    check("A", "Larch 6.x ('driftless' line)", _larch_ver)

    def _larch_patches():
        import larch_patches
        larch_patches.apply()
        from larch.model.from_spec import NumbaModel
        flags = {
            "d2_loglike.ridge": getattr(NumbaModel.d2_loglike, "_ridge_patched", False),
            "logsums.shim": getattr(NumbaModel.logsums, "_datatree_shim", False),
            "cov.pf_sync": getattr(NumbaModel.calculate_parameter_covariance,
                                   "_cov_sync", False),
        }
        bad = [k for k, v in flags.items() if not v]
        assert not bad, f"unbound: {bad}"
        return ", ".join(f"{k}=ok" for k in flags)
    check("A", "Larch runtime patches bound", _larch_patches)

    def _searchlib():
        from SearchLibrium.multinomial_logit import MultinomialLogit  # noqa: F401
        return "available"
    check("A", "SearchLibrium", _searchlib,
          skip_if=lambda: None if _try(lambda: __import__("SearchLibrium")) is not None
          else "not installed")

    def _jax():
        import jax
        return f"jax {jax.__version__}"
    check("A", "JAX", _jax)

    # ── B. cross-stage contracts ──────────────────────────────────────────
    section("B. cross-stage contracts (synthetic frames)")
    import numpy as np
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(0, 1, n)
    X_collinear = np.column_stack([z, z, 0.3 * z + rng.normal(0, .5, n)])

    def _s3_ridge():
        from stage3_activity_duration import BudgetLognormal
        y = np.clip(np.exp(1 + .5 * z + rng.normal(0, .4, n)), 5, None)
        m = BudgetLognormal()
        m.fit(X_collinear, y)
        assert np.isfinite(m.std_err_).all() and np.isfinite(m.p_value_).all()
        return f"λ={m.l2}"
    check("B", "stage3 ridge inference finite (collinear X)", _s3_ridge)

    def _s2_ridge():
        from stage2_day_duration import TobitModel
        y = np.clip(460 + 30 * z + rng.normal(0, 40, n), 300, 720)
        t = TobitModel(lo=300, hi=720)
        t.fit(X_collinear, y)
        assert np.isfinite(t.std_err_).all() and np.isfinite(t.p_value_).all()
        return f"λ={t.ridge}"
    check("B", "stage2 Tobit ridge inference finite", _s2_ridge)

    def _s4_ridge():
        os.environ["STAGE4_ESTIMATOR"] = "scipy"
        from stage4_schedule_placement import MNLTimingModel
        y = rng.integers(0, 6, n)
        mm = MNLTimingModel()
        mm.feature_names_ = ["a", "b", "c"]
        mm.purpose_ = "t"
        mm.fit(X_collinear, y)
        assert np.isfinite(mm.std_err_export_).all()
        return f"λ={mm.ridge}"
    check("B", "stage4 timing ridge inference finite", _s4_ridge)

    def _s1_ridge():
        from stage1_activity_frequency import OrderedLogit
        y = rng.integers(0, 4, n)
        ol = OrderedLogit()
        ol.fit(X_collinear, y)
        assert np.isfinite(ol.std_err_).all()
        return "GA_S1_RIDGE default"
    check("B", "stage1 ordered logit ridge inference finite", _s1_ridge)

    def _larch_cov():
        import larch as lx
        import pandas as pd
        from larch_regularization import elasticnet_objective
        ca = pd.DataFrame({
            "person_id": np.repeat(np.arange(n), 3),
            "altid": np.tile([0, 1, 2], n), "chosen": 0.0,
            "dup_a": np.repeat(z, 3), "dup_b": np.repeat(z, 3)})
        ca["chosen"] = ((ca.altid == 1) & (rng.random(len(ca)) < .4)).astype(float) \
            + ((ca.altid == 2) & (rng.random(len(ca)) < .2)).astype(float)
        ca["chosen"] = ca.groupby("person_id")["chosen"].transform(
            lambda s: (s == s.max()).astype(float))
        co = ca.groupby("person_id").first()[["dup_a", "dup_b"]]
        dt = lx.DataTree(
            base=lx.Dataset.construct.from_idca(ca.set_index(["person_id", "altid"])),
            person=lx.Dataset.construct.from_idco(co),
            relationships=("base.person_id @ person.person_id",))
        m = lx.Model(dt, compute_engine="numba")
        m.choice_ca_var = "chosen"
        m.utility_co[1] = lx.P("a1") + lx.P("ba") * lx.X("person.dup_a") \
            + lx.P("bb") * lx.X("person.dup_b")
        m.utility_co[2] = lx.P("a2")
        m.set_cap(25)
        with elasticnet_objective(m):
            m.maximize_loglike(options={"ftol": 1e-8})
            m.calculate_parameter_covariance()
        pf = m.pf
        assert pf["std_err"].notna().all() and pf["p_value"].notna().all()
        return "finite SE/p_value on singular design"
    check("B", "Larch ridge-aware covariance → pf", _larch_cov)

    def _s3_dict_override():
        import pandas as pd
        import stage3_activity_duration as s3
        persons = pd.DataFrame({
            "person_id": np.arange(300), "HHID": np.repeat(np.arange(75), 4),
            "ptype": rng.choice(["FullTimeWorker", "SchoolChild", "Retired"], 300),
            "sexp": rng.integers(1, 3, 300), "vehrd": rng.integers(0, 3, 300),
            "nprd": rng.integers(1, 5, 300), "WORKSTAT": rng.integers(0, 3, 300),
            "STUDYING": rng.integers(0, 5, 300), "incp": rng.integers(1, 15, 300),
            "AGEGROUP": rng.integers(1, 12, 300),
            "CARLICENCE": rng.integers(0, 2, 300)})
        tours = pd.DataFrame({
            "person_id": rng.integers(0, 300, 900),
            "tour_id": np.arange(900),
            "main_purpose": rng.choice(
                ["Full-time Work", "Shopping", "Social Visit"], 900),
            "duration": rng.uniform(20, 200, 900),
            "start_time": rng.uniform(400, 1000, 900)})
        tours["end_time"] = tours["start_time"] + tours["duration"]
        ga = {"work": ["start_hr", "log_dist", "B", "CARLICENCE",
                       "hh_alone", "income_veryhi"]}
        models = s3.estimate_duration_models(tours, persons,
                                             feature_override=ga, min_sample=30)
        wf = set(models["work"]["features"])
        assert {"CARLICENCE", "hh_alone", "income_veryhi"} <= wf, wf
        assert "vehrd" not in wf, "default-spec leakage"
        return "engineered GA features reach the final model"
    check("B", "stage3 GA dict-override → enrichment → refit", _s3_dict_override)

    def _joint_cdap():
        import cdap_households
        from cdap_households_estimation import estimate_joint_cdap_by_hh_size
        rows, tr = [], []
        pid = 0
        for hh in range(1, 121):
            size = int(rng.choice([1, 2, 3]))
            for m_ in range(size):
                child = size > 1 and m_ == size - 1
                rows.append({"person_id": pid, "HHID": hh,
                             "ptype": "SchoolChild" if child else "FullTimeWorker",
                             "sexp": 2, "vehrd": 1, "CARLICENCE": 0,
                             "incp": 8, "AGEGROUP": 2 if child else 6})
                if not child:
                    tr.append({"person_id": pid, "main_purpose": "Full-time Work"})
                pid += 1
        res = estimate_joint_cdap_by_hh_size(
            pd.DataFrame(rows), pd.DataFrame(tr),
            cdap_households.HouseholdCdapConfig(), per_size=True)
        assert res["sizes"] and res["interaction"] is not None
        return f"sizes={sorted(res['sizes'])} converged={res['converged']}"
    check("B", "joint household CDAP (per-size)", _joint_cdap)

    def _salvage():
        import tempfile
        import _run_ga_single_stage as rgs
        from ga_hypothesis_testing import GAResult, Individual
        d = tempfile.mkdtemp()
        os.environ["GA_CHECKPOINT_DIR"] = d
        r = GAResult(stage=3, best_features=["x"], best_fitness=-0.9,
                     best_individual=Individual(genes=np.zeros(2, dtype=np.int8)),
                     history=[], all_candidates=[], feature_names=["x"],
                     fitness_metric="m", n_restarts=1, cache_hits=0)
        with open(os.path.join(d, "ga_stage3_purposes.pkl"), "wb") as fh:
            import pickle as pk
            pk.dump({"Stage3   Work": {"seg_result": r, "feats": ["x", "y"]}}, fh)
        res, n = rgs._salvage_partial_result(3)
        assert res is not None and n == 1
        return "stage-3 partial rebuilt"
    check("B", "salvage partial result (stage 3)", _salvage)

    def _merge_fallback():
        import subprocess
        import tempfile
        base = tempfile.mkdtemp(prefix="merge_t_")
        st3 = os.path.join(base, "stage3")
        os.makedirs(st3, exist_ok=True)
        import pickle as pk
        from ga_hypothesis_testing import AllStagesResult
        a = AllStagesResult()
        # minimal GAResult to merge
        from ga_hypothesis_testing import GAResult, Individual
        a.results[3] = GAResult(stage=3, best_features=["x"], best_fitness=-1.0,
                                best_individual=Individual(genes=np.zeros(1, dtype=np.int8)),
                                history=[], all_candidates=[], feature_names=["x"],
                                fitness_metric="m", n_restarts=0, cache_hits=0)
        with open(os.path.join(st3, "ga_results_checkpoint.pkl"), "wb") as fh:
            pk.dump(a, fh)
        out = subprocess.run(
            [sys.executable, "-X", "utf8", "merge_ga_results.py",
             "--run-root", os.path.dirname(base), "--run-tag",
             os.path.basename(base), "--stage-ids", "3", "--strict"],
            capture_output=True, text=True)
        merged = os.path.join(base, "merged", "ga_results.pkl")
        assert os.path.exists(merged), out.stdout + out.stderr
        with open(merged, "rb") as fh:
            m = pk.load(fh)
        assert 3 in m.results
        return "checkpoint preferred over missing ga_results.pkl"
    check("B", "merge prefers salvaged checkpoint", _merge_fallback)

    def _matsim_roundtrip():
        import matsim_export as mx
        import xml.etree.ElementTree as ET
        trips = pd.DataFrame({
            "person_id": [1, 1, 2], "row_type": "trip",
            "OTAZ": [1, 2, 1], "DTAZ": [2, 1, 3],
            "departure_time": [400.0, 500.0, 420.0],
            "mode_name": ["DA", "Walk", "Transit"],
            "purpose": ["work", "home", "shop"],
            "origin_lat": [-27.4, -27.5, -27.4], "origin_lon": [153.0, 153.1, 153.0],
            "dest_lat": [-27.5, -27.4, -27.6], "dest_lon": [153.1, 153.0, 153.2]})
        persons, stats = mx.build_plans(trips, {}, transform=(lambda lo, la: (lo, la)))
        assert len(persons) == 2 and stats["n_legs"] == 3
        d = tempfile.mkdtemp()
        p = os.path.join(d, "plans.xml")
        mx.write_plans_xml(persons, p)
        ET.parse(p)
        return f"{stats['n_persons_out']} persons / {stats['n_legs']} legs"
    check("B", "MATSim plans round-trip", _matsim_roundtrip)

    # ── C. chain wiring ───────────────────────────────────────────────────
    section("C. chain wiring")

    def _afterany():
        src = open("job_submit_ga_parallel_chain.pbs", encoding="utf-8").read()
        n_any = src.count("depend=afterany:")
        n_ok = src.count("depend=afterok:")
        assert n_any >= 6 and n_ok == 0, f"afterany={n_any} afterok={n_ok}"
        return f"{n_any} afterany deps, no afterok"
    check("C", "orchestrator uses afterany (chain survives stage failure)", _afterany)

    def _ckpt_dir_set():
        src = open("_run_ga_single_stage.py", encoding="utf-8").read()
        assert "GA_CHECKPOINT_DIR" in src and "_salvage_partial_result" in src
        return "checkpoint dir + salvage wired"
    check("C", "stage runner checkpoints + salvage wired", _ckpt_dir_set)

    def _merge_ckpt():
        src = open("merge_ga_results.py", encoding="utf-8").read()
        assert "ga_results_checkpoint.pkl" in src
        return "merge falls back to salvaged checkpoint"
    check("C", "merge checkpoint fallback", _merge_ckpt)

    def _best_ckpt_hook():
        src = open("ga_hypothesis_testing.py", encoding="utf-8").read()
        n = src.count("_save_best_ckpt(")
        assert n >= 8, f"only {n} hook sites"
        return f"{n} save-on-new-best hook sites (4 engines, main + restarts)"
    check("C", "save-on-new-best hooks present", _best_ckpt_hook)

    # ── D. live run state ─────────────────────────────────────────────────
    if not args.skip_live:
        section("D. live run state")
        import subprocess

        def _qstat():
            out = subprocess.run(["qstat", "-u", os.environ.get("USER", "ahernz")],
                                 capture_output=True, text=True).stdout
            lines = [l for l in out.splitlines() if l.strip()][2:]
            running = [l.split()[0] for l in lines if l.split()[3] == "R"]
            queued = [l.split()[0] for l in lines if l.split()[3] == "Q"]
            return f"{len(running)} running, {len(queued)} queued"
        check("D", "PBS jobs", _qstat)

        def _chain_state():
            base = os.path.join("runs", "ga_parallel", "sanity371")
            if not os.path.isdir(base):
                return "sanity371 not started yet"
            parts = []
            for s in (0, 1, 2, 3, 4, 5):
                d = os.path.join(base, f"stage{s}")
                if not os.path.isdir(d):
                    continue
                marks = []
                if os.path.exists(os.path.join(d, "PARTIAL")):
                    marks.append("PARTIAL")
                if os.path.exists(os.path.join(d, "ga_results.pkl")):
                    marks.append("done")
                for ck in ("ga_stage0", "ga_stage1_purposes", "ga_stage2_targets",
                           "ga_stage3_purposes", "ga_stage5_segments"):
                    if os.path.exists(os.path.join(d, ck + ".pkl")):
                        marks.append("ckpt")
                        break
                parts.append(f"s{s}:{','.join(marks) or 'pending'}")
            return " | ".join(parts) if parts else "no stage dirs yet"
        check("D", "sanity371 stage artifacts", _chain_state)

    # ── summary ───────────────────────────────────────────────────────────
    fails = [r for r in RESULTS if r[2].startswith("FAIL")]
    skips = [r for r in RESULTS if r[2] == "SKIP"]
    print("\n" + "=" * 70)
    print(f"STABILITY CHECK: {len(RESULTS) - len(fails) - len(skips)} PASS / "
          f"{len(fails)} FAIL / {len(skips)} SKIP")
    for sec, name, detail in fails:
        print(f"  FAIL {sec}/{name}: {detail}")
    print("=" * 70)
    return 1 if fails else 0


def _try(fn):
    try:
        fn()
        return True
    except Exception:                              # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
