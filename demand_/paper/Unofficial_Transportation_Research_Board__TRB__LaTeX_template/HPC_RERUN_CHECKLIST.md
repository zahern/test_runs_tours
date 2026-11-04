# HPC Re-run Checklist — TRB Hard-Braking Paper

Generated 22 Aug 2026 after wiring the verified `hard_braking_events.csv` pipeline
(N=10,780 segments; 2,504,777 raw events; 1,415,932 segment-assigned) into
`trb_template.tex`. Local machine lacks jax/LaTeX; all items below need the HPC.

## 1. Model re-runs (nearmiss_hb_analysis.py)
> **Submitted 26 Aug 2026**: job `25082675.aqua` (`hb_crash_new`, run_hb.pbs) and
> job `25082739.aqua` (`nearmiss_crash_new`, run_nearmiss.pbs) — both QUEUED on
> cpu_batch_exec (4 cpu / 64 GB). `step_02_mcr_search` checkpoint cleared so the
> role search re-runs; all other HB checkpoints resume. Both jobs write into the
> paper folder, so `preprint_main.tex` picks up the refreshed tables/figures on
> the next compile. Post-completion: sync hand-written numbers (tab:mcr,
> tab:decomp_conv, §5.5/§5.11 narration) from the new CSVs, then recompile.
- [~] **SA feature subsets** on current sample — `sa_best_nm.tex`, `sa_best_crash.tex`
      and the §Dual-SA narration numbers (penalised BIC 68,467 / 29,743 are legacy 10,302-gen).
      → **covered by job `25082739`** (queued).
- [x] ~~**Significance-pruned MCR search**~~ — RESOLVED LOCALLY 22 Aug 2026:
      fixed-effect NB refits of both searched variable sets require **zero pruning
      steps** (all coefficients p<0.001; `pruned_specs_results.csv`,
      `pruned_specs.tex`, Table tab:pruned_specs). Degeneracy is attributable to the
      random-correlated block parameterisation. Remaining HPC task: significance-
      constrained ROLE search (which random parts to keep), not variable selection.
      → **unconstrained role search re-running via job `25082675`** (step_02 cleared).
      The significance-constrained variant remains a code change — do NOT edit
      nearmiss_hb_analysis.py while the job is queued (it executes the share copy).
      - HB: drop/fix rdm-ind SDs (IC p=0.103, IV_M p=0.166); degenerate rdm-cor means
        (RSDSO mean ≈ −17.04, p=0.337; PA_COM1 p=0.995; IV_M p=0.997); unlabeled param19–24.
        (param19–24 labelling now fixed at source — see section 2.)
      - Crash: degenerate SL mean ≈ −12.66 (untested); RSDSDG10 mean p=0.997;
        overdispersion insignificant once rdm-cor block included (p=0.121).
- [~] **RP-NB fixed-vs-random comparison table** on current data
      (`rp_nb_comparison.csv` is legacy: ΔBIC −1782 NM / −955 crash, both LR p≈0).
      → **covered by job `25082739`** (nearmiss_crash_analysis.py, ~line 1658).
- [x] ~~**Poisson-vs-NB + functional forms**~~ — DONE locally 22 Aug 2026
      (`local_model_comparison_results.csv`, `local_model_comparison.tex`;
       NB preferred LR=31,400.7; log-AADT wins by 362.7 BIC; raw tier counts add
       nothing directly, p=0.074).
- [~] **Joint-MLE copulas + structure search + conditional-prediction tables**
      (`copula_joint_comparison.csv`, `copula_search_best.csv`,
       Tables tab:copula_joint–tab:copula_pred currently flagged provisional).
      → **covered by job `25082739`** (queued). Note: the conditional-prediction
      support truncation K is now adaptive in nearmiss_hb_analysis.py; if the
      refreshed crash-side tables still look tail-truncated, port the same
      treatment to nearmiss_crash_analysis.py.
- [~] Random-parameters magnitude refresh for Discussion heterogeneity claims.
      → **covered by jobs `25082675` (role search) + `25082739` (RP-NB table)**.

## 2. Script bugs to fix before rerun
- [x] ~~Fragment generator: label correlated-block factor elements (currently `param15–param24`)~~
      FIXED AT SOURCE 26 Aug 2026: `fit_model` labels Cholesky elements
      `chol(<var_i>|<var_j>)` (lower-triangular, row-major) instead of anonymous paramN.
- [x] ~~Populate z/p for rdm-cor component rows (currently `--`)~~ PARTIAL 26 Aug 2026:
      rows are now labelled so `--` entries are interpretable; z/p stay `--` where
      the Hessian yields no usable SE (do not fabricate statistics).
- [x] ~~IRR column sign bug (e.g., crash param17 β=+0.268 but IRR=0.307)~~
      FIXED AT SOURCE 26 Aug 2026: IRR computed only for mean-type parameters
      (intercept/fixed/random-mean); SD/Cholesky/dispersion rows report `--`.
      Root cause: Estimate showed the softplus-transformed SD while IRR used
      exp(raw log-scale param).
- [x] ~~`trb_empirical_bayes.py` CV block: EB(reference-prior) RMSE=0.04 vs observed
      baseline 4.92 — implausible scaling~~ FIXED AT SOURCE 26 Aug 2026:
      reference-prior EB now shrinks with the reference model's OWN moment
      dispersion (`ref_prior_dispersion`), estimated on TRAIN folds only in CV.
      Previously the crash SPF dispersion (α≈18) collapsed w→0, making
      EB(reference) numerically the raw count. Also fixed the `\topruleEstimator`
      glued token and extra brace in the eb_results.tex writer.
- [x] LaTeX emission hardening 26 Aug 2026 (`nearmiss_hb_analysis.py`): `_tex_escape`
      applied to all `_save_tex_table` cells/captions/notes (bare underscores, bare
      `%`, `R^2` outside math mode; `$...$` spans pass through); HB category rows
      display as Candidate / Critical-Likely / Extreme.
- [x] Synthesis verdict 26 Aug 2026: weighs CON evidence (net = PRO − CON). 6/6
      previously printed "STRONG PROXY", contradicting the paper; now prints
      "COMPLEMENTARY SIGNAL". Net + verdict appended to hb_proxy_evidence.csv.
- [x] Conditional-prediction support 26 Aug 2026: `conditional_crash_prediction`
      truncation K now adaptive (per-segment NB CDF ≥ 1−1e-6, hard cap 200)
      instead of the arbitrary flat K=60. NOTE: copula-prediction tables
      (tab:copula_pred) were computed with K=60; re-run on HPC will refresh them.

## 2b. Time-based intervention strategy (NEW 26 Aug 2026)
- `hb_time_intervention.py`: hour×weekday intervention scheduling, gated on HB
  spatial transferability (dLL>0, ≥60% positive folds, coef stability ≥2).
  Current evidence PASSES the gate (dLL=+61.5, 9/11 folds, stability 7.17).
- Local run: `python hb_time_intervention.py` (uses temporal_risk_profile.csv).
  HPC run: add `--from-raw hard_braking_events_year.csv` to rebuild the profile
  from the 2.5M timestamped events AND the independent −0.30g scan (fills
  shape_rho; locally unevaluated). Floor rounding reproduces the paper's
  top-decile numbers (16/168 cells → 10.8% capture; weekday/weekend 0.825≈0.83).
- `--rank-by events` gives the volume-targeted variant (26.8% capture in the
  same 16 cells) for enforcement budgeting.
- Outputs: hb_time_intervention_{schedule,regional,summary}.csv, _heatmap.png,
  .tex appendix snippet (hook stubbed, commented, in preprint_main.tex).

## 2c. Random-walk specification validation (NEW 26 Aug 2026)
- `hb_randomwalk_validation.py`: Metropolis walk over inclusion masks
  (symmetric single-variable toggle, Boltzmann target, T=5 BIC), energy =
  significance-penalised BIC under the profile estimator (Poisson IRLS +
  Hauer moment alpha; raw NB2 MLE diverges on these counts). 600 steps from
  the final condensed specs, full sample AND 4,000-row subsample.
- Full-sample: accept 12%/11%, plateaus by step ~300/~150; stable cores
  17 (HB) / 10 (Crash) vars at P>=0.5; 7/11 and 6/8 selected vars at P=1.0;
  fragile selections exposed (SL/PA_COM1/IP in HB; RSDSDG10/MEDSFBar in Crash).
  Subsample: same headline behaviour, core membership shifts (Jaccard
  0.52/0.33) - sample sensitivity documented in the paper.
- Outputs: hb_randomwalk_trace.png, hb_randomwalk_stability.csv,
  hb_randomwalk_summary.csv, hb_randomwalk.tex - all wired into
  preprint_main.tex section 5.5 (fig:randomwalk, tab:randomwalk).

## 3. Build
- [x] ~~`latexmk trb_template.tex`~~ — SUPERSEDED 26 Aug 2026: the paper now builds
      as `preprint_main.tex` (preprint format, 50 pp, clean compile, zero
      undefined references; local MiKTeX). `trb_template.tex` is kept as the
      original TRB-format source but still contains the severity-threshold typo
      (line ~157) and the `Song2026` cite key, deliberately left untouched —
      port both fixes back before any TRB-format rebuild.
- [ ] After jobs `25082675` / `25082739` finish: verify regenerated tables/figures,
      sync hand-written numbers (tab:mcr, tab:decomp_conv, §5.5/§5.11), recompile
      `preprint_main.tex`.

## Already wired into the paper (no rerun needed)
DE benchmarks (hb_de_best_*.tex, fully significant), encompassing (LR=181.9),
proportionality, transfer (+124.7/+375.8), IFM copula comparison,
latent risk (ρ=0.147), category copulas, LGA transferability, synthesis table,
EB section, refreshed Data section + descriptive stats,
local fixed-effects model comparison (§model-comparison-local: NB-vs-Poisson,
log-AADT functional form, specification hierarchy incl. raw-tier null result),
significance-pruned spec resolution (tab:pruned_specs — zero steps needed),
risk-level screening calibration (§risk-cal / tab:risk_cal: top-decile PAI 6.31,
63.1% capture, slope 1.072),
out-of-sample robustness (§oos / tab:oos: cross-fitted EB ρ=0.887 PAI₁₀ 6.59;
calibrated NB/Poisson magnitude forecasts CITL≈1.01; heavy-tail limits documented),
temporal feasibility note (Conclusions: hour×weekday risk index, threshold-
robust diurnal shape ρ=0.98).
