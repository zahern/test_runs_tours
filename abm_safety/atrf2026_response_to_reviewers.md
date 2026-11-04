# ATRF 2026 — Response to Reviewers (submission 124)

*Tour Defence: Extensive Hypothesis Testing in Safety-Augmented Travel Demand Models*

Revision source: `updated_paper.tex` (changes marked `\textcolor{red}`), formatted
paper: `atrf2026.tex`. Run evidence cited from `runs/ga_parallel/sa_bandit__guided01`
/ `sa_bandit_staged__guided01` and the demand_ data-build chain.

---

## Reviewer 1

**R1.1 — "does not adequately discuss how the safety metric impacts destination
choice."**
Addressed twice over. (a) The paper now details the destination-side safety terms:
`c_crash` is a *forced-include* variable in the destination-choice candidate pool
(constraint architecture, §Constraint Architecture), and the results section
(§Safety-driven destination avoidance) reports the estimated destination crash
coefficients (negative and significant in all three segments of the current run:
work/edu/nonmand `c_crash` ≈ −0.19 to −0.24). (b) Since submission we extended
safety *upstream* of destination choice as well: a time-decayed home-zone crash-
exposure accessibility (`crash_acc = log(1+Σ_j e^{−βt_ij}·CRASH_ij)`) now enters
the coordinated daily activity pattern (CDAP) model and the activity-generation
candidate pools, so crash exposure can suppress participation itself, not just
re-route it. Described in §Safety as a Behavioral Determinant.

**R1.2 — "how crash records are perceived... road geometry, speed limits, road
conditions, traffic volume, heavy vehicles."**
A dedicated construct-clarification paragraph now states explicitly that we model
*objective crash exposure as a proxy for perceived risk*, not perceived safety
itself; it enumerates which perception-relevant factors the SPFs embed (geometry,
speed environment, volume) and which they cannot (lighting, harassment risk,
personal experience, media salience), and states that resolving the proxy
relationship requires perception data. (Same paragraph answers R2.1.)

**R1.3 — "GA... does not provide sufficient detail on parameters, constraints,
convergence criteria."**
The revision carries: §SA-Bandit Hybrid Search (Thompson-sampling proposal
mechanism, SA acceptance rule, temperature auto-calibration, budget structure
pop×gens with named budget tiers fast/standard/thorough = 100/350/900
generation-equivalents), §Constraint Architecture (force-include / never-random /
collinearity-group repairs), Appendix §GA Operator Definitions (bit-flip rates,
diversity threshold δ=0.10), Appendix §Candidate Variable Pools (full pools per
stage), and §Reproducing All Experiments (exact submission scripts + environment
variables). Random-parameter (mixed-logit) selection is now itself a searched
gene in the chromosome rather than a post-hoc toggle.

**R1.4 — Figure 5 heatmap readability (values in cells, colour band).**
Accepted; the regenerated figure adds per-cell counts and a perceptually-uniform
colour scale so off-peak/minor purposes are legible.

**R1.5 — Figure 4 vs Figure 5 edu-bike inconsistency.**
Root cause found and fixed in code, not just in the figure: the mode-share
*application* path used a flat softmax over coefficients estimated under a nested
structure, plus a NaN-attribute bug that silently deleted active-mode shares.
Both fixed (nested-logit probability at prediction time; NaN attribute
imputation); figures regenerate from one pipeline path so they cannot diverge.

**R1.6 — Table 2 heading ("Lower share without crash penalty").**
Fixed via an explicit sign convention stated once in §Metrics Evaluated:
Δ = share(with safety) − share(without safety); headings now spell the direction
in words. (Same fix answers R2.2.)

**R1.7 — typos ("pracitices", "eveyrone").** Fixed.

## Reviewer 2

**R2.1 — perceived vs objective safety constructs conflated.**
See R1.2 — the requested statement now appears verbatim in spirit: objective
crash exposure as a proxy for perceived risk, unless perception data are
available.

**R2.2 — transit/walk shares higher WITH safety yet labelled "lower with crash
penalty"; sign convention unstated.**
See R1.6. With the convention stated, the observed positive transit/walk Δ reads
unambiguously: penalising car-borne crash exposure shifts demand *toward*
transit/walk. The counter-intuitive labelling was an editing error, not a result
change.

**R2.3 — conclusions overstate heterogeneity/policy findings.**
Softened as requested: the heterogeneity-by-person-type/time-of-day/neighbourhood
and safety-investment claims are now framed as extensions the architecture
supports, with only the aggregate evidence claimed as demonstrated.

**R2.4 — OD crash exposure construction under-specified (route assumptions,
normalization).**
A new paragraph in §Origin–Destination crash frequency matrix states: exposure is
accumulated along the single network shortest path between zone centroids (no
stochastic route set); the skim is TOTAL expected crashes per year along the
path, not a per-km/min/vehicle rate; distance and time enter utilities as
separate terms, so the crash coefficient is conditional on them (which makes the
negative estimates conservative); log and per-km variants were offered to the
variable search as alternative functional forms.

---

### Note on regenerated results
Between submission and revision the estimation/application pipeline was repaired
in ways that change numbers (nested-logit application path, logsum unit fix,
zone universe extended to the full 371-zone SEQ system including Tweed and the
NSW Richmond Valley zones, identification priors on the nested logit). All
tables/figures in the revision are regenerated from the corrected pipeline; the
qualitative findings (crash terms consistently selected by the search; omitting
safety misestimates active-mode use) are unchanged.
