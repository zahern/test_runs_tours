"""Opt-in latent-class gene block for the Stage-5 destination GA.

When ``STAGE5_LC_GENES=1`` the destination chromosome is extended with a
structure block appended AFTER the base destination terms (so base gene
indices, collinear groups, require-one groups and sidecars stay valid):

    lc_on            fit a latent-class model instead of the pooled MNL
    lc_nc3           three classes instead of two
    lc_gap_mid       hard constraint: crash_class_gap >= 0.5
    lc_gap_hi        hard constraint: crash_class_gap >= 1.0
    lc_floor_hi      share_floor = 0.15 (default 0.10)
    lc_memb:<var>    person-type membership covariate (subset of the pool)

Decode + repair is enforced inside the fitness, so the GA operators never
need to know about the block:

  * ``lc_on`` forces a crash component into the utility spec (the estimator
    refuses a latent-class fit without one) and at least one membership
    covariate (falling back to every available person type);
  * a gap gene maps to a hard fitness constraint, so the GA actively searches
    for a spec where a CRASH-defined class exists at acceptable BIC instead
    of accepting whatever split EM happens to find.

Pooled genomes (``lc_on=0``) are delegated to the existing larch fitness by
the caller, so a run with the block enabled still contains the current model
as a competitor and BIC/n remains comparable (both arms are exact MNL
likelihoods).

The membership gene pool is BEHAVIORAL person types only (STUDYING, age5p,
sexp, safety_sensitivity); structural scale vars (incp, vehrd) stay out of
the genome unless re-added via STAGE5_LC_MEMB_EXTRA.

The identification doctor (``identify_weak_features``, mirrors larch's
overspecification check) can guard every LC eval via STAGE5_LC_DOCTOR:
"reject" fails flagged genomes (any near-null direction, condition number
over the cap, or a suggested drop); "prune" drops the doctor's suggested
features (crash components protected) and refits once.  Thresholds:
STAGE5_LC_DOCTOR_TOL / _MAX_WEAK / _MAX_COND.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

BEHAVIORAL_MEMBERSHIP_VARS = [
    "STUDYING", "age5p", "sexp", "safety_sensitivity",
]
STRUCTURAL_MEMBERSHIP_VARS = ["incp", "vehrd"]
DEFAULT_MEMBERSHIP_VARS = list(BEHAVIORAL_MEMBERSHIP_VARS)

_CRASH_PREFERENCE = [
    "c_crash", "c_crash_safety_dest", "c_crash_smooth",
]

_STRUCTURE_GENES = ["lc_on", "lc_nc3", "lc_gap_mid", "lc_gap_hi",
                    "lc_floor_hi"]


def default_membership_vars() -> List[str]:
    return list(DEFAULT_MEMBERSHIP_VARS)


def lc_gene_names(membership_vars: Optional[Sequence[str]] = None) -> List[str]:
    """Names of the appended structure block, in chromosome order."""
    mv = list(membership_vars) if membership_vars else default_membership_vars()
    return _STRUCTURE_GENES + [f"lc_memb:{v}" for v in mv]


def extend_candidates(base_candidates: Sequence[str],
                      membership_vars: Optional[Sequence[str]] = None
                      ) -> List[str]:
    """Base destination candidates + the LC structure block (append-only).

    The default membership gene pool is BEHAVIORAL person types only; extra
    (structural) vars can be appended via STAGE5_LC_MEMB_EXTRA="incp,vehrd",
    or an explicit membership_vars list (trusted as passed, e.g. tests).
    """
    if membership_vars is None:
        membership_vars = list(default_membership_vars())
        _extra = os.environ.get("STAGE5_LC_MEMB_EXTRA", "").strip()
        if _extra:
            membership_vars = list(dict.fromkeys(
                membership_vars
                + [v.strip() for v in _extra.split(",") if v.strip()]))
    return list(base_candidates) + lc_gene_names(membership_vars)


def decode_lc_spec(genes: np.ndarray,
                   candidates: Sequence[str],
                   base_candidates: Sequence[str],
                   available_membership: Optional[Iterable[str]] = None,
                   crash_tokens: Sequence[str] = ("crash",),
                   require_crash: bool = True) -> dict:
    """Genome -> LC spec dict with repair.

    Returns ``{"lc_on": False, "features": base_selected, ...}`` for pooled
    genomes; the caller then runs the existing larch path.  For LC genomes the
    repair guarantees a crash component and a non-empty membership subset.
    """
    genes = np.asarray(genes, dtype=int).ravel()
    bits = {str(c): int(g) for c, g in zip(candidates, genes)}
    base = [c for c in base_candidates if bits.get(str(c), 0) == 1]
    spec = {
        "lc_on": False,
        "features": base,
        "n_classes": 1,
        "membership_vars": [],
        "gap_min": 0.0,
        "share_floor": 0.0,
        "n_init": 1,
        "perturb_sigma": 0.0,
        "maxiter": 1,
        "membership_l2": 0.0,
    }
    # Auto-enable the LC arm when any structure gene is set: otherwise those
    # genes are no-ops for pooled genomes and the GA drifts neutrally over
    # them instead of exploring the latent-class landscape.
    _lc_any = (bits.get("lc_on", 0)
               or any(bits.get(g, 0) for g in _STRUCTURE_GENES)
               or any(str(c).startswith("lc_memb:") and bits.get(str(c), 0)
                      for c in candidates))
    if not _lc_any:
        return spec

    # Repair 1: at least one class-specific component must be crash-based.
    # Prefer a crash coef already in the pool; otherwise append the canonical
    # c_crash and let the frame-coef mapping decide (a segment whose frame has
    # no crash column then fails loudly in the estimator instead of silently
    # dropping the requirement).
    if require_crash and not any(
            any(t.lower() in str(c).lower() for t in crash_tokens)
            for c in base):
        _pool_crash = [c for c in base_candidates
                       if any(t.lower() in str(c).lower()
                              for t in crash_tokens)]
        if _pool_crash:
            base = base + [_pool_crash[0]]
        else:
            for c in _CRASH_PREFERENCE:
                if c in base_candidates:
                    base = base + [c]
                    break
            else:
                base = base + [_CRASH_PREFERENCE[0]]

    # Repair 2: membership subset over BEHAVIORAL person types only.  Explicitly
    # selected lc_memb bits are honoured; the silent fallback is restricted to
    # the behavioral pool (intersected with the available frame columns).
    avail = [str(v) for v in (available_membership or [])]
    memb = [c.split(":", 1)[1] for c in candidates
            if str(c).startswith("lc_memb:") and bits.get(str(c), 0) == 1]
    if avail:
        memb = [m for m in memb if m in avail]
    if not memb:
        _behav = [v for v in default_membership_vars()
                   if (not avail or v in avail)]
        memb = _behav or list(avail or []) or default_membership_vars()

    spec.update({
        "lc_on": True,
        "features": base,
        "n_classes": 3 if bits.get("lc_nc3", 0) else 2,
        "membership_vars": memb,
        "gap_min": (1.0 if bits.get("lc_gap_hi", 0)
                    else 0.5 if bits.get("lc_gap_mid", 0) else 0.0),
        "share_floor": 0.15 if bits.get("lc_floor_hi", 0) else 0.10,
        "n_init": int(os.environ.get("STAGE5_LC_N_INIT", "2")),
        "perturb_sigma": float(os.environ.get("STAGE5_LC_SIGMA", "0.5")),
        "maxiter": int(os.environ.get("STAGE5_LC_MAXITER", "150")),
        "membership_l2": float(os.environ.get("STAGE5_LC_MEMB_L2", "0.1")),
    })
    return spec


def doctor_gate(frame, feature_cols: Sequence[str], *,
                crash_tokens: Sequence[str] = ("crash",),
                mode: Optional[str] = None,
                max_weak: Optional[int] = None,
                max_cond: Optional[float] = None,
                tol: Optional[float] = None) -> Tuple[list, dict, Optional[str]]:
    """Run the SearchLibrium identification doctor on a candidate spec.

    Eigendecomposes the scaled exact information matrix (mirrors larch's
    overspecification check) and reports orthogonality/identification via the
    condition number and near-null directions.  Crash components are protected
    so the class-specific crash requirement survives pruning.

    Modes (STAGE5_LC_DOCTOR): "0" = off; "reject" = fail flagged genomes;
    "prune" = drop the doctor's suggested_drop and refit once (rejects when
    nothing actionable can be pruned).  Returns
    (kept_feature_cols, doctor_info, reject_reason_or_None).
    """
    from SearchLibrium.mh_choice_estimator import identify_weak_features

    mode = (mode if mode is not None
            else os.environ.get("STAGE5_LC_DOCTOR", "0")).strip().lower()
    info: dict = {"doctor_mode": mode}
    if mode in ("0", "off", "no", "false", ""):
        return list(feature_cols), info, None
    if mode not in ("reject", "prune"):
        raise ValueError(f"STAGE5_LC_DOCTOR must be 0|reject|prune, got {mode!r}")
    if tol is None:
        tol = float(os.environ.get("STAGE5_LC_DOCTOR_TOL", "1e-8"))
    if max_weak is None:
        max_weak = int(os.environ.get("STAGE5_LC_DOCTOR_MAX_WEAK", "0"))
    if max_cond is None:
        max_cond = float(os.environ.get("STAGE5_LC_DOCTOR_MAX_COND", "1e12"))
    crash_cols = [f for f in feature_cols
                  if any(t.lower() in str(f).lower() for t in crash_tokens)]
    try:
        rep = identify_weak_features(frame, list(feature_cols), tol=tol,
                                     protected=crash_cols)
    except Exception as exc:
        info["doctor_error"] = f"{type(exc).__name__}: {exc}"
        return list(feature_cols), info, None
    n_weak = len(rep.weak_directions)
    cond = float(rep.condition_number)
    drops = [str(d) for d in rep.suggested_drop]
    info.update({"doctor_n_weak": n_weak, "doctor_condition": cond,
                 "doctor_drops": drops, "doctor_protected": crash_cols})
    flagged = (n_weak > max_weak) or (cond > max_cond) or bool(drops)
    if not flagged:
        return list(feature_cols), info, None
    reason = (f"doctor flagged spec: n_weak={n_weak} (max {max_weak}), "
              f"cond={cond:.2e} (max {max_cond:.2e}), drops={drops}")
    if mode == "reject" or not drops:
        return list(feature_cols), info, reason
    kept = [f for f in feature_cols if f not in set(drops)]
    info["pruned"] = drops
    if not kept:
        return kept, info, f"{reason}; pruning left no features"
    if crash_cols and not any(f in kept for f in crash_cols):
        return kept, info, f"{reason}; pruning removed the crash component"
    return kept, info, None


def fit_lc_genome(frame, coef_to_col: dict, spec: dict, *,
                  complexity_penalty: float = 1.0, seed: int = 7,
                  require_crash: bool = True) -> Tuple[float, dict]:
    """Fit a decoded LC genome on the long frame.

    Returns ``(score, info)`` with ``score = -BIC/n`` (GA convention) or the
    -1e9 failure sentinel; ``info`` always explains the outcome.  Hard
    constraints (all failures, not soft penalties):

      * the estimator must report ``converged``;
      * no class may collapse below half the share floor;
      * ``crash_class_gap`` must clear the genome's ``gap_min`` when set.
    """
    from SearchLibrium.mh_choice_estimator import estimate_dest_choice_lc

    if not spec.get("lc_on"):
        return None, {"reason": "pooled genome (caller uses larch path)"}
    feature_cols = [coef_to_col[c] for c in spec["features"] if c in coef_to_col]
    if not feature_cols:
        return -1e9, {"reason": "no frame columns for selected features",
                      "features": list(spec["features"])}
    feature_cols, doctor_info, doctor_reject = doctor_gate(frame, feature_cols)
    if doctor_reject is not None:
        out = {"reason": doctor_reject, "features": list(spec["features"])}
        out.update(doctor_info)
        return -1e9, out
    memb = [m for m in spec.get("membership_vars", []) if m in frame.columns]
    try:
        res = estimate_dest_choice_lc(
            frame, feature_cols,
            n_classes=int(spec["n_classes"]),
            membership_vars=memb or None,
            n_init=int(spec["n_init"]),
            perturb_sigma=float(spec["perturb_sigma"]),
            share_floor=float(spec["share_floor"]),
            membership_l2=float(spec["membership_l2"]),
            maxiter=int(spec["maxiter"]),
            seed=int(seed),
            require_crash=bool(require_crash),
        )
    except Exception as exc:
        return -1e9, {"reason": f"{type(exc).__name__}: {exc}",
                      "features": list(spec["features"])}
    shares = [float(s) for s in res["shares"]]
    info = {
        "ll": float(res["loglike"]),
        "bic": float(res["bic"]),
        "shares": shares,
        "converged": bool(res.get("converged")),
        "iterations": int(res.get("iterations", 0)),
        "crash_gap": float(res.get("crash_class_gap") or 0.0),
        "class_crash": res.get("class_crash"),
        "membership_vars": memb,
        "features": list(spec["features"]),
        "restart_ll_span": float(res.get("restart_ll_span", 0.0) or 0.0),
    }
    _require_conv = os.environ.get("STAGE5_LC_REQUIRE_CONV", "1").strip() != "0"
    if not info["converged"] and _require_conv:
        info["reason"] = f"not converged (iters={info['iterations']})"
        return -1e9, info
    if shares and min(shares) < 0.5 * float(spec["share_floor"]):
        info["reason"] = f"class collapse shares={shares}"
        return -1e9, info
    if float(spec.get("gap_min", 0.0)) > 0 \
            and info["crash_gap"] < float(spec["gap_min"]):
        info["reason"] = (f"crash gap {info['crash_gap']:.2f} < "
                          f"gap_min {spec['gap_min']:.2f}")
        return -1e9, info
    d = len(feature_cols)
    n = int(res["n_cases"])
    k = (int(spec["n_classes"]) * d
         + (int(spec["n_classes"]) - 1) * (len(memb) + 1 if memb else 0))
    score = -(-2.0 * info["ll"]
              + float(complexity_penalty) * k * np.log(max(n, 2))) / max(n, 1)
    info["k"] = k
    info["n"] = n
    info.update(doctor_info)
    return float(score), info
