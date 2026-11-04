"""SparseEA-AGDS-style solver for binary inclusion-mask search.

Adaptation of the SparseEA-AGDS algorithm (evolutionary algorithm with
adaptive genetic operator and dynamic scoring mechanism for large-scale
sparse optimisation; Wang et al., Sci Rep 15:9267, 2025) to the paper's
single-objective variable-selection problem:

    minimise  penalised BIC(mask)  over  mask in {0,1}^J

The original is a many-objective algorithm (SparseEA framework + adaptive
genetic operator + dynamic variable scoring + reference-point environmental
selection). The adaptation keeps the three distinctive mechanisms and maps
them onto this problem as follows:

  * dec/mask representation ... binary inclusion mask (gene j = 1 <=> variable
    j in the NB specification). This is the natural "sparse" encoding here:
    J = 30 candidates, feasible masks are sparse by the penalised-BIC fitness.
  * adaptive genetic operator ... per-individual crossover (pc) and mutation
    (pm) probabilities are updated every generation from the change in the
    individual's non-dominated layer: individuals that move to a better
    (lower) layer get lower pc/pm (protect good solutions); individuals that
    fall back to a worse layer get higher pc/pm (explore). This is Algorithm 1
    of the paper in single-objective form.
  * dynamic scoring mechanism ... each decision variable j carries a score
    s_j, updated each generation by weighted accumulation over the layer
    changes of the individuals that select j (Algorithm 2 of the paper).
    Mask mutation and uniform crossover are biased toward high-score
    variables, so superior variables gain extra genetic opportunities and the
    search favours sparse, high-quality masks.
  * environmental selection ... the original uses reference-point selection
    for many objectives. Because the problem here is single-objective
    (sparsity is already priced into the penalised-BIC fitness), truncation
    is lexicographic (penalised BIC, then model size). Dominance layers over
    (penalised BIC, model size) are still computed every generation, but only
    to drive the adaptive operator and the dynamic scoring -- their original
    role -- not truncation. Documented as a stand-in, not the full procedure.
  * stagnation-triggered local mode ... when the best has not improved for
    10 generations, per-locus rates shrink so offspring become 1-2 bit
    neighbours (hill-climbing) instead of destructive macro-mutations. This
    is a no-restart cooling schedule, consistent with the paper's single-run
    policy.

Single continuous run, no restarts, fully seeded for reproducibility.

Public API
----------
sparsea_agds_search(fitness_fn, J, pop_size=60, max_gen=60, seed=42,
                    min_active=2, pc0=0.9, pm0=0.15)
    fitness_fn(mask) -> float (minimise; penalised BIC; 1e12 = infeasible).
    Returns dict(best_mask, best_fitness, best_size, history_best,
    history_gen_best, history_mean/median/min/max, history_pop, scores).
    history_pop[g] is the full per-individual fitness vector of the
    selected population at generation g (gen 0 = init), so callers can
    plot best-current plus faint population traces / bands.
"""

import numpy as np


def _dominance_layers(objs):
    """Non-dominated sorting on (penalised BIC, model size); minimise both.

    Returns integer layer per individual (0 = non-dominated front).
    O(P^2) -- fine for P ~ tens and G ~ tens.
    """
    P = objs.shape[0]
    layer = np.full(P, -1, dtype=int)
    remaining = np.ones(P, dtype=bool)
    cur = 0
    while remaining.any():
        idx = np.where(remaining)[0]
        front = np.ones(len(idx), dtype=bool)
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b:
                    continue
                # b dominates a if b <= a on both and < on at least one
                if (objs[idx[b], 0] <= objs[idx[a], 0]
                        and objs[idx[b], 1] <= objs[idx[a], 1]
                        and (objs[idx[b], 0] < objs[idx[a], 0]
                             or objs[idx[b], 1] < objs[idx[a], 1])):
                    front[a] = False
                    break
        layer[idx[front]] = cur
        remaining[idx[front]] = False
        cur += 1
    return layer


def _crowding(objs, layer):
    """Crowding distance within each layer (normalised per objective)."""
    P = objs.shape[0]
    dist = np.zeros(P)
    for L in np.unique(layer):
        idx = np.where(layer == L)[0]
        if len(idx) <= 2:
            dist[idx] = np.inf
            continue
        for m in range(objs.shape[1]):
            order = idx[np.argsort(objs[idx, m])]
            span = objs[order[-1], m] - objs[order[0], m]
            dist[order[0]] = dist[order[-1]] = np.inf
            if span <= 0:
                continue
            for k in range(1, len(order) - 1):
                dist[order[k]] += (objs[order[k + 1], m]
                                   - objs[order[k - 1], m]) / span
    return dist


def sparsea_agds_search(fitness_fn, J, pop_size=60, max_gen=60, seed=42,
                        min_active=2, pc0=0.9, pm0=0.15, verbose=True,
                        log_every=10):
    """Run the SparseEA-AGDS-style masked search. Single run, no restarts."""
    rng = np.random.default_rng(seed)
    J = int(J)

    # --- initialisation: sparse-biased Bernoulli masks (feasibility first) ---
    pop = np.zeros((pop_size, J), dtype=int)
    for i in range(pop_size):
        p_in = float(rng.uniform(0.1, 0.4))
        mask = (rng.random(J) < p_in).astype(int)
        while mask.sum() < min_active:  # repair: enforce inclusion floor
            mask[rng.integers(0, J)] = 1
        pop[i] = mask

    def _eval(population):
        fit = np.array([float(fitness_fn(m)) for m in population])
        size = population.sum(axis=1).astype(float)
        return fit, np.column_stack([fit, size])

    fit, objs = _eval(pop)
    layer = _dominance_layers(objs)

    # --- state: per-individual operator rates + per-variable scores ---
    pc = np.full(pop_size, pc0)
    pm = np.full(pop_size, pm0)
    scores = np.zeros(J)  # dynamic variable scores, weighted accumulation

    def _repair(mask):
        while mask.sum() < min_active:
            # prefer switching on a high-score variable
            w = scores + 1e-9
            w = w / w.sum()
            mask[rng.choice(J, p=w)] = 1
        return mask

    best_ever_mask = pop[int(np.argmin(fit))].copy()
    best_ever_fit = float(fit.min())
    history_best = [best_ever_fit]
    # Population traces for convergence plots (best current + full traces):
    # per-generation best of the *current* population, mean/median/min/max,
    # plus the full per-individual fitness vector (downstream plots render
    # faint individual traces / bands from these).
    history_gen_best = [float(fit.min())]
    history_mean = [float(fit.mean())]
    history_median = [float(np.median(fit))]
    history_min = [float(fit.min())]
    history_max = [float(fit.max())]
    history_pop = [np.asarray(fit, dtype=float).copy()]
    gens_no_improve = 0

    for g in range(1, max_gen + 1):
        # stagnation-triggered local-search mode: when the best has not
        # improved for a while, shrink step sizes so offspring are 1-2 bit
        # neighbours (hill-climbing) instead of destructive macro-mutations.
        local_mode = gens_no_improve >= 10
        base_pm = 0.03 if local_mode else pm0
        prev_layer = layer.copy()
        offspring = np.zeros_like(pop)
        size = pop.sum(axis=1)
        for i in range(pop_size):
            # binary tournament on (penalised BIC, model size)
            cand = rng.choice(pop_size, size=2, replace=False)
            a, b = cand
            if (fit[a], size[a]) <= (fit[b], size[b]):
                p1 = pop[a].copy()
            else:
                p1 = pop[b].copy()
            cand = rng.choice(pop_size, size=2, replace=False)
            a, b = cand
            if (fit[a], size[a]) <= (fit[b], size[b]):
                p2 = pop[a].copy()
            else:
                p2 = pop[b].copy()

            # score-biased uniform crossover at the individual's own rate
            # (plain 0.5 crossover while scores are still flat)
            child = p1.copy()
            if rng.random() < pc[i]:
                if scores.max() - scores.min() > 1e-12:
                    w = scores - scores.min()
                    bias = 0.5 + 0.5 * (w / w.max())
                    swap = rng.random(J) < bias
                else:
                    swap = rng.random(J) < 0.5
                child[swap] = p2[swap]
            # directed score-biased mutation at the individual's own rate:
            # switch OFF low-score selected vars, switch ON high-score
            # unselected vars (superior variables gain extra opportunities,
            # inferior ones are pruned -> sparsity). Falls back to uniform
            # while scores are still flat (early generations).
            s = scores - scores.min()
            s = s / s.max() if s.max() > 0 else np.full(J, 0.5)
            rate = base_pm * (pm[i] / pm0)     # individual adaptivity x mode
            drop_p = rate * 2.0 * (1.0 - s)   # high for low-score vars
            add_p = rate * 2.0 * s            # high for high-score vars
            flip = np.zeros(J, dtype=bool)
            sel = child == 1
            flip[sel] = rng.random(int(sel.sum())) < np.clip(drop_p[sel], 0, 1)
            flip[~sel] = rng.random(int((~sel).sum())) < np.clip(add_p[~sel], 0, 1)
            child[flip] = 1 - child[flip]
            offspring[i] = _repair(child)

        fit_o, objs_o = _eval(offspring)
        # (mu + lambda) environmental selection on the joint pool.
        # Single objective: lexicographic (penalised BIC, model size), so
        # selection pressure stays on fit with a sparsity tiebreak. The
        # dominance layers below are retained for the adaptive operator
        # and the dynamic scoring (their original role), not for truncation.
        pool = np.vstack([pop, offspring])
        fit_p = np.concatenate([fit, fit_o])
        objs_p = np.vstack([objs, objs_o])
        layer_p = _dominance_layers(objs_p)
        size_p = objs_p[:, 1]
        order = sorted(range(len(pool)),
                       key=lambda k: (fit_p[k], size_p[k]))
        keep = np.array(order[:pop_size])
        pop, fit, objs = pool[keep], fit_p[keep], objs_p[keep]
        layer = layer_p[keep]

        # --- adaptive genetic operator (Algorithm 1, single-objective form):
        # layer improved -> widen search (higher pc/pm); worsened or flat ->
        # narrow it. Combined with the stagnation-triggered local mode above,
        # this cools an converged population into hill-climbing step sizes.
        maxL = max(int(layer.max()), int(prev_layer.max()), 1)
        for i in range(pop_size):
            # match offspring to nearest parent by Hamming distance for delta
            d = np.abs(pool - pop[i]).sum(axis=1)
            ancestor = int(np.argmin(d))
            old_L = int(layer_p[ancestor]) if ancestor < len(layer_p) else maxL
            delta = (old_L - int(layer[i])) / maxL  # + => improved
            pc[i] = float(np.clip(pc[i] * (1.00 + 0.10 * delta), 0.5, 1.0))
            pm[i] = float(np.clip(pm[i] * (1.00 + 0.30 * delta), 0.5 * pm0, 2.0 * pm0))

        # --- dynamic scoring (Algorithm 2): weighted accumulation over the
        # layer changes of the individuals selecting each variable ---
        for j in range(J):
            users = np.where(pop[:, j] == 1)[0]
            if len(users) == 0:
                scores[j] *= 0.99
                continue
            contrib = 0.0
            for u in users:
                d = np.abs(pool - pop[u]).sum(axis=1)
                ancestor = int(np.argmin(d))
                old_L = int(layer_p[ancestor]) if ancestor < len(layer_p) else maxL
                contrib += (old_L - int(layer[u])) / maxL
            scores[j] = 0.9 * scores[j] + 0.1 * contrib / max(len(users), 1)

        if fit.min() < best_ever_fit - 1e-12:
            best_ever_fit = float(fit.min())
            best_ever_mask = pop[int(np.argmin(fit))].copy()
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        history_best.append(best_ever_fit)
        history_gen_best.append(float(fit.min()))
        history_mean.append(float(fit.mean()))
        history_median.append(float(np.median(fit)))
        history_min.append(float(fit.min()))
        history_max.append(float(fit.max()))
        history_pop.append(np.asarray(fit, dtype=float).copy())
        if verbose and (g % log_every == 0 or g == max_gen):
            print(f"  [AGDS gen {g:3d}/{max_gen}] best penalised BIC = "
                  f"{best_ever_fit:.2f} (size {int(best_ever_mask.sum())})",
                  flush=True)

    return {"best_mask": best_ever_mask.astype(int),
            "best_fitness": float(best_ever_fit),
            "best_size": int(best_ever_mask.sum()),
            "history_best": [float(v) for v in history_best],
            "history_gen_best": [float(v) for v in history_gen_best],
            "history_mean": [float(v) for v in history_mean],
            "history_median": [float(v) for v in history_median],
            "history_min": [float(v) for v in history_min],
            "history_max": [float(v) for v in history_max],
            "history_pop": [np.asarray(v, dtype=float).copy()
                            for v in history_pop],
            "scores": scores.copy()}
