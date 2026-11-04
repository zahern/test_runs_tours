"""
parking_price_gp.py — Genetic-programming discovery of the parking price function.

BEAM background (LBNL-UCB-STI/beam,
  src/main/scala/beam/agentsim/infrastructure/parking/):
  - PricingModel.scala: only FlatFee(fee) or Block(fee, interval); cost =
    fee or (duration/interval)*fee. No zone/duration response.
  - ParkingMNL.scala: choice utility has ParkingTicketCost (-1.0) +
    WalkingEgressCost + EnrouteDetourCost + type preference.
  Local Stage 7 (stage7_parking_choice.py) mirrors this but the fee is a
  fixed heuristic (CBD $8 vs $3 / location_table avg) — no price function.

What this module adds (opt-in, estimation default untouched):
  price_garage = softplus( f(zone, duration) )  with f discovered by GP over
  terminals {empdens, popdens, act_dur_hr, capacity, cbd, peak} and ops
  {+,-,*,/, slog, sq}. On-street price is fixed 0 (free kerb), so the MNL
  choice probability responds to the *garage premium* f returns.

Fitness (user choice): revealed-choice log-likelihood on QTS-HHTS stops
  (4_QTS_STOPS.xlsx VEHPARKED): 'On the street' -> on-street,
  'Car park (off-street)' -> garage. 'Residential property' treated as
  garage-free (price 0) and excluded from fitness; NaN/'Select one' dropped.
  MNL: P(garage) = 1/(1+exp(-(asc + b_fee*(0-price) + b_walk*(w_g-w_s)))).
  b_fee/b_walk/asc are co-evolved constants (metaheuristic over both
  structure AND parameters in one loop).

Scope hook (user choice: Stage 6 + Stage 7):
  - Stage 7: use price_garage() as parking_fee for garage alts in
    _build_parking_ca (replaces avg_parking_cost heuristic).
  - Stage 6: price enters auto utility as c_parking * expected_price(dtaz).
    apply_gp_price_to_plan() below shows the call pattern; it is NOT wired
    into stage6_mode_choice.py — pass the returned column explicitly.

PIPELINE ISOLATION: import-safe, __main__-guarded, numpy/pandas only.
Reads raw stops + landuse; writes only saved_data/parking_gp/. No .pbs,
stage, or load_data() references this file.

Usage:
  python parking_price_gp.py --max-rows 8000 --gens 25 --pop 60 --seed 1
"""
import argparse
import os

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_ROOT = os.path.normpath(os.path.join(CODE_DIR, ".."))
OUT_DIR = os.path.join(CODE_DIR, "saved_data", "parking_gp")

TERMINALS = ["emp", "pop", "dur", "cap", "cbd", "peak"]
OPS = ["+", "-", "*", "/", "slog", "sq"]

PMAX = 30.0  # $ cap (BEAM parseFee also rejects negatives; softplus handles it)


# ------------------------------------------------------------- expression
class Node:
    __slots__ = ("op", "kids", "feat", "const")

    def __init__(self, op=None, kids=None, feat=None, const=None):
        self.op = op
        self.kids = kids or []
        self.feat = feat
        self.const = const

    def depth(self):
        if self.op is None:
            return 0
        return 1 + max((k.depth() for k in self.kids), default=0)

    def size(self):
        if self.op is None:
            return 1
        return 1 + sum(k.size() for k in self.kids)

    def expr(self):
        if self.op is None:
            return self.feat if self.feat is not None else f"{self.const:.3f}"
        if self.op == "slog":
            return f"slog({self.kids[0].expr()})"
        if self.op == "sq":
            return f"sq({self.kids[0].expr()})"
        return f"({self.kids[0].expr()}{self.op}{self.kids[1].expr()})"


def rand_tree(rng, depth):
    if depth <= 0 or rng.random() < 0.35:
        if rng.random() < 0.7:
            return Node(feat=str(rng.choice(TERMINALS)))
        return Node(const=float(rng.normal(0, 1)))
    op = str(rng.choice(OPS))
    if op in ("slog", "sq"):
        return Node(op=op, kids=[rand_tree(rng, depth - 1)])
    return Node(op=op, kids=[rand_tree(rng, depth - 1), rand_tree(rng, depth - 1)])


def eval_tree(t, F):
    """F: dict terminal -> np array. Returns np array (raw f, pre-softplus)."""
    if t.op is None:
        if t.feat is not None:
            return F[t.feat]
        return np.full_like(next(iter(F.values())), t.const, dtype=float)
    if t.op == "+":
        return eval_tree(t.kids[0], F) + eval_tree(t.kids[1], F)
    if t.op == "-":
        return eval_tree(t.kids[0], F) - eval_tree(t.kids[1], F)
    if t.op == "*":
        return np.clip(eval_tree(t.kids[0], F) * eval_tree(t.kids[1], F), -1e6, 1e6)
    if t.op == "/":
        return eval_tree(t.kids[0], F) / (1.0 + np.abs(eval_tree(t.kids[1], F)))
    if t.op == "slog":
        return np.log1p(np.abs(eval_tree(t.kids[0], F)))
    if t.op == "sq":
        v = eval_tree(t.kids[0], F)
        return np.clip(v * v, 0, 1e6)
    raise ValueError(t.op)


def softplus(v):
    return np.log1p(np.exp(np.clip(v, -30, 30)))


def mutate(rng, t, depth=3):
    import copy
    t = copy.deepcopy(t)
    # node pickup
    nodes = []

    def walk(n):
        nodes.append(n)
        for k in n.kids:
            walk(k)
    walk(t)
    tgt = nodes[int(rng.integers(0, len(nodes)))]
    if tgt.op is None and rng.random() < 0.5:
        tgt.const = (tgt.const or 0.0) + float(rng.normal(0, 0.3))  # const creep
        if tgt.feat is not None and rng.random() < 0.3:
            tgt.feat = str(rng.choice(TERMINALS))
    else:
        sub = rand_tree(rng, depth)
        tgt.op, tgt.kids, tgt.feat, tgt.const = sub.op, sub.kids, sub.feat, sub.const
    return t


def crossover(rng, a, b):
    import copy
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    na, nb = [], []

    def walk(n, acc):
        acc.append(n)
        for k in n.kids:
            walk(k, acc)
    walk(a, na)
    walk(b, nb)
    ta, tb = na[int(rng.integers(0, len(na)))], nb[int(rng.integers(0, len(nb)))]
    ta.op, ta.kids, ta.feat, ta.const = tb.op, tb.kids, tb.feat, tb.const
    return a


# ------------------------------------------------------------------- data
def load_choice_data(max_rows=None):
    """Revealed parking-type choice from QTS stops + zone attrs from landuse."""
    st = pd.read_excel(os.path.join(TRT_ROOT, "4_QTS_STOPS.xlsx"),
                       engine="openpyxl",
                       usecols=["STOPID", "HHID", "PERSID", "MODE", "VEHPARKED",
                                "DESTPURP", "DESTSA1_2021", "DURATION", "STARTIME"])
    if max_rows:
        st = st.sample(n=min(max_rows, len(st)), random_state=1).reset_index(drop=True)
    # keep car-driver stops with observed parking location
    st = st[st["MODE"].isin(["Car driver", "Truck driver", "Motorcycle driver"])].copy()
    mp = {"On the street": 0, "Car park (off-street)": 1}
    st["y"] = st["VEHPARKED"].map(mp)
    st = st[st["y"].notna()].copy()
    st["y"] = st["y"].astype(int)
    # zone attrs
    lu = None
    for p in [os.path.join(CODE_DIR, "saved_data", "zone_landuse_overture.csv"),
              os.path.join(CODE_DIR, "saved_data", "zone_order_and_landuse.csv")]:
        if os.path.exists(p):
            lu = pd.read_csv(p)
            break
    sa1 = pd.read_csv(os.path.join(TRT_ROOT, "SA1_SHP", "sa1_2021.csv"))
    s12 = sa1.set_index("SA1_CODE21")["SA2_CODE21"].astype(int).to_dict()
    st["sa2"] = st["DESTSA1_2021"].map(s12)
    emp = pop = cap = {}
    if lu is not None:
        key = "SA2_MAIN16" if "SA2_MAIN16" in lu.columns else ("ORDER" if "ORDER" in lu.columns else None)
        if key:
            lu = lu.set_index(key)
            emp = lu.get("TOTEMP", pd.Series(dtype=float)).to_dict()
            pop = lu.get("TOTPOP", pd.Series(dtype=float)).to_dict()
            cap = lu.get("parking_capacity", pd.Series(dtype=float)).to_dict()
    st["emp"] = np.log1p(pd.to_numeric(st["sa2"].map(emp), errors="coerce").fillna(1000))
    st["pop"] = np.log1p(pd.to_numeric(st["sa2"].map(pop), errors="coerce").fillna(2000))
    st["cap"] = np.log1p(pd.to_numeric(st["sa2"].map(cap), errors="coerce").fillna(200))
    st["dur"] = (pd.to_numeric(st["DURATION"], errors="coerce").fillna(60) / 60.0).clip(0.1, 12)
    st["cbd"] = st["sa2"].isin([101021007, 101021008]).astype(float)  # Brisbane CBD SA2s (approx)
    hh = pd.to_numeric(st["STARTIME"], errors="coerce").fillna(600)
    st["peak"] = (((hh >= 420) & (hh <= 540)) | ((hh >= 960) & (hh <= 1080))).astype(float)
    return st.reset_index(drop=True)


def loglike(tree, theta, df):
    """MNL LL for garage(1) vs on-street(0); theta=(asc, b_fee, b_walk)."""
    asc, b_fee, b_walk = theta
    F = {k: df[k].values.astype(float) for k in TERMINALS}
    price = np.clip(softplus(eval_tree(tree, F)), 0, PMAX)  # garage premium $
    # walk-egg differential: garage parks closer on average in dense zones
    w_diff = -0.2 * np.tanh((df["emp"].values - df["emp"].values.mean()))
    v = asc + b_fee * (0.0 - price) + b_walk * w_diff
    p = 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = df["y"].values
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # parsimony penalty (BIC-like on tree size) — functional form stays readable
    ll -= 0.5 * np.log(len(df)) * tree.size() * 0.1
    return ll, price


# --------------------------------------------------------------------- GP
def run_gp(df, gens=25, pop=60, seed=1):
    rng = np.random.default_rng(seed)
    pop_trees = [rand_tree(rng, 3) for _ in range(pop)]
    pop_theta = np.column_stack([rng.normal(0, 0.5, pop),   # asc
                                 -np.abs(rng.normal(0.3, 0.2, pop)),  # b_fee<0
                                 -np.abs(rng.normal(0.3, 0.2, pop))]).tolist()  # b_walk<0
    best = (-1e18, None, None)
    for g in range(gens):
        scored = []
        for t, th in zip(pop_trees, pop_theta):
            try:
                ll, _ = loglike(t, th, df)
            except Exception:
                ll = -1e18
            if not np.isfinite(ll):
                ll = -1e18
            scored.append(ll)
            if ll > best[0]:
                best = (ll, t, list(th))
        order = np.argsort(scored)[::-1]
        print(f"[gp] gen {g + 1}/{gens} best LL={scored[order[0]]:.1f} "
              f"expr={pop_trees[order[0]].expr()} theta={np.round(pop_theta[order[0]], 3).tolist()}")
        elite = [pop_trees[i] for i in order[:6]]
        elite_th = [pop_theta[i] for i in order[:6]]
        nxt_t, nxt_th = list(elite), [list(t) for t in elite_th]
        while len(nxt_t) < pop:
            i, j = rng.choice(len(elite), 2, replace=False)
            kid = crossover(rng, elite[i], elite[j]) if rng.random() < 0.6 else mutate(rng, elite[i])
            if kid.depth() > 5:  # depth cap keeps price function interpretable
                kid = elite[i]
            th = list(elite_th[i])
            th = [v + float(rng.normal(0, 0.05)) for v in th]
            th[1], th[2] = -abs(th[1]), -abs(th[2])  # keep cost sensitivities negative
            nxt_t.append(kid)
            nxt_th.append(th)
        pop_trees, pop_theta = nxt_t, nxt_th
    return best


# --------------------------------------------- Stage 6 + Stage 7 plug-in
def price_garage(tree, emp, pop_, dur, cap=200.0, cbd=0.0, peak=0.0):
    """Evaluate the discovered price function on scalar/array inputs ($)."""
    F = {"emp": np.asarray(np.log1p(np.maximum(emp, 0)), float),
         "pop": np.asarray(np.log1p(np.maximum(pop_, 0)), float),
         "dur": np.asarray(np.maximum(dur, 0.1), float),
         "cap": np.asarray(np.log1p(np.maximum(cap, 0)), float),
         "cbd": np.asarray(cbd, float), "peak": np.asarray(peak, float)}
    return np.clip(softplus(eval_tree(tree, F)), 0, PMAX)


def apply_gp_price_to_plan(plan, landuse, tree, price_col="gp_park_price"):
    """Opt-in helper: attach expected garage price to each auto trip row.

    Stage 7: pass as parking_fee for garage alts (replaces avg_parking_cost).
    Stage 6: add `c_parking * <price_col>` to the DA/SR utility (parking cost
    sensitivity estimated alongside mode constants on real HHTS).
    NOT called by any pipeline code — wire explicitly in experiment scripts.
    """
    plan = plan.copy()
    key = "ORDER" if "ORDER" in landuse.columns else "zone_id"
    lu = landuse.set_index(key)
    emp = plan["DTAZ"].map(lu.get("TOTEMP", pd.Series(dtype=float))).fillna(1000)
    pop_ = plan["DTAZ"].map(lu.get("TOTPOP", pd.Series(dtype=float))).fillna(2000)
    cap = plan["DTAZ"].map(lu.get("parking_capacity", pd.Series(dtype=float))).fillna(200)
    dur = plan.get("predicted_duration_min", 60)
    dur = (pd.to_numeric(dur, errors="coerce").fillna(60) / 60.0).clip(0.1, 12)
    plan[price_col] = price_garage(tree, emp.values, pop_.values, dur.values, cap.values)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    df = load_choice_data(a.max_rows)
    print(f"choice data: {len(df)} car stops | garage share={df.y.mean():.3f}")
    ll, tree, theta = run_gp(df, a.gens, a.pop, a.seed)
    print(f"\nBEST LL={ll:.1f}\n  price_garage = softplus{tree.expr()}\n  asc,b_fee,b_walk={np.round(theta, 4).tolist()}")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "best_price_function.txt"), "w") as f:
        f.write(f"LL={ll:.1f}\nprice_garage = softplus({tree.expr()})\n"
                f"asc,b_fee,b_walk={theta}\n"
                f"(BEAM PricingModel analogue: FlatFee/Block with fee=f(zone,duration))\n")
    print(f"wrote {OUT_DIR}/best_price_function.txt (pipeline untouched)")


if __name__ == "__main__":
    main()
