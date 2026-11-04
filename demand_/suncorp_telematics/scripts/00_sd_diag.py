"""SD structural check from the small JSON only (fast)."""
import json

import numpy as np

RES = ("/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/"
       "suncorp_telematics/results")
aud = json.load(open(f"{RES}/game_day_holdout_metrics.json"))["audit"]
curves = aud.get("sd", {}).get("curves", {})
forms = aud.get("sd", {}).get("forms", {})

print("=" * 68)
print("1. FITTED Greenshields (vf km/h, kj) PER CLASS")
print("=" * 68)
for k, v in sorted(curves.items()):
    vf, kj = float(v[0]), float(v[1])
    ok = "OK " if 5 <= vf <= 140 else "BAD"
    qmax = vf * kj / 4.0 if np.isfinite(kj) else float("nan")
    print(f"  {k:16s} vf={vf:6.2f}  kj={kj:6.2f}  q_max={qmax:8.1f}  {ok}")
print("\n  theory: kj ~ 100-200 veh/km/lane; q_max ~ 1500-2200 veh/h/lane")

print("\n" + "=" * 68)
print("2. MONOTONICITY OF GP FORMS (x = density k, evaluated 0.05..3.0)")
print("=" * 68)
x = np.linspace(0.05, 3.0, 60)
env = {"x": x, "sq": lambda a: a * a, "sqrt": np.sqrt}
bad = []
for cls, f in sorted(forms.items()):
    sp, pr = f.get("speed"), f.get("prop")
    if sp is None:
        continue
    try:
        y = eval(sp, {"__builtins__": {}}, env)  # noqa: S307
        sp_down = bool((np.diff(y) <= 1e-9).all())
    except Exception as e:  # noqa: BLE001
        print(f"  {cls:16s} speed unevaluable: {e}")
        continue
    pr_dir = "n/a"
    if pr is not None:
        try:
            yp = eval(pr, {"__builtins__": {}}, env)  # noqa: S307
            pr_dir = "falls" if (np.diff(yp) <= 1e-9).all() else "RISES(!)"
        except Exception:  # noqa: BLE001
            pr_dir = "?"
    tags = []
    if not sp_down:
        tags.append("v RISES with k")
    if pr_dir == "RISES(!)":
        tags.append("prop RISES with k")
    if tags:
        bad.append(cls)
    print(f"  {cls:16s} v:{'falls' if sp_down else 'RISES'}  "
          f"prop:{pr_dir}{'   <== ' + ', '.join(tags) if tags else ''}")
print(f"\n  non-monotone classes: {len(bad)} {bad}")

print("\n" + "=" * 68)
print("3. SD vs REGRESSION LIFT (why corr was ~0.04)")
print("=" * 68)
c = aud.get("sd_lift_calibration", {})
print(f"  lift_obs = {c.get('a')} + {c.get('b')} * lift_sd_share "
      f"(se_b={c.get('se_b')} p={c.get('p_b')} n={c.get('n_fixtures')})")
t = c.get("train_lift", {})
if t:
    obs = np.array([v["obs"] for v in t.values()])
    sdl = np.array([v["sd"] for v in t.values()])
    print(f"  observed lift share: mean {obs.mean():+.3f} sd {obs.std():.3f} "
          f"min {obs.min():+.3f} max {obs.max():+.3f}")
    print(f"  SD twin lift share:  mean {sdl.mean():+.3f} sd {sdl.std():.3f} "
          f"min {sdl.min():+.3f} max {sdl.max():+.3f}")
    r = np.corrcoef(obs, sdl)[0, 1]
    print(f"  corr(obs_lift, sd_lift) across train games = {r:+.3f}")
    print(f"  SD twin expl variation: sd ratio "
          f"{sdl.std()/max(obs.std(),1e-9):.2f}")
    print("\n  -> the twin's cross-game spread is the signal. Compare its sd")
    print("     to the observed sd above: if the twin's sd is much SMALLER,")
    print("     congestion never bites in the simulation (curves too flat /")
    print("     density never reaches the knee), and the regression on noise")
    print("     produces the huge negative slope b.")
