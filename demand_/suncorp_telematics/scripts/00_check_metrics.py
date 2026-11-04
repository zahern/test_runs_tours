import json

RES = ("/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/"
       "suncorp_telematics/results")
d = json.load(open(f"{RES}/game_day_holdout_metrics.json"))
a = d["audit"]
print("audit keys:", list(a.keys()))
ts = a.get("two_stage", {})
print("two_stage keys:", list(ts.keys()))
for k in ("pruned_propensity", "propensity"):
    pp = ts.get(k, {})
    print(f"{k}: {list(pp.keys())[:12]} n_terms={len(pp.get('terms') or {})}")
