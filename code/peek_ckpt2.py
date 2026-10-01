import pickle
for ck in ["runs/ga_parallel/destfix/stage5/ga_results_checkpoint.pkl",
           "runs/ga_parallel/destfix/stage5/ga_results.pkl"]:
    with open(ck, "rb") as f:
        d = pickle.load(f)
    print(ck, "->", type(d))
    if isinstance(d, dict):
        for k, v in list(d.items())[:15]:
            extra = list(v.keys())[:8] if isinstance(v, dict) else ""
            print("   ", k, type(v).__name__, extra)