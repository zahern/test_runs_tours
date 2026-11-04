import pickle, sys, os
for tag in ["paper_missing_20260929","qld_full40","qld_lccm3"]:
    p=f"runs/ga_parallel/{tag}/merged/ga_results.pkl"
    print("===",tag, os.path.getsize(p))
    try:
        m=pickle.load(open(p,"rb"))
        print(type(m))
        if hasattr(m,"results"): print("results keys",list(m.results.keys()))
        elif isinstance(m,dict): print("keys",list(m.keys())[:20])
    except Exception as e:
        print("ERR",type(e).__name__,e)
