import pickle, os
for tag in ["qld_shard2","qld_shard1","qld_shard4_retry_20260924","qld_lccm"]:
    p=f"runs/ga_parallel/{tag}/merged/ga_results.pkl"
    print("===",tag, os.path.getsize(p))
    try:
        m=pickle.load(open(p,"rb"))
        print(type(m), "keys", list(m.results.keys()) if hasattr(m,"results") else list(m.keys())[:20])
    except Exception as e: print("ERR",e)
