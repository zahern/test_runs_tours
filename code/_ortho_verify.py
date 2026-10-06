import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, cloudpickle
import openmatrix as omx


def patch():
    from larch.dataset import flow as f
    def g(self, n):
        if n == "_obj": raise AttributeError(n)
        o = self.__dict__.get("_obj")
        if o is None: raise AttributeError(n)
        r = getattr(o, n)
        try: r = self.transfer_dimension_attrs(r)
        except Exception: pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(f, c, None)
        if k is not None: k.__getattr__ = g


patch()
import stage5_destination_choice as S5

cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
skims = omx.open_file(r"Z:\test_runs_tours\demand_\skims_371.omx", "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
dt = S5._dest_tree(
    cache["seg_acdc"], cache["persons"], cache["lu_with_st"], skims,
    cache["logsums"], sampled_alts=cache["sampled_alts"], skim_df=skim_df,
    crash_skims=skims, segment="work")
skims.close()
ob = dt.attrs.get("_ortho_betas", {})
print(f"_ortho_betas entries: {len(ob)}")
for coeff, info in sorted(ob.items()):
    col = info.get("col") if isinstance(info, dict) else info
    beta = info.get("beta") if isinstance(info, dict) else "?"
    present = col in dt.data_vars
    print(f"  {coeff:26s} -> {str(col):44s} beta={beta} in_dataset={present}")
