"""Diagnose v154 HB searched-SPF LL=-2.6M: refit the same 10-var spec fixed NB."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import fit_model

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "near_miss_hb_ecf.csv"), low_memory=False)
FEATS10 = ['AADT', 'IC', 'IP', 'IV_M', 'PA_COM1', 'PSDSG24', 'RSDSDL5',
           'RSDSO', 'SKD_SP', 'SL']
for c in FEATS10:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
for off in ["log_len"]:
    m = fit_model(df, FEATS10, "HB_Total", off, label=f"diag-{off}")
    print(f"offset={off}: LL={m['ll']:.1f} BIC={m['bic']:.1f} k={m['k']} n={m['n']} converged={m['converged']}", flush=True)
    print(m["coef_table"].to_string(index=False), flush=True)
