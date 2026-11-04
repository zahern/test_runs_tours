import time, warnings
import numpy as np, pandas as pd
t0 = time.time()
print(f'py start', flush=True)

from SearchLibrium import MultinomialLogit
print(f'SL imported in {time.time()-t0:.1f}s', flush=True)

np.random.seed(42); N = 30
avail_feats = ['day_position','time_consumed','duration_min','time_remaining','vehrd']
long_rows = []
for obs in range(N):
    feat = np.random.rand(5); feat[1]*=600; feat[2]=feat[2]*470+10; feat[3]*=800
    for alt in range(6):
        long_rows.append(list(feat) + [obs, alt, int(alt==np.random.randint(6))])
long = pd.DataFrame(long_rows, columns=avail_feats+['case_id','alt_id','chosen'])
alts = np.arange(6)

mnl = MultinomialLogit()
mnl.setup(
    X=long[avail_feats].fillna(0).values.astype(float),
    y=long['chosen'].values.astype(int),
    varnames=avail_feats, alts=alts, isvars=avail_feats,
    ids=long['case_id'].values.astype(int),
    base_alt=alts[0],
)
print(f'setup done in {time.time()-t0:.1f}s Kf={mnl.Kf}', flush=True)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mnl.fit()
print(f'fit done loglik={getattr(mnl,"loglik",None)} in {time.time()-t0:.1f}s', flush=True)
