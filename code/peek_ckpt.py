import os, pickle
os.chdir('/mnt/hpccs01/home/ahernz/test_runs_tours/code')
for ck in ['runs/ga_parallel/destfix/stage5/ga_results_checkpoint.pkl',
           'runs/ga_parallel/destfix/stage5/ga_results.pkl']:
    print('==', ck, os.path.exists(ck))
    if os.path.exists(ck):
        try:
            with open(ck, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, dict):
                print('  keys:', sorted(d.keys())[:30])
                for k, v in list(d.items())[:6]:
                    print(f'    {k}: {type(v).__name__}')
        except Exception as e:
            print('  ERR', repr(e))