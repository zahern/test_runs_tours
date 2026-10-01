import re
p = '/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/ga_metacount_mv_staged__24967559.aqua/pipeline_run.log'
lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
keys = re.compile(r'valid|usual|tours|overlap|school|work zone', re.I)
for i, ln in enumerate(lines):
    if keys.search(ln):
        print(f"{i:5d}: {ln}")
