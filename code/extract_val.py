p = '/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/ga_metacount_mv_staged__24967559.aqua/pipeline_run.log'
lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
# find the usual-location validation section (after "Usual location assignment complete.")
start = None
for i, ln in enumerate(lines):
    if 'Usual location assignment complete' in ln:
        start = i
        break
if start is None:
    print("no marker found")
else:
    for ln in lines[start:start+60]:
        print(ln)
