import json
base = r'z:\test_runs_tours\code\runs\ga__21951661.aqua\stage_reports'
for s in range(1, 6):
    d = json.load(open(f'{base}/ga_stage{s}.json'))
    bic = d.get('best_BIC_per_obs', '?')
    ngen = d.get('n_generations', '?')
    neval = d.get('n_candidates_evaluated', '?')
    nfeat = d.get('n_selected_features', '?')
    note = str(d.get('best_fitness_note', ''))[:80]
    print(f'Stage {s}: BIC/obs={bic} n_gen={ngen} n_eval={neval} n_feat={nfeat}')
    if note:
        print(f'  note: {note}')
