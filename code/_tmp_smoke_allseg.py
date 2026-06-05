
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_connect import Synth_Connect
from stage5_destination_choice import run_stage5

sc = Synth_Connect()
out = sc.process_survey(save=False)
land = sc.get_land()
synth_s4 = out['tours'].sample(n=min(180, len(out['tours'])), random_state=11).copy()
print(f"SMOKE_ALLSEG: trips={len(out['trips'])}, persons={len(out['persons'])}, s4={len(synth_s4)}")

plan, models = run_stage5(
    out['trips'],
    out['persons'],
    synth_s4,
    'downcasted_matrices_base.omx',
    land,
    tours=out['tours'],
    seed=11,
    save_path='runs/stage5_minimal_allseg_models.pkl',
    segments=['work', 'edu', 'nonmand'],
    verbose=False,
)

print('SMOKE_ALLSEG_OK')
print('SEGMENTS_ESTIMATED', list(models.keys()))
if 'DTAZ' in plan.columns:
    print('UNIQUE_DTAZ', int(plan['DTAZ'].nunique()))
if {'row_type', 'mode_name'}.issubset(plan.columns):
    trip = plan.loc[plan['row_type'] == 'trip']
    print('MODE_SHARE', trip['mode_name'].value_counts(normalize=True).round(3).to_dict())
