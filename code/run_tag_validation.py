#!/usr/bin/env python
"""Standalone HHTS validation for a run tag: observed-vs-predicted plots
for every stage + submodel.

Usage:
    RUN_TAG=ga371_20260912r2 python run_tag_validation.py

Loads fitted stage model pkls under runs/ga_parallel/<TAG>/, rebuilds the
HHTS frames via Synth_Connect().process_survey(save=False), and calls
validate_hhts.run_validation. Plots land in <TAG>/plots/validation/.
Missing model pkls are tolerated (that stage is skipped by run_validation).
"""
import os
import pickle

from data_connect import Synth_Connect
from validate_hhts import run_validation

TAG = os.environ.get('RUN_TAG', 'ga371_20260912r2')
BASE = os.path.join('runs', 'ga_parallel', TAG)


def _load(*parts):
    p = os.path.join(BASE, *parts)
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    print(f'  [validate] missing {p} -> stage will be skipped', flush=True)
    return {}


def main():
    print('building HHTS frames via process_survey ...', flush=True)
    out = Synth_Connect().process_survey(save=False)
    print('loading stage models ...', flush=True)
    s1 = _load('stage1', 'stage1_models.pkl')
    s2 = _load('stage2', 'stage2_models.pkl')
    s3 = _load('stage3', 'stage3_models.pkl')
    s4 = _load('stage4', 'stage4_models.pkl')
    s5 = _load('stage5', 'stage5_models.pkl')
    print('running validation ...', flush=True)
    run_validation(out=out, s1_models=s1, s2_models=s2, s3_models=s3,
                   s4_models=s4, s5_models=s5, out_dir=BASE)
    print('done. plots in', os.path.join(BASE, 'plots', 'validation'),
          flush=True)


if __name__ == '__main__':
    main()
