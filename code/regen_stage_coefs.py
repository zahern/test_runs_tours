#!/usr/bin/env python3
"""Regenerate stage coefficient CSVs in a GA run dir from the saved model pkls
(and the stage0 CDAP HTML report, whose Larch models can't be unpickled in the
current env). Uses the FIXED writers so _inference_pf / std_err_export_ SEs are
picked up instead of the bare pf NaNs the old runs wrote.

Usage: python regenerate_stage_coefs.py <run_dir> [run_dir ...]
"""
import os
import re
import sys
import pickle
import shutil

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)


def regen_stage1_4(run_dir: str) -> None:
    from ga_hypothesis_testing import _ga_save_stage_coef_csv
    for sid in (1, 2, 3, 4):
        stage_dir = os.path.join(run_dir, f'stage{sid}')
        pkl = os.path.join(stage_dir, f'stage{sid}_models.pkl')
        if not os.path.exists(pkl):
            print(f"  skip stage{sid}: no {pkl}")
            continue
        with open(pkl, 'rb') as f:
            models = pickle.load(f)
        try:
            _ga_save_stage_coef_csv(models, f'stage{sid}', stage_dir)
            print(f"  stage{sid}: wrote stage{sid}_coefficients.csv "
                  f"({len(models)} segments) -> {stage_dir}")
        except Exception as e:
            print(f"  stage{sid}: FAILED ({e})")


def regen_stage0(run_dir: str) -> None:
    """Rebuild stage0_cdap_coefficients.csv from the HTML report, which was
    written from the enriched _inference_pf frame (has std_err / t-stat)."""
    stage_dir = os.path.join(run_dir, 'stage0')
    html_path = os.path.join(stage_dir, 'stage0_cdap_models.html')
    csv_path = os.path.join(stage_dir, 'stage0_cdap_coefficients.csv')
    if not os.path.exists(html_path):
        print(f"  stage0: no {html_path} to parse")
        return
    html = open(html_path, encoding='utf-8').read()

    # Two sections: MNL and Mixed Logit. Each table row:
    # <tr><td>PARAM</td><td style="text-align:right">VALUE</td><td ...>SE</td><td ...>T</td><td>stars</td></tr>
    rows = []
    # MNL table = first table after "MNL — Base Model"
    m_sections = html.split('<h2>')
    cur_seg = None
    for chunk in m_sections:
        head = chunk.split('</h2>')[0]
        if 'MNL' in head:
            cur_seg = 'mnl'
        elif 'Mixed Logit' in head:
            cur_seg = 'mxl'
        else:
            continue
        for m in re.finditer(
            r'<tr><td>([^<]+)</td><td[^>]*>([^<]+)</td>'
            r'<td[^>]*>([^<]+)</td><td[^>]*>([^<]+)</td>',
            chunk):
            param, value, se, t = m.groups()
            try:
                vf = float(value.replace(',', ''))
                sef = float(se.replace(',', ''))
                tf = float(t.replace(',', ''))
            except ValueError:
                continue
            rows.append({
                'stage': 'stage0_cdap', 'segment': cur_seg,
                'param': param, 'value': vf, 'std_err': sef,
                't_stat': tf, 'p_value': 0.0, 'source': 'html_report',
            })

    if not rows:
        print(f"  stage0: no rows parsed from {html_path}")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  stage0: rebuilt stage0_cdap_coefficients.csv "
          f"({len(df)} rows) -> {csv_path}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for run_dir in args:
        run_dir = os.path.abspath(run_dir)
        print(f"== {run_dir} ==")
        regen_stage0(run_dir)
        regen_stage1_4(run_dir)


if __name__ == '__main__':
    main()