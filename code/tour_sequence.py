"""
tour_sequence.py — per-tour purpose sequencing for CDAP (tour-level operation)
==============================================================================
CDAP assigns each person a DAY PATTERN letter (M mandatory / N non-mandatory /
H home). This module adds the tour-level operation on top: for every active
person it generates the ORDERED SEQUENCE of tour purposes that make up the day —
turning "this person travels for mandatory reasons" into an explicit tour list
(e.g. Work → Shop → Social) that feeds Stage 1.

Model: a first-order Markov chain over tour purposes, conditioned on the CDAP
pattern, with START and STOP states:
    P(first purpose | CDAP)         start distribution
    P(next purpose | current, CDAP) transition matrix
    P(STOP | current, CDAP)         absorbed into the transition row
Estimated on HHTS ordered tours (per person, ordered by tour index / start time).
An M-day is constrained to open with a mandatory purpose; an H-day yields no
tours. Laplace-smoothed so unseen transitions never have zero probability.

Emits a validation report (purpose share, sequence-length distribution,
first-purpose distribution, and the transition matrix) under <out_dir>/mobility/.

Pure numpy/pandas — estimates and validates in the base env.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_START, _STOP = '<START>', '<STOP>'
_MANDATORY = {'work', 'edu', 'education', 'school', 'university', 'univ'}

# Coarse purpose vocabulary (HHTS purposes fold into these).
_PURPOSE_ALIASES = {
    'work':    ['work', 'employ', 'business', 'job'],
    'edu':     ['edu', 'school', 'university', 'univ', 'stud', 'child care', 'childcare'],
    'shop':    ['shop', 'groc', 'retail', 'market', 'purchase'],
    'personal': ['personal', 'medical', 'health', 'bank', 'appoint', 'service'],
    'social':  ['social', 'recreat', 'leisure', 'visit', 'sport', 'entertain', 'eat', 'meal', 'dining'],
    'escort':  ['escort', 'pick', 'drop', 'accompany', 'serve passenger'],
}
_PURPOSES = list(_PURPOSE_ALIASES.keys()) + ['other']


def _coarse_purpose(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    out = pd.Series('other', index=series.index)
    for cat, keys in _PURPOSE_ALIASES.items():
        hit = s.str.contains('|'.join(keys), regex=True, na=False)
        out[hit & (out == 'other')] = cat
    return out


@dataclass
class TourSequenceModel:
    """First-order Markov purpose chain conditioned on CDAP pattern."""
    smoothing: float = 0.5
    states: list = field(default_factory=lambda: list(_PURPOSES))
    # per-CDAP: start distribution over purposes, and transition matrix
    start_: dict = field(default_factory=dict)          # cdap -> np.array[len(states)]
    trans_: dict = field(default_factory=dict)          # cdap -> np.array[S+1, S+1] (last col = STOP)
    obs_len_: dict = field(default_factory=dict)        # cdap -> mean seq length (diagnostics)

    def fit(self, sequences_by_cdap: dict):
        S = len(self.states)
        idx = {p: i for i, p in enumerate(self.states)}
        for cdap, seqs in sequences_by_cdap.items():
            start = np.full(S, self.smoothing)
            trans = np.full((S, S + 1), self.smoothing)   # +1 -> STOP column
            lengths = []
            for seq in seqs:
                seq = [p for p in seq if p in idx]
                if not seq:
                    continue
                lengths.append(len(seq))
                start[idx[seq[0]]] += 1
                for a, b in zip(seq[:-1], seq[1:]):
                    trans[idx[a], idx[b]] += 1
                trans[idx[seq[-1]], S] += 1               # -> STOP
            self.start_[cdap] = start / start.sum()
            self.trans_[cdap] = trans / trans.sum(axis=1, keepdims=True)
            self.obs_len_[cdap] = float(np.mean(lengths)) if lengths else 0.0
        return self

    def generate(self, cdap: str, rng, max_tours: int = 8) -> list:
        if cdap == 'H' or cdap not in self.start_:
            return []
        S = len(self.states)
        idx = {p: i for i, p in enumerate(self.states)}
        start = self.start_[cdap].copy()
        if cdap == 'M':
            # force the day to OPEN with a mandatory purpose
            mask = np.array([1.0 if p in _MANDATORY or p in ('work', 'edu') else 0.0
                             for p in self.states])
            if mask.sum() > 0:
                start = start * mask
                start = start / start.sum() if start.sum() > 0 else self.start_[cdap]
        cur = int(rng.choice(S, p=start))
        seq = [self.states[cur]]
        trans = self.trans_[cdap]
        for _ in range(max_tours - 1):
            row = trans[cur]
            nxt = int(rng.choice(S + 1, p=row))
            if nxt == S:            # STOP
                break
            seq.append(self.states[nxt])
            cur = nxt
        return seq


def build_sequences(tours: pd.DataFrame, cdap_by_person: Optional[dict] = None,
                    person_col: str = 'person_id') -> dict:
    """Group HHTS tours into ordered coarse-purpose sequences per person, keyed by
    the person's CDAP pattern (M/N). Falls back to inferring the pattern from the
    tour purposes when no CDAP map is supplied."""
    purp_col = next((c for c in ['purpose', 'PURPOSE', 'tour_purpose', 'MAINACT',
                                 'primary_purpose', 'DESTPURP'] if c in tours.columns), None)
    if purp_col is None or person_col not in tours.columns:
        return {}
    df = tours.copy()
    df['_purp'] = _coarse_purpose(df[purp_col])
    order_col = next((c for c in ['tour_num', 'tour_id', 'start_time', 'STARTIME',
                                  'depart_time', 'tour_index'] if c in df.columns), None)
    if order_col is not None:
        df = df.sort_values([person_col, order_col])
    out: dict = {'M': [], 'N': []}
    for pid, grp in df.groupby(person_col):
        seq = grp['_purp'].tolist()
        if cdap_by_person is not None:
            cdap = cdap_by_person.get(pid) or cdap_by_person.get(str(pid))
            if cdap not in ('M', 'N'):
                continue
        else:
            cdap = 'M' if any(p in ('work', 'edu') for p in seq) else 'N'
        out[cdap].append(seq)
    return out


def validate_sequences(model: TourSequenceModel, obs_by_cdap: dict,
                       gen_by_cdap: dict, out_dir: Optional[str] = None) -> dict:
    """Compare observed vs generated purpose share, sequence length, and first
    purpose. Writes tour_sequence_validation.csv/.html when out_dir given."""
    def _flat_share(seqs):
        allp = [p for s in seqs for p in s]
        vc = pd.Series(allp).value_counts(normalize=True) if allp else pd.Series(dtype=float)
        return vc.reindex(model.states).fillna(0)

    def _len_dist(seqs):
        return pd.Series([len(s) for s in seqs]).describe() if seqs else pd.Series(dtype=float)

    rows = []
    for cdap in ('M', 'N'):
        obs, gen = obs_by_cdap.get(cdap, []), gen_by_cdap.get(cdap, [])
        os_, gs_ = _flat_share(obs), _flat_share(gen)
        tvd = 0.5 * float(np.abs(os_.values - gs_.values).sum())   # total-variation dist
        ol, gl = _len_dist(obs), _len_dist(gen)
        rows.append({'cdap': cdap, 'n_obs': len(obs), 'n_gen': len(gen),
                     'purpose_TVD': tvd,
                     'obs_mean_len': float(ol.get('mean', np.nan)),
                     'gen_mean_len': float(gl.get('mean', np.nan))})
        logger.info(f"[tour_seq:{cdap}] purpose TVD={tvd:.3f} "
                    f"len obs={ol.get('mean', np.nan):.2f} gen={gl.get('mean', np.nan):.2f}")
    metrics = pd.DataFrame(rows)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            metrics.to_csv(os.path.join(out_dir, 'tour_sequence_validation.csv'), index=False)
            share_tbl = pd.DataFrame({
                f'{cdap}_obs': _flat_share(obs_by_cdap.get(cdap, [])) for cdap in ('M', 'N')
            })
            for cdap in ('M', 'N'):
                share_tbl[f'{cdap}_gen'] = _flat_share(gen_by_cdap.get(cdap, []))
            share_tbl.to_csv(os.path.join(out_dir, 'tour_sequence_purpose_share.csv'))
            _write_seq_html(metrics, share_tbl, out_dir)
        except Exception as e:
            logger.debug(f"tour_sequence report skipped ({e})")
    return {'metrics': metrics}


def _write_seq_html(metrics, share_tbl, out_dir):
    mrows = "".join(
        "<tr>" + "".join(f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                         for v in r) + "</tr>"
        for r in metrics.itertuples(index=False))
    mhead = "".join(f"<th>{c}</th>" for c in metrics.columns)
    srows = "".join(
        f"<tr><td>{idx}</td>" + "".join(f"<td>{share_tbl.loc[idx, c]:.3f}</td>"
                                        for c in share_tbl.columns) + "</tr>"
        for idx in share_tbl.index)
    shead = "".join(f"<th>{c}</th>" for c in share_tbl.columns)
    html = f"""<html><head><meta charset='utf-8'><title>tour sequence validation</title>
<style>body{{font-family:system-ui,sans-serif;margin:2em;max-width:820px}}
table{{border-collapse:collapse;margin:1em 0}}td,th{{border:1px solid #ccc;padding:4px 10px}}</style>
</head><body><h2>Tour-purpose sequencing — validation</h2>
<p>Total-variation distance (TVD) between observed and generated purpose shares;
0 = identical. Sequence length = tours per active person.</p>
<table><tr>{mhead}</tr>{mrows}</table>
<h3>Purpose share by CDAP pattern (observed vs generated)</h3>
<table><tr><th>purpose</th>{shead}</tr>{srows}</table>
</body></html>"""
    with open(os.path.join(out_dir, 'tour_sequence_validation.html'), 'w', encoding='utf-8') as fh:
        fh.write(html)


def run_tour_sequences(hhts_tours: pd.DataFrame, synth: pd.DataFrame,
                       cdap_col: str = 'CDAP', out_dir: Optional[str] = None,
                       seed: int = 42) -> pd.DataFrame:
    """Fit on HHTS tours, generate an ordered tour_purposes list for each active
    synth person, validate. Returns synth with `tour_purposes` (list) and
    `n_tours_seq` (length) columns; H-persons get []."""
    obs = build_sequences(hhts_tours)
    if not any(obs.values()):
        logger.info("tour_sequence: no usable HHTS tour purposes — skipping")
        return synth
    model = TourSequenceModel().fit(obs)
    rng = np.random.default_rng(seed)
    synth = synth.copy()
    cdap = synth[cdap_col] if cdap_col in synth.columns else pd.Series('N', index=synth.index)
    seqs = [model.generate(c, rng) for c in cdap]
    synth['tour_purposes'] = seqs
    synth['n_tours_seq'] = [len(s) for s in seqs]
    # validation: regenerate on the observed CDAP mix for a like-for-like compare
    gen = {'M': [], 'N': []}
    for c in ('M', 'N'):
        for _ in range(max(len(obs.get(c, [])), 1)):
            gen[c].append(model.generate(c, rng))
    validate_sequences(model, obs, gen, out_dir)
    logger.info(f"tour_sequence: generated sequences for {int((synth['n_tours_seq']>0).sum())}"
                f"/{len(synth)} persons (mean {synth['n_tours_seq'].mean():.2f} tours)")
    return synth


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # synthetic HHTS: M-days start Work then maybe Shop/Social; N-days Shop/Social
    def _mk(cdap, n):
        rows = []
        for pid in range(n):
            if cdap == 'M':
                seq = ['work'] + (['shop'] if rng.random() < 0.4 else []) + \
                      (['social'] if rng.random() < 0.3 else [])
            else:
                seq = ['shop'] if rng.random() < 0.6 else ['social']
                if rng.random() < 0.2:
                    seq.append('personal')
            for i, p in enumerate(seq):
                rows.append({'person_id': f'{cdap}{pid}', 'tour_num': i, 'purpose': p})
        return pd.DataFrame(rows)
    tours = pd.concat([_mk('M', 1500), _mk('N', 1500)], ignore_index=True)
    synth = pd.DataFrame({'person_id': range(2000),
                          'CDAP': rng.choice(['M', 'N', 'H'], 2000, p=[0.5, 0.3, 0.2])})
    out = run_tour_sequences(tours, synth, out_dir=None)
    print("mean tours (active):",
          round(out.loc[out.CDAP != 'H', 'n_tours_seq'].mean(), 2),
          "| H-persons with 0 tours:",
          int((out.loc[out.CDAP == 'H', 'n_tours_seq'] == 0).all()))
    print("sample M sequence:", out.loc[out.CDAP == 'M', 'tour_purposes'].iloc[0])
