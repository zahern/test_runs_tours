"""Build the QLD model appendix for the preprint.

Reads per-shard GA artifacts under runs/ga_parallel/<tag> and writes
  abm_safety/preprint/appendix_models.tex
plus per-segment convergence plots under
  abm_safety/preprint/figures/appendix/conv_s<stage>_<segment>.png

Per segment: best-feature table (ga_reports/ga_stageN_best_features.csv),
fit stats (ga_reports/ga_summary.csv + shard_info.json), its own
convergence plot (plots/ga_stageN.png preferred, else plots/ga_all_stages.png),
and -- where the run artifacts carry them -- an estimated-coefficient table
(stageN_coefficients.csv for stages 1-3, stage5_models_coefficients.csv for
stage 5; top rows by |t|, capped). Stages 0/4 carry no coefficient CSVs in
the run artifacts, so those paragraphs keep feature tables only.
Segments missing from the primary tag fall back to --fallback tags in order;
each paragraph is annotated with its source run tag.

Usage:
  python build_appendix_models.py [--tag qld_shard4_retry_20260924]
                                  [--fallback qld_shard2,qld_shard1]
"""
import csv
import json
import math
import os
import shutil
import sys

STAGE_NAMES = {
    0: 'CDAP (coordinated daily activity pattern)',
    1: 'Activity frequency',
    2: 'Day budget',
    3: 'Activity duration',
    4: 'Schedule placement',
    5: 'Tour mode and destination',
}

EXPECTED = {
    0: ['all'],
    1: ['work', 'edu', 'nonmand'],
    2: ['start', 'end', 'budget'],
    3: ['work', 'edu', 'nonmand'],
    4: ['work', 'edu', 'escort', 'shopping', 'othmaint', 'othdiscr',
         'eat', 'social'],
    5: ['work', 'edu', 'nonmand'],
}


def _esc(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_') \
        .replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def _read_csv_rows(path):
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


def _completed_search_rows(rows):
    """Keep only rows produced by a completed GA search, not salvage output."""
    completed = []
    for row in rows:
        metric = str(row.get('fitness_metric', '')).strip().lower()
        try:
            generations = int(float(row.get('generations', '')))
            fitness = float(row.get('best_fitness', ''))
        except Exception:
            continue
        if (not metric.startswith('mean(-bic/n)') or generations < 2
            or not math.isfinite(fitness)):
            continue
        completed.append(row)
    return completed


def _fmt_num(x, nd=4):
    try:
        v = float(x)
    except Exception:
        return '---'
    if v != v:
        return '---'
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-4):
        return '%.2e' % v
    return ('%.{}f'.format(nd)) % v


def _coef_table(sdir, sid, seg):
    """Return LaTeX lines for an estimated-coefficient table, or [].

    Stages 1-3 schema: stage,segment,param,value,std_err,t_stat,p_value,source
    (segment looks like 'work/PartTimeWorker' -- match the seg prefix).
    Stage 5 schema: segment,model_type,parameter,value,std_err,tstat,pval,...
    Top COEF_MAX rows by |t|, non-holdfast, finite values.
    """
    COEF_MAX = 25
    cands = []
    if sid in (1, 2, 3):
        cands = [os.path.join(sdir, 'stage%d_coefficients.csv' % sid),
                 os.path.join(sdir, 'ga_reports', 'stage%d_coefficients.csv' % sid)]
    elif sid == 5:
        cands = [os.path.join(sdir, 'stage5_models_coefficients.csv')]
    rows = []
    for c in cands:
        rows = _read_csv_rows(c)
        if rows:
            break
    if not rows:
        return []
    cols = set(rows[0].keys())
    if {'param', 'value'}.issubset(cols):
        segc, paramc, tc, sec, pc = 'segment', 'param', 't_stat', 'std_err', 'p_value'
        holdc = None
    elif {'parameter', 'value'}.issubset(cols):
        segc, paramc, tc, sec, pc = 'segment', 'parameter', 'tstat', 'std_err', 'pval'
        holdc = 'holdfast' if 'holdfast' in cols else None
    else:
        return []
    keep = []
    NONMAND = {'eat', 'escort', 'shopping', 'othmaint', 'othdiscr', 'social',
               '_nonmand'}
    for r in rows:
        s = str(r.get(segc, ''))
        if sid in (1, 2, 3):
            head = s.split('/')[0]
            if not (s == seg or s.startswith(seg + '/')
                    or (seg == 'nonmand' and head in NONMAND)):
                continue
        else:
            if s != seg:
                continue
        if holdc and str(r.get(holdc, '')).strip().lower() in ('true', '1'):
            continue
        try:
            t = abs(float(r.get(tc, '')))
        except Exception:
            continue
        if not math.isfinite(t):
            continue
        try:
            v = float(r.get('value', ''))
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        keep.append((t, r))
    if not keep:
        return []
    keep.sort(key=lambda x: -x[0])
    keep = keep[:COEF_MAX]
    modelc = 'model_type' if 'model_type' in cols else None
    lines = ['\\begin{center}{\\small\\begin{tabular}{l l r r r r}\\toprule']
    lines.append('Param & Slot & Value & SE & $t$ & $p$ \\\\ \\midrule')
    for _, r in keep:
        slot = _esc(r.get(modelc, r.get(segc, ''))) if modelc else _esc(str(r.get(segc, '')))
        lines.append('%s & %s & %s & %s & %s & %s \\\\' % (
            _esc(str(r.get(paramc, ''))), slot,
            _fmt_num(r.get('value', '')), _fmt_num(r.get(sec, '')),
            _fmt_num(r.get(tc, ''), 2), _fmt_num(r.get(pc, ''))))
    lines.append('\\bottomrule\\end{tabular}}\\end{center}')
    lines.append('Top %d of %d estimated coefficients by $|t|$ (non-holdfast); '
                 'full table in the run artifacts.' % (len(keep), len(rows)))
    return lines


def main():
    tag = 'qld_shard4_retry_20260924'
    fallbacks = ['qld_shard2', 'qld_shard1']
    for a in sys.argv[1:]:
        if a.startswith('--tag='):
            tag = a.split('=', 1)[1]
        elif a.startswith('--fallback='):
            fallbacks = [t.strip() for t in a.split('=', 1)[1].split(',') if t.strip()]
        elif not a.startswith('-'):
            tag = a
    tags = [tag] + [t for t in fallbacks if t != tag]
    here = os.path.dirname(os.path.abspath(__file__))
    based = {t: os.path.join(here, 'runs', 'ga_parallel', t) for t in tags}
    # abm_safety/ is a sibling of code/, not a child.
    pre = os.path.abspath(os.path.join(here, '..', 'abm_safety', 'preprint'))
    figdir = os.path.join(pre, 'figures', 'appendix')
    os.makedirs(figdir, exist_ok=True)

    out = []
    out.append('% Auto-generated by build_appendix_models.py -- do not hand-edit.')
    out.append('%% Primary run: %s; fallback runs (per-paragraph labelled): %s.'
               % (tag, ', '.join(fallbacks) if fallbacks else 'none'))
    out.append('\\appendix')
    out.append('\\section{QLD estimated models (primary run ' + _esc(tag) + ')}')
    out.append('\\label{app:qld-models}')
    out.append('One subsection per estimation stage. Each segment table lists the '
               'searched best-feature set; fitness is mean $-\\mathrm{BIC}/n$ '
               '(higher is better) with the completed search-generation count. '
               'Each convergence panel is that segment\'s own BIC trajectory '
               '--- there is no pooled convergence plot. '
               'Estimated-coefficient tables (top rows by $|t|$) are included '
               'where the run artifacts carry them (stages 1--3 and 5); '
               'stages 0 and 4 report best-feature sets only. '
               'Segments absent from the primary run are taken from the fallback '
               'runs, labelled per paragraph.')
    n_fig = 0
    for sid in (0, 1, 2, 3, 4, 5):
        out.append('')
        if sid > 0:
            out.append('\\clearpage')
        out.append('\\subsection{Stage~%d --- %s}' % (sid, STAGE_NAMES[sid]))
        out.append('\\label{app:stage%d}' % sid)
        for seg in EXPECTED[sid]:
            sdir, used = None, None
            info, summ, feats = {}, [], []
            for t in tags:
                cand = os.path.join(based[t], 'stage%d' % sid, 'shards', seg)
                pkl = os.path.join(cand, 'ga_results.pkl')
                if not (os.path.isfile(pkl) and os.path.getsize(pkl) > 0):
                    continue
                f = _read_csv_rows(os.path.join(
                    cand, 'ga_reports', 'ga_stage%d_best_features.csv' % sid))
                if not f:
                    continue
                info = _read_json(os.path.join(cand, 'shard_info.json'))
                summ = _read_csv_rows(os.path.join(
                    cand, 'ga_reports', 'ga_stage%d_summary.csv' % sid))
                if not summ:
                    summ = _read_csv_rows(os.path.join(
                        cand, 'ga_reports', 'ga_summary.csv'))
                summ = _completed_search_rows(summ)
                if not summ:
                    continue
                sdir, used = cand, t
                feats = f
                break
            # usable = artifact exists AND (search finished with features)
            title = 'Stage~%d / %s' % (sid, _esc(seg))
            if sdir is None:
                out.append('')
                out.append('\\paragraph{%s (pending rerun).}' % title)
                out.append('No usable shard artifact in any of %s; '
                           're-estimation queued (see merge log).'
                           % _esc(', '.join(tags)))
                continue
            fit = info.get('best_fitness', None)
            el = info.get('elapsed_min', None)
            gens = ''
            if summ:
                row = summ[0]
                if fit is None:
                    try:
                        fit = float(row.get('best_fitness', ''))
                    except Exception:
                        fit = None
                try:
                    gens = str(int(float(row.get('generations', ''))))
                except Exception:
                    gens = row.get('generations', '')
            try:
                fit_s = '$%.4f$' % float(fit)
            except Exception:
                fit_s = '---'
            try:
                el_s = '%.0f\\,min' % float(el)
            except Exception:
                el_s = '---'
            out.append('')
            out.append('\\paragraph{%s.}' % title)
            out.append('Best fitness %s mean($-\\mathrm{BIC}/n$), %s completed search generations, '
                              'wall-clock %s. Source run: %s%s.'
                              % (fit_s, gens or '---', el_s, _esc(used),
                                  ' (fallback)' if used != tag else ''))
            # feature table grouped by slot
            groups = {}
            for r in feats:
                groups.setdefault(r.get('group', ''), []).append(
                    r.get('feature', ''))
            out.append('\\begin{center}\\begin{tabular}{l p{10cm}}\\toprule')
            out.append('Slot & Best features \\\\ \\midrule')
            for g in sorted(groups):
                out.append('%s & %s \\\\'
                           % (_esc(g), ', '.join(_esc(f) for f in groups[g])))
            out.append('\\bottomrule\\end{tabular}\\end{center}')
            # estimated-coefficient table (stages 1-3 and 5 only)
            for _cl in _coef_table(sdir, sid, seg):
                out.append(_cl)
            # convergence figure
            src = os.path.join(sdir, 'plots', 'ga_stage%d.png' % sid)
            if not os.path.isfile(src):
                src = os.path.join(sdir, 'plots', 'ga_all_stages.png')
            if os.path.isfile(src):
                dst = os.path.join(figdir, 'conv_s%d_%s.png' % (sid, seg))
                shutil.copy2(src, dst)
                out.append('\\begin{figure}[h!]\\centering')
                out.append('\\includegraphics[width=0.85\\linewidth]'
                           '{figures/appendix/conv_s%d_%s.png}' % (sid, seg))
                out.append('\\caption{Stage~%d / %s: searched BIC trajectory.}'
                           % (sid, _esc(seg)))
                out.append('\\label{fig:conv-s%d-%s}' % (sid, seg))
                out.append('\\end{figure}')
                n_fig += 1
    tex = os.path.join(pre, 'appendix_models.tex')
    with open(tex, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    n_coef = sum(1 for ln in out if ln.startswith('Top ') and 'coefficients by' in ln)
    print('wrote %s (%d convergence figures, %d coefficient tables; tags: %s)'
          % (tex, n_fig, n_coef, ', '.join(tags)))


if __name__ == '__main__':
    main()
