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
import re
import shutil
import sys

# Shared reporting categories + always-show patterns (single source of truth in
# ga_constraints, also used to set larch's model.ordering at build time).
try:
    from ga_constraints import (stage5_variable_groups as _s5_groups,
                                 STAGE5_REPORT_ALWAYS_SHOW as _ALWAYS_SHOW)
except Exception:
    def _s5_groups(kind='mode'):
        return (("Safety", r".*crash.*", r".*CRASH.*", r".*safety.*"),
                ("Other", r".*"))
    _ALWAYS_SHOW = (r".*crash.*", r".*CRASH.*", r".*safety.*", r".*logsum.*",
                    r"c_size_term", r"log_correction")


def _matches_any(name, patterns):
    s = str(name)
    for p in patterns:
        try:
            if re.fullmatch(p, s) or re.search(p, s):
                return True
        except Exception:
            continue
    return False


def _group_of(name, groups):
    for cat in groups:
        if _matches_any(name, cat[1:]):
            return cat[0]
    return "Other"

STAGE_NAMES = {
    0: 'CDAP (coordinated daily activity pattern)',
    1: 'Activity frequency',
    2: 'Day budget',
    3: 'Activity duration',
    4: 'Schedule placement',
    5: 'Tour mode and destination',
}

# Stage 5 is reported from the headline merged run.  Its coefficients are the
# ones the main text cites in Table ga_shard2; the primary tag's stage-5 refit
# is a different fit with different likelihoods, so mixing the two made the
# appendix and the main text disagree on the crash coefficients.
S5_TAG = os.environ.get('APPENDIX_S5_TAG', 'qld_shard4_retry_20260924')

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

    Rows are grouped into reporting categories (Structural / LOS / Safety /
    Land-use / ...) exactly as larch's ``model.ordering`` would. Force-included
    structural + safety terms (crash / safety / logsum / size) are ALWAYS shown
    -- even when held at a bound or below the top-|t| cap -- so the CRASH terms
    never silently vanish; held rows are flagged. Everything else is the top
    rows by |t|.
    """
    COEF_MAX = int(os.environ.get('APPENDIX_COEF_MAX', '40'))
    cands = []
    if sid in (1, 2, 3):
        cands = [os.path.join(sdir, 'stage%d_coefficients.csv' % sid),
                 os.path.join(sdir, 'ga_reports', 'stage%d_coefficients.csv' % sid)]
    elif sid == 5:
        # Prefer the full fitted table; fall back to the prefit table (some
        # runs carry only a Stage-5 prefit) so the crash terms still surface.
        cands = [os.path.join(sdir, 'stage5_models_coefficients.csv'),
                 os.path.join(sdir, 'stage5_models_prefit_coefficients.csv')]
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
        holdc = 'holdfast' if 'holdfast' in cols else None
    elif {'parameter', 'value'}.issubset(cols):
        segc, paramc, tc, sec, pc = 'segment', 'parameter', 'tstat', 'std_err', 'pval'
        holdc = 'holdfast' if 'holdfast' in cols else None
    else:
        return []
    modelc = 'model_type' if 'model_type' in cols else None
    groupc = 'group' if 'group' in cols else None
    kind = 'dest'  # grouping; mode/dest decided per-row below when model_type present
    NONMAND = {'eat', 'escort', 'shopping', 'othmaint', 'othdiscr', 'social',
               '_nonmand'}
    # Collect this segment's rows (value finite; keep held rows).
    seg_rows = []
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
        try:
            v = float(r.get('value', ''))
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        try:
            t = abs(float(r.get(tc, '')))
        except Exception:
            t = float('nan')
        held = bool(holdc and str(r.get(holdc, '')).strip().lower()
                    in ('true', '1'))
        seg_rows.append({'r': r, 't': t, 'held': held,
                         'name': str(r.get(paramc, ''))})
    if not seg_rows:
        return []
    # Collapse duplicate exports of the same parameter in the same slot.  Some
    # stage 2/3 CSVs contain the pre-prune fit and a later refit under the same
    # segment key, which printed the same coefficient twice and pushed real
    # terms out of the |t| cap.  Later rows supersede earlier ones.
    dedup = {}
    for x in seg_rows:
        key = (str(x['r'].get(modelc, '')), str(x['r'].get(segc, '')), x['name'])
        dedup[key] = x
    seg_rows = list(dedup.values())
    # Select rows: always-show (crash/safety/logsum/size) + top-|t| of the rest.
    pinned = [x for x in seg_rows if _matches_any(x['name'], _ALWAYS_SHOW)]
    rest = [x for x in seg_rows
            if x not in pinned and not x['held'] and math.isfinite(x['t'])]
    rest.sort(key=lambda x: -x['t'])
    budget = max(0, COEF_MAX - len(pinned))
    chosen = pinned + rest[:budget]
    # Group chosen rows by reporting category.
    mt0 = str(chosen[0]['r'].get(modelc, '')) if (chosen and modelc) else ''
    groups = _s5_groups('mode' if mt0 == 'mode' else 'dest') if sid == 5 \
        else (("All", r".*"),)
    by_group = {}
    for x in chosen:
        g = (str(x['r'].get(groupc, '')) if groupc and x['r'].get(groupc)
             else _group_of(x['name'], groups))
        by_group.setdefault(g, []).append(x)
    # Preserve the category order from the group spec; unknowns last.
    order = [c[0] for c in groups] + [g for g in by_group if g not in
                                      {c[0] for c in groups}]
    lines = ['\\begin{center}{\\small\\begin{tabular}{l l r r r r}\\toprule']
    lines.append('Param & Slot & Value & SE & $t$ & $p$ \\\\ \\midrule')
    n_shown = 0
    for g in order:
        xs = by_group.get(g)
        if not xs:
            continue
        xs.sort(key=lambda x: (not _matches_any(x['name'], _ALWAYS_SHOW),
                               -(x['t'] if math.isfinite(x['t']) else -1)))
        lines.append('\\multicolumn{6}{l}{\\itshape %s} \\\\' % _esc(g))
        for x in xs:
            r = x['r']
            nm = _esc(x['name']) + (' \\textsuperscript{(held)}' if x['held'] else '')
            slot = _esc(str(r.get(modelc, r.get(segc, '')))) if modelc \
                else _esc(str(r.get(segc, '')))
            # Inference-consistency guard on read: some stage 1-3 CSVs carry a
            # t_stat/p_value misaligned with value/SE (e.g. t=957 for -13.43/0.52).
            # The Wald t is value/SE by definition, so recompute t and p whenever
            # SE is valid and the stored t disagrees. Keeps reused (not re-run)
            # stages correct in the paper without re-estimation.
            t_disp, p_disp = r.get(tc, ''), r.get(pc, '')
            try:
                _v = float(r.get('value', '')); _se = float(r.get(sec, ''))
                _t = float(t_disp)
                if math.isfinite(_v) and math.isfinite(_se) and _se > 0:
                    _impl = _v / _se
                    if (not math.isfinite(_t)
                            or abs(_impl - _t) > max(0.05, 0.01 * abs(_impl))):
                        from scipy.stats import norm as _n
                        t_disp = _impl
                        p_disp = float(2.0 * _n.sf(abs(_impl)))
            except (TypeError, ValueError):
                pass
            lines.append('%s & %s & %s & %s & %s & %s \\\\' % (
                nm, slot, _fmt_num(r.get('value', '')), _fmt_num(r.get(sec, '')),
                _fmt_num(t_disp, 2), _fmt_num(p_disp)))
            n_shown += 1
    lines.append('\\bottomrule\\end{tabular}}\\end{center}')
    # Convergence line from the per-model metadata when present.
    conv = ''
    _r0 = chosen[0]['r']
    _ll = _r0.get('loglike'); _ll0 = _r0.get('loglike_null')
    _rho = _r0.get('rho2'); _nit = _r0.get('n_iter'); _cv = _r0.get('converged')
    conv_bits = []
    if _ll not in (None, ''):
        conv_bits.append('LL $=%s$' % _fmt_num(_ll, 1))
    if _rho not in (None, ''):
        conv_bits.append(r'$\rho^2=%s$' % _fmt_num(_rho, 3))
    if _nit not in (None, ''):
        conv_bits.append('%s iterations' % _esc(str(_nit)))
    if _cv not in (None, ''):
        conv_bits.append('converged' if str(_cv).lower() in ('true', '1')
                         else 'NOT converged')
    if conv_bits:
        conv = ' Final fit: ' + ', '.join(conv_bits) + '.'
    lines.append('%d coefficients shown (force-included safety/structural terms '
                 'always listed, held rows flagged); full table in the run '
                 'artifacts.%s' % (n_shown, conv))
    return lines


def _safety_benefit_figure(tags, based, figdir):
    """Forest plot of the Stage-5 crash/safety coefficients across segments.

    Replaces the "plots don't show the benefit" gap: reads the force-included
    crash / safety terms (value +/- 1.96*SE) from each segment's Stage-5 coef
    CSV (full table, else prefit) and draws a coefficient forest so the safety
    effect and its sign are visible at a glance. Returns the saved path or None.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pat = (r".*crash.*", r".*CRASH.*", r".*safety.*")
    items = []  # (label, value, lo, hi, model_type)
    for seg in EXPECTED[5]:
        rows = []
        for t in tags:
            for sub in (os.path.join('stage5', 'promoted', 'refit', seg),
                        os.path.join('stage5', 'shards', seg)):
                cand = os.path.join(based[t], sub)
                for fn in ('stage5_models_coefficients.csv',
                           'stage5_models_prefit_coefficients.csv'):
                    rows = _read_csv_rows(os.path.join(cand, fn))
                    if rows:
                        break
                if rows:
                    break
            if rows:
                break
        for r in rows:
            nm = str(r.get('parameter', r.get('param', '')))
            if not _matches_any(nm, pat):
                continue
            try:
                v = float(r.get('value', ''))
                se = float(r.get('std_err', ''))
            except Exception:
                continue
            if not (math.isfinite(v) and math.isfinite(se)):
                continue
            mt = str(r.get('model_type', ''))
            items.append(('%s/%s: %s' % (seg, mt or '?', nm), v,
                          v - 1.96 * se, v + 1.96 * se, mt))
    if not items:
        return None
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(items))))
    ys = range(len(items))
    colors = {'mode': '#1f77b4', 'dest': '#d62728'}
    for y, (lab, v, lo, hi, mt) in zip(ys, items):
        c = colors.get(mt, '#555555')
        ax.plot([lo, hi], [y, y], '-', color=c, lw=1.6, alpha=0.8)
        ax.plot([v], [y], 'o', color=c, ms=5)
    ax.axvline(0.0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for lab, *_ in items], fontsize=7)
    ax.set_xlabel('Coefficient (95% CI); negative = safety/crash deters the choice')
    ax.set_title('Stage 5 safety / crash coefficients by segment')
    import matplotlib.lines as _ml
    ax.legend(handles=[_ml.Line2D([], [], color=colors['mode'], marker='o',
                                  ls='-', label='mode'),
                       _ml.Line2D([], [], color=colors['dest'], marker='o',
                                  ls='-', label='destination')],
              fontsize=8, loc='best')
    fig.tight_layout()
    dst = os.path.join(figdir, 'safety_benefit_forest.png')
    fig.savefig(dst, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dst


def main():
    tag = 'qld_conv1'
    fallbacks = ['qld_shard4_retry_20260924', 'qld_shard2', 'qld_shard1']
    for a in sys.argv[1:]:
        if a.startswith('--tag='):
            tag = a.split('=', 1)[1]
        elif a.startswith('--fallback='):
            fallbacks = [t.strip() for t in a.split('=', 1)[1].split(',') if t.strip()]
        elif not a.startswith('-'):
            tag = a
    tags = [tag] + [t for t in fallbacks if t != tag]
    s5_tags = [S5_TAG] + [t for t in tags if t != S5_TAG]
    here = os.path.dirname(os.path.abspath(__file__))
    based = {t: os.path.join(here, 'runs', 'ga_parallel', t)
             for t in dict.fromkeys(tags + s5_tags)}
    verdicts = _read_json(os.path.join(
        os.path.abspath(os.path.join(here, '..', 'abm_safety', 'preprint')),
        'convergence_verdicts.json')) or {}
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
               'Each panel is that segment\'s own trajectory --- there is no '
               'pooled convergence plot. The top panel shows best-so-far BIC/n, '
               'the population mean and the mean--worst spread; the bottom panel '
               'shows population diversity. The verdict printed above each '
               'panel is computed from the search\'s own stagnation counter: a '
               'run that ended at its generation cap is reported as a '
               'budget-limited plateau, not as converged, and a smoke-budget '
               'run is labelled as not a convergence result. Stage~5 carries no '
               'convergence panel: that search wrote a salvaged-partial history, '
               'so its tables are refits of the best-so-far genome, not a '
               'completed trajectory. '
               'Estimated-coefficient tables are grouped by model component '
               '(structural / level-of-service / safety / land-use / person '
               'interactions); the force-included safety and structural terms '
               '(crash, safety, logsum, size) are always listed --- even when '
               'held at a bound or below the top-$|t|$ cap --- and held rows '
               'are flagged, with the final-fit log-likelihood, $\\rho^2$, and '
               'iteration count reported beneath each table. Tables appear '
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
            # Stage 5 is handled by the coefficient block below so every
            # segment comes from the same headline run.
            for t in ([] if sid == 5 else tags):
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
            if sdir is None and sid == 5:
                # Stage-5 search pkls are frequently absent/FAILED even when the
                # fitted coefficient tables were written (promoted refit, else
                # the shard). Emit the coefficient table from those so the
                # estimated model -- including the force-included CRASH terms --
                # is reported instead of a "pending rerun" stub.
                for t in s5_tags:
                    for sub in (os.path.join('stage5', 'shards', seg),
                                os.path.join('stage5', 'promoted', 'refit', seg)):
                        cdir = os.path.join(based[t], sub)
                        if (os.path.isfile(os.path.join(
                                cdir, 'stage5_models_coefficients.csv'))
                            or os.path.isfile(os.path.join(
                                cdir, 'stage5_models_prefit_coefficients.csv'))):
                            sdir, used = cdir, t
                            break
                    if sdir is not None:
                        break
                if sdir is not None:
                    _cl = _coef_table(sdir, sid, seg)
                    if _cl:
                        out.append('')
                        out.append('\\paragraph{%s.}' % title)
                        out.append('Estimated coefficients from the final refit '
                                   'of the Stage-5 best-so-far genome (source '
                                   'run %s%s). The Stage-5 search wrote a '
                                   'salvaged-partial history, so no convergence '
                                   'panel is claimed for this segment.'
                                   % (_esc(used),
                                      ' (fallback)' if used != tag else ''))
                        _feat = _read_csv_rows(os.path.join(
                            sdir, 'ga_reports', 'ga_stage5_best_features.csv'))
                        if _feat:
                            _groups = {}
                            for _r in _feat:
                                _groups.setdefault(_r.get('group', ''), []).append(
                                    _r.get('feature', ''))
                            out.append('\\begin{center}\\begin{tabular}{l p{10cm}}'
                                       '\\toprule')
                            out.append('Slot & Best features \\\\ \\midrule')
                            for _g in sorted(_groups):
                                out.append('%s & %s \\\\'
                                           % (_esc(_g), ', '.join(
                                               _esc(_f) for _f in _groups[_g])))
                            out.append('\\bottomrule\\end{tabular}\\end{center}')
                        for _l in _cl:
                            out.append(_l)
                        continue
                sdir = None
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
            # A pooled non-mandatory search returns one feature set replicated
            # under every purpose group.  Six identical rows would imply six
            # purpose-specific searches that never happened; collapse them.
            _uniq = {tuple(v) for v in groups.values()}
            if sid in (1, 3) and seg == 'nonmand' and len(_uniq) == 1:
                groups = {'nonmand (combined)': next(iter(groups.values()))}
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
            # The panels are produced by build_convergence_panels.py from each
            # shard's own per-generation history, with the verdict copied from
            # convergence_verdicts.json.  They must never be overwritten by the
            # legacy plots/ga_stageN.png: that plot swaps the population mean
            # for a band of every proposed chromosome, so a converged run can
            # look like a search that never settles.  Only fall back to the
            # legacy plot when no rebuilt panel exists for this segment.
            dst = os.path.join(figdir, 'conv_s%d_%s.png' % (sid, seg))
            _v = (verdicts.get('s%d_%s' % (sid, seg), {}) or {}).get('verdict', '')
            rebuilt = bool(_v) and os.path.isfile(dst)
            if not rebuilt:
                src = os.path.join(sdir, 'plots', 'ga_stage%d.png' % sid)
                if not os.path.isfile(src):
                    src = os.path.join(sdir, 'plots', 'ga_all_stages.png')
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            if os.path.isfile(dst):
                out.append('\\begin{figure}[h!]\\centering')
                out.append('\\includegraphics[width=0.85\\linewidth]'
                           '{figures/appendix/conv_s%d_%s.png}' % (sid, seg))
                if rebuilt:
                    _cap = ('Stage~%d / %s: searched BIC trajectory (top: '
                            'best-so-far, population mean and mean--worst spread) '
                            'and population diversity (bottom).'
                            % (sid, _esc(seg)))
                    _cap += ' ' + _esc(_v) + '.'
                else:
                    _cap = ('Stage~%d / %s: legacy proposal-range plot '
                            '(best-so-far and the range of proposed '
                            'chromosomes; no population mean or verdict).'
                            % (sid, _esc(seg)))
                out.append('\\caption{%s}' % _cap)
                out.append('\\label{fig:conv-s%d-%s}' % (sid, seg))
                out.append('\\end{figure}')
                n_fig += 1
    # ── Safety-benefit forest (crash/safety coefficients across segments) ──
    out.append('')
    out.append('\\clearpage')
    out.append('\\subsection{Stage~5 safety effects at a glance}')
    out.append('\\label{app:safety-forest}')
    bfig = _safety_benefit_figure(tags, based, figdir)
    if bfig:
        out.append('The force-included crash and safety terms from the Stage~5 '
                   'mode and destination models, as coefficient estimates with '
                   '95\\% confidence intervals. Points left of the dashed zero '
                   'line indicate that higher crash exposure / lower safety '
                   'reduces the utility of the mode or destination.')
        out.append('\\begin{figure}[h!]\\centering')
        out.append('\\includegraphics[width=0.9\\linewidth]'
                   '{figures/appendix/safety_benefit_forest.png}')
        out.append('\\caption{Stage~5 crash/safety coefficients (95\\% CI) by '
                   'segment and sub-model.}')
        out.append('\\label{fig:safety-forest}')
        out.append('\\end{figure}')
        n_fig += 1
    else:
        out.append('No Stage~5 crash/safety coefficients were found in the '
                   'available run artifacts.')

    tex = os.path.join(pre, 'appendix_models.tex')
    with open(tex, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    n_coef = sum(1 for ln in out if 'coefficients shown' in ln)
    print('wrote %s (%d figures, %d coefficient tables; primary=%s; tags: %s)'
          % (tex, n_fig, n_coef, tag, ', '.join(tags)))


if __name__ == '__main__':
    main()
