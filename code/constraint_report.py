"""
constraint_report.py
=====================
Initial "what is the GA searching, and how is it constrained?" report.

Cross-references the candidate schema (constraint_schema.json, produced by
build_constraint_schema.py) with the ACTIVE merged constraints (built-in
defaults in ga_constraints.py + the GUI overlay ga_constraints_user.json) and
writes, per stage:

  * every candidate term the GA can search over, and
  * how each term is constrained — force-included, forced/never random, in a
    mutually-exclusive (collinear) group, part of a require-one group, sign-
    constrained, protected backbone, or free to search.

Outputs (in the code dir, or --out-dir):
  constraint_report.csv    flat one-row-per-(stage,segment,term) — diff/track it
                           over time as you tune variables & constraints.
  constraint_report.html   readable per-stage tables + a summary.

Run (no pipeline data needed — reads the schema JSON + constraint config):
    python constraint_report.py
    python build_constraint_schema.py && python constraint_report.py   # refresh first
"""
import argparse
import html
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def _normalize_groups(x):
    """Normalize a constraint's group structure to a list of term-lists.

    Handles: list-of-lists (mutually_exclusive), a single (min, [terms]) tuple
    or a list of them (min_behavioral), and a bare [terms] list.
    """
    if not x:
        return []
    # single (min, [terms]) tuple, e.g. (1, ['a','b','c'])
    if (isinstance(x, (tuple, list)) and len(x) == 2
            and isinstance(x[0], int) and isinstance(x[1], (list, tuple))):
        return [list(x[1])]
    out = []
    for g in x:
        if (isinstance(g, (tuple, list)) and len(g) == 2
                and isinstance(g[0], int) and isinstance(g[1], (list, tuple))):
            out.append(list(g[1]))          # (min, [terms])
        elif isinstance(g, (list, tuple)):
            out.append(list(g))             # [terms]
    return out


def _group_index(term, groups):
    """Return the 1-based index of the first group containing term, else None."""
    for i, g in enumerate(_normalize_groups(groups)):
        if term in g:
            return i + 1
    return None


def _annotate(term, con, ptype_land_terms):
    """Constraint annotations for one candidate term."""
    fr = con.get('force_random') or {}
    a = {
        'force_include': term in (con.get('force_include') or []),
        'force_random': fr.get(term, '') if isinstance(fr, dict) else (term in fr),
        'never_random': term in (con.get('never_random') or []),
        'collinear_group': _group_index(term, con.get('mutually_exclusive')),
        'require_one_group': _group_index(term, con.get('min_behavioral')),
        'protected_backbone': term in (con.get('protected_backbone') or []),
        'ptype_land': term in ptype_land_terms,
    }
    sc = con.get('sign_constraints') or {}
    a['sign'] = sc.get(term, '') if isinstance(sc, dict) else ''
    # "Free" = searchable with no hard include/exclude/group pinning.
    a['status'] = ('forced-in' if a['force_include']
                   else 'collinear-grp' if a['collinear_group']
                   else 'require-one' if a['require_one_group']
                   else 'free')
    return a


def load_availability(search_dirs):
    """Aggregate candidate_availability_stage<N>.csv (written by the Stage 2/3 GA
    diagnostic) into {(stageN, term): status}. A term counts as 'searchable' if
    it was searchable in ANY target/purpose, else the worst status seen."""
    import csv
    import glob
    rank = {'searchable': 3, 'near_constant': 2, 'not_in_data': 1}
    best = {}
    seen_files = set()
    for d in search_dirs:
        if not d:
            continue
        for path in glob.glob(os.path.join(d, 'candidate_availability_stage*.csv')):
            rp = os.path.realpath(path)
            if rp in seen_files:
                continue
            seen_files.add(rp)
            try:
                with open(path, encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        key = (f"stage{r['stage']}", r['term'])
                        st = r['status']
                        if key not in best or rank.get(st, 0) > rank.get(best[key], 0):
                            best[key] = st
            except Exception:
                continue
    return best


def build_rows(schema, get_constraints, ptype_land_terms, avail=None):
    avail = avail or {}
    rows = []
    for stage, sdef in schema.get('stages', {}).items():
        try:
            con = get_constraints(stage)
        except Exception:
            con = {}
        segs = sdef.get('segments') or ['-']
        for term in sdef.get('candidates', []):
            ann = _annotate(term, con, ptype_land_terms)
            # Availability only exists for stages 2/3 (where the diagnostic runs).
            ann['availability'] = avail.get((stage, term), '')
            for seg in segs:
                rows.append({
                    'stage': stage,
                    'label': sdef.get('label', stage),
                    'name_kind': sdef.get('name_kind', ''),
                    'segment': seg,
                    'term': term,
                    **ann,
                })
    return rows


def write_csv(rows, path):
    import csv
    cols = ['stage', 'label', 'name_kind', 'segment', 'term', 'status',
            'availability', 'force_include', 'force_random', 'never_random',
            'collinear_group', 'require_one_group', 'protected_backbone',
            'ptype_land', 'sign']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def write_html(rows, schema, path):
    by_stage = {}
    for r in rows:
        by_stage.setdefault((r['stage'], r['label'], r['name_kind']), []).append(r)
    parts = ["<meta charset='utf-8'><style>",
             "body{font-family:system-ui,Arial;margin:1.5rem;}",
             "table{border-collapse:collapse;margin:.5rem 0 2rem;font-size:13px;}",
             "th,td{border:1px solid #ccc;padding:3px 8px;text-align:left;}",
             "th{background:#f0f0f0;position:sticky;top:0;}",
             ".forced-in{background:#ffe0b2;}.collinear-grp{background:#e1bee7;}",
             ".require-one{background:#c8e6c9;}.free{background:#fff;}",
             "code{background:#f5f5f5;padding:0 3px;}</style>"]
    parts.append(f"<h1>GA constraint / search report</h1>"
                 f"<p>Generated from <code>constraint_schema.json</code> "
                 f"(schema built {html.escape(str(schema.get('generated','?')))}) "
                 f"+ active constraints (ga_constraints defaults + "
                 f"ga_constraints_user.json overlay).</p>")

    # Summary table
    parts.append("<h2>Summary</h2><table><tr><th>Stage</th><th>kind</th>"
                 "<th>candidates</th><th>forced-in</th><th>collinear-grp</th>"
                 "<th>require-one</th><th>force-random</th><th>free</th></tr>")
    for (stage, label, kind), rs in by_stage.items():
        # de-dup terms (segments repeat them)
        terms = {r['term']: r for r in rs}
        c = Counter(r['status'] for r in terms.values())
        n_fr = sum(1 for r in terms.values() if r['force_random'])
        parts.append(f"<tr><td>{html.escape(label)}</td><td>{kind}</td>"
                     f"<td>{len(terms)}</td><td>{c.get('forced-in',0)}</td>"
                     f"<td>{c.get('collinear-grp',0)}</td>"
                     f"<td>{c.get('require-one',0)}</td><td>{n_fr}</td>"
                     f"<td>{c.get('free',0)}</td></tr>")
    parts.append("</table>")

    for (stage, label, kind), rs in by_stage.items():
        parts.append(f"<h2>{html.escape(label)} "
                     f"<small>({html.escape(kind)} names)</small></h2>")
        parts.append("<table><tr><th>term</th><th>segment</th><th>status</th>"
                     "<th>availability</th><th>force_incl</th><th>random</th>"
                     "<th>never_rand</th><th>collin#</th><th>req-one#</th>"
                     "<th>sign</th><th>ptype×land</th></tr>")
        for r in rs:
            av = r.get('availability', '')
            av_cell = (f"<td style='background:#ffcdd2'>{av}</td>"
                       if av == 'not_in_data'
                       else f"<td style='background:#fff9c4'>{av}</td>"
                       if av == 'near_constant' else f"<td>{av}</td>")
            parts.append(
                f"<tr class='{r['status']}'><td><code>{html.escape(r['term'])}</code></td>"
                f"<td>{html.escape(str(r['segment']))}</td><td>{r['status']}</td>"
                f"{av_cell}"
                f"<td>{'Y' if r['force_include'] else ''}</td>"
                f"<td>{html.escape(str(r['force_random']))}</td>"
                f"<td>{'Y' if r['never_random'] else ''}</td>"
                f"<td>{r['collinear_group'] or ''}</td>"
                f"<td>{r['require_one_group'] or ''}</td>"
                f"<td>{html.escape(str(r['sign']))}</td>"
                f"<td>{'Y' if r['ptype_land'] else ''}</td></tr>")
        parts.append("</table>")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--schema', default=os.path.join(HERE, 'constraint_schema.json'))
    ap.add_argument('--out-dir', default=HERE)
    ap.add_argument('--availability-dir', default='',
                    help='Extra dir to search for candidate_availability_stage*.csv '
                         '(also searches out-dir, its parent, and GA_CHECKPOINT_DIR).')
    args = ap.parse_args()

    if not os.path.exists(args.schema):
        raise SystemExit(
            f"{args.schema} not found — build it first:\n"
            "  python build_constraint_schema.py        (real data / all stages)\n"
            "  python build_constraint_schema.py --no-data   (fast, stages 1-4)")

    schema = json.load(open(args.schema, encoding='utf-8'))
    from ga_constraints import get_constraints

    # Availability (Stage 2/3): fold in candidate_availability_stage*.csv from
    # the out-dir, its parent (run dir), and GA_CHECKPOINT_DIR if set.
    avail = load_availability([
        args.out_dir,
        os.path.dirname(os.path.abspath(args.out_dir)),
        os.environ.get('GA_CHECKPOINT_DIR', ''),
        args.availability_dir,
    ])
    if avail:
        na = sum(1 for v in avail.values() if v == 'not_in_data')
        print(f"Availability data found: {len(avail)} terms "
              f"({na} not-in-data) — folded into the report.\n")

    # ptype×land interaction terms (from the GUI overlay / schema default).
    pl = schema.get('default_ptype_land') or []
    try:
        from ga_constraints import load_constraint_overlay
        pl = load_constraint_overlay().get('ptype_land', pl) or pl
    except Exception:
        pass
    ptype_land_terms = set()
    for rule in pl:
        ptype_land_terms.add(rule.get('ptype', ''))
        ptype_land_terms.add(rule.get('land', ''))

    rows = build_rows(schema, get_constraints, ptype_land_terms, avail)
    csv_path = os.path.join(args.out_dir, 'constraint_report.csv')
    html_path = os.path.join(args.out_dir, 'constraint_report.html')
    write_csv(rows, csv_path)
    write_html(rows, schema, html_path)

    # Printed summary
    terms_by_stage = {}
    for r in rows:
        terms_by_stage.setdefault(r['stage'], {})[r['term']] = r['status']
    print(f"Wrote {csv_path}\nWrote {html_path}\n")
    print(f"{'stage':14} {'candidates':>10} {'forced-in':>10} "
          f"{'collinear':>10} {'require1':>9} {'free':>6}")
    for stage, terms in terms_by_stage.items():
        c = Counter(terms.values())
        print(f"{stage:14} {len(terms):>10} {c.get('forced-in',0):>10} "
              f"{c.get('collinear-grp',0):>10} {c.get('require-one',0):>9} "
              f"{c.get('free',0):>6}")


if __name__ == '__main__':
    main()
