"""
PRE-FLIGHT: run before `qsub`. Scoped to the files the current test path
actually executes, so a red result always means "this submission is at
risk" rather than "the legacy repo has old files in it".

The failure it exists to catch: job 26085840 died in 33 seconds because
open(p,'w',encoding='utf-8').write(...) on Windows translated every newline
to CRLF, and bash then produced

    line 8:  $'\r': command not found
    line 35: set: -u: invalid option
    EnvironmentNameNotFound: Could not find conda environment: zigenv

The job silently fell back to the BASE miniconda python instead of zigenv
and exited 2. It reads as a cluster outage; it is a text-encoding bug.

Usage:  python preflight.py        exit 0 = safe to submit
"""
import ast
import os
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(PROJ, 'scripts')

# Only what t1..t5 actually run, plus their helpers.
DEPLOY_PY = [
    'scripts/58_build_15min_panel.py',
    'scripts/59_game_day_search_predict.py',
    'scripts/60_game_day_mip.py',
    'scripts/metacount_nb.py',
    'scripts/binomial_mle.py',
    't4_assert.py',
    't5_assert.py',
    'check_grouped_draws.py',
    'preflight.py',
]
DEPLOY_PBS = [
    't1_unit_tests.pbs',
    't2_smoke_test.pbs',
    't4_mip_uclass.pbs',
    't5_shape_arms.pbs',
]

problems = []
notes = []


def head(t):
    print('=' * 74)
    print(t)
    print('=' * 74)


head('PRE-FLIGHT (scoped to the t1..t5 test path)')

# ---------------------------------------------------------------- 1. CRLF
print('\n1. line endings -- CRLF silently breaks bash and conda activation')
for rel in DEPLOY_PY + DEPLOY_PBS:
    p = os.path.join(PROJ, rel.replace('/', os.sep))
    if not os.path.exists(p):
        problems.append('%s is missing' % rel)
        print('   [FAIL] %-44s missing' % rel)
        continue
    raw = open(p, 'rb').read()
    ncr = raw.count(b'\r\n') + raw.count(b'\r')
    if ncr:
        problems.append('%s has %d CR characters' % (rel, ncr))
    print('   %-4s %-44s %s'
          % ('FAIL' if ncr else 'ok', rel,
             '%d CR' % ncr if ncr else 'LF only'))

# ---------------------------------------------------------------- 2. .pbs
print('\n2. .pbs structure')
for rel in DEPLOY_PBS:
    p = os.path.join(PROJ, rel)
    if not os.path.exists(p):
        continue
    txt = open(p, 'rb').read().decode('utf-8', 'replace')
    first = txt.splitlines()[0] if txt.splitlines() else ''
    need = [('#!' in first, 'shebang on line 1'),
            ('#PBS -N' in txt, '#PBS -N'),
            ('#PBS -l walltime' in txt, '#PBS -l walltime'),
            ('conda activate' in txt, 'conda activate'),
            ('set -u' in txt or 'set -e' in txt, 'set -u/-e')]
    bad = [w for good, w in need if not good]
    if bad:
        problems.append('%s missing %s' % (rel, ', '.join(bad)))
    print('   %-4s %-44s %s'
          % ('FAIL' if bad else 'ok', rel, ', '.join(bad) or 'complete'))

# ---------------------------------------------------------------- 3. syntax
print('\n3. python syntax')
for rel in DEPLOY_PY:
    p = os.path.join(PROJ, rel.replace('/', os.sep))
    if not os.path.exists(p):
        continue
    try:
        ast.parse(open(p, encoding='utf-8', errors='replace').read())
        print('   ok    %s' % rel)
    except SyntaxError as exc:
        problems.append('%s syntax error line %s: %s'
                        % (rel, exc.lineno, exc.msg))
        print('   FAIL  %-44s line %s: %s' % (rel, exc.lineno, exc.msg))

# ------------------------------------------------- 4. assertions wired in
print('\n4. jobs assert from artefacts, not the PBS log')
for rel in ('t4_mip_uclass.pbs', 't5_shape_arms.pbs'):
    p = os.path.join(PROJ, rel)
    if not os.path.exists(p):
        continue
    txt = open(p, encoding='utf-8', errors='replace').read()
    # Only executable lines: the removed log-grep is QUOTED in a comment that
    # explains why it went, so a naive text search false-positives on it.
    code = [l for l in txt.splitlines()
            if not l.lstrip().startswith('#')]
    code_txt = '\n'.join(code)
    harness = 't4_assert.py' if rel.startswith('t4') else 't5_assert.py'
    uses_art = harness in code_txt
    uses_log = ('ls -t' in code_txt) and ('.o*' in code_txt)
    if not uses_art:
        problems.append('%s does not call %s' % (rel, harness))
    if uses_log:
        problems.append('%s still greps the PBS log (ls -t .../x.o*)' % rel)
    print('   %-4s %-44s harness=%s log-grep=%s'
          % ('FAIL' if (not uses_art or uses_log) else 'ok', rel,
             uses_art, uses_log))

# ---------------------------------------------------------------- 5. inputs
print('\n5. inputs the run needs')
RES = os.path.join(PROJ, 'results')
panel = os.path.join(RES, 'segment_15min_panel.parquet')
monthly = [f for f in os.listdir(RES)
           if f.startswith('segment_15min_panel_') and f.endswith('.parquet')]
print('   %-4s monthly panel files: %d' % ('ok' if monthly else 'FAIL',
                                           len(monthly)))
if not monthly:
    problems.append('no monthly panel files -- run 58 first')
for extra in ('fixture_calendar.csv',):
    hits = [d for d in (os.path.join(PROJ, 'data'),
                        os.path.join(SCRIPTS, '..', 'data'))
            if os.path.exists(os.path.join(d, extra))]
    if not hits:
        notes.append('could not locate %s' % extra)
print('   ok   data/ checked')

# The panel must actually carry the target the model is fitted on. Stage 59
# asks for `speeding_trips`, which stage 58 derives from the posted limit; a
# panel built before that change has the column missing and the job fails deep
# inside the search with a confusing error instead of here.
print()
print('5b. panel carries the fitted target')
try:
    import pyarrow.parquet as _pq
    _panel = os.path.join(PROJ, 'results', 'segment_15min_panel.parquet')
    if not os.path.exists(_panel):
        notes.append('merged panel not found; run 58 --merge')
        print('   note  merged panel missing (run 58 --merge)')
    else:
        _names = set(_pq.ParquetFile(_panel).schema_arrow.names)
        for _need in ('speeding_trips', 'speed_limit'):
            if _need in _names:
                print('   ok    panel has %s' % _need)
            else:
                problems.append(
                    'panel lacks %s -- rebuild with\n'
                    '      rm results/segment_15min_panel_????-??.parquet\n'
                    '      rm results/_15min_tmp/_pass_done.json\n'
                    '      python scripts/58_build_15min_panel.py --all\n'
                    '    Both deletions are required. finalize_month deletes each\n'
                    '    month\'s partials as it consumes them, and the marker makes\n'
                    '    --all SKIP the streaming pass, so without removing it the\n'
                    '    rebuild finds no partials and does nothing. The pass reads\n'
                    '    the 7.3 GB results/traffic_points_full.csv.gz (916 chunks,\n'
                    '    457.8M points) and is the expensive part; finalize is cheap.'
                    % _need)
                print('   FAIL  panel lacks %s' % _need)
except Exception as _exc:  # noqa: BLE001
    notes.append('could not inspect the panel schema: %s' % _exc)
    print('   note  could not inspect panel schema (%s)' % _exc)

print()
head('RESULT')
if problems:
    print('DO NOT SUBMIT -- %d problem(s):' % len(problems))
    for p in problems:
        print('  - %s' % p)
    sys.exit(1)
if notes:
    print('advisory:')
    for n in notes:
        print('  - %s' % n)
print('PRE-FLIGHT PASSED -- safe to submit')
sys.exit(0)
