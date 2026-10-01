"""
build_convergence_report.py
===========================
Turn per-shard ``convergence_trace.csv`` files (appended by
ga_hypothesis_testing._save_best_ckpt on every new global best) into
showable improving-convergence evidence for the paper.

For each search label it draws best-fitness-so-far over wall time and writes
a CSV summary (first/best fitness, gain, #improvements, span). Sources, in
priority order:
  1. convergence_trace.csv (exact per-improvement timestamps, new runs);
  2. PBS .o logs (Gen best_BIC/n lines with wall-time interpolation — the
     historical record for runs that predate the trace writer);
  3. best_*.npz fitness + file mtimes (endpoints only, last resort).

Usage:
  python build_convergence_report.py --tag qld_shard4_retry_20260924
      [--run-root runs/ga_parallel] [--out outdir] [--code-dir .]
  (defaults: outdir = <run-root>/<tag>/merged/convergence/)
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


_GEN_RE = re.compile(r"Gen\s+(\d+).*?best_BIC/n=([-\d\.]+)")
_GEN_FEAT_RE = re.compile(r"n_feat=(\d+)")
_NEWBEST_RE = re.compile(r"New best:?\s*(?:\[[^\]]*\]:\s*)?BIC/n=([-\d\.]+)")
_MAINBEST_RE = re.compile(r"Main run best:\s*([-\d\.]+)")
_START_RE = re.compile(r"Started\s*: (.+)")
_FINISH_RE = re.compile(r"Finished:\s*(.+?)\s*exit_code")
_OFILE_RE = re.compile(r"ga_s(\d+)([a-z]+)_(.+)\.o(\d+)$")
_SIDE_RES = (
    (re.compile(r"dest\b.*\bGA\b|\bDest\b.*SA/GA", re.I), "Dest"),
    (re.compile(r"mode\b.*\bGA\b|\bMode\b.*SA/GA", re.I), "Mode"),
)
# Content purpose headers override the filename segment: a shard file can
# contain another purpose's search (e.g. s51nonmand logs show Social GA
# sections). Without this, unrelated searches merge under one label and
# fabricate jumps (the bogus S51-nonmand +0.93 "gain").
_PURP_RE = re.compile(r"\[Stage\s*5\d?\s+([A-Za-z/\-]+?)(?:\s*\(trip\)|"
                      r"\s*\(tour\)|\s+trip\)|\s*\(combined\))?\s*\]",
                      re.I)


def _parse_log_time(s):
    """Parse PBS Started/Finished stamps; None on failure."""
    s = re.sub(r"\s+[A-Z]{3,5}\s+(\d{4})$", r" \1", s.strip())
    for fmt in ("%a %b %d %I:%M:%S %p %Y", "%a %b %d %H:%M:%S %Y",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.Timestamp.strptime(s, fmt)
        except Exception:
            continue
    return None


def _collect_logs(code_dir, tag):
    """Reconstruct per-search best-so-far curves from PBS .o Gen lines.

    Timestamps are linearly interpolated between the job's Started/Finished
    stamps (file mtime when Finished is absent, e.g. SIGKILLs) — honest
    approximations for curve shape; the fitness values themselves are exact.
    Returns (points DataFrame, covered shard-key set).
    """
    rows, covered = [], set()
    try:
        ofiles = glob.glob(os.path.join(code_dir, "*.o*"))
    except Exception:
        return pd.DataFrame(), covered
    for path in sorted(ofiles):
        base = os.path.basename(path)
        m = _OFILE_RE.search(base)
        if not m or tag not in base:
            continue
        stage, seg = m.group(1), m.group(2)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except Exception:
            continue
        t0 = t1 = None
        for ln in lines:
            ms = _START_RE.search(ln)
            if ms and t0 is None:
                t0 = _parse_log_time(ms.group(1))
            mf = _FINISH_RE.search(ln)
            if mf:
                t1 = _parse_log_time(mf.group(1))
        try:
            mt = pd.Timestamp.fromtimestamp(os.path.getmtime(path))
        except Exception:
            mt = None
        t0 = t0 or mt
        t1 = t1 or mt
        if t0 is None or t1 is None or t1 < t0:
            continue
        side, events = "", []
        cur_seg = seg
        seen_segs = {seg}
        for ln in lines:
            pm = _PURP_RE.search(ln)
            if pm:
                cur_seg = pm.group(1).strip().lower()
                seen_segs.add(cur_seg)
            for rx, name in _SIDE_RES:
                if rx.search(ln):
                    side = name
                    break
            g = _GEN_RE.search(ln)
            if g:
                fmf = _GEN_FEAT_RE.search(ln)
                events.append((int(g.group(1)), -float(g.group(2)),
                               int(fmf.group(1)) if fmf else float("nan"),
                               side, cur_seg))
                continue
            nb = _NEWBEST_RE.search(ln)
            if nb:
                events.append((None, -float(nb.group(1)), float("nan"),
                               side, cur_seg))
        if not events:
            continue
        for _cs in seen_segs:
            covered.add((stage, _cs))
        n = len(events)
        for i, (gen, fit, nf, sd, cs) in enumerate(events):
            frac = (i / (n - 1)) if n > 1 else 0.0
            ts = t0 + (t1 - t0) * frac
            label = f"S{stage} {cs}" + (f" {sd}" if sd else "")
            rows.append({"timestamp": ts, "label": label, "fitness": fit,
                         "n_features": nf, "fitness_metric": "",
                         "trace": base + " (log-interpolated)",
                         "gen": gen})
        covered.add((stage, seg))
    if not rows:
        return pd.DataFrame(), covered
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp"])
    return df, covered


def _collect_traces(tagdir):
    rows = []
    for path in glob.glob(os.path.join(tagdir, '**', 'convergence_trace.csv'),
                           recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or 'fitness' not in df.columns:
            continue
        df = df.copy()
        df['trace'] = os.path.relpath(path, tagdir)
        try:
            df['ts'] = pd.to_datetime(df['timestamp'])
        except Exception:
            continue
        rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def _npz_shard_key(path):
    """(stage, seg) from a shard-dir path, or None outside stage layout."""
    m = re.search(r"stage(\d+)[/\\]shards[/\\]([^/\\]+)", path)
    return (m.group(1), m.group(2)) if m else None


def _collect_npz_fallback(tagdir, have_labels, covered_shards=frozenset()):
    """Best-so-far points from best_*.npz mtimes.

    Only for shards with neither trace rows nor log curves (covered_shards
    holds (stage, seg) keys already represented) — otherwise the figure
    drowns in single-point hex-slug files whose curves the logs show better.
    """
    rows = []
    for path in glob.glob(os.path.join(tagdir, '**', 'best_*.npz'),
                           recursive=True):
        # Backup dirs (e.g. work_prev set aside for genuine reseeds) are not
        # live runs — their endpoints would duplicate live-shard curves.
        if '_prev' in path.replace(os.sep, '/'):
            continue
        try:
            z = np.load(path, allow_pickle=True)
            fit = float(z['fitness']) if 'fitness' in z else float('nan')
            genes = np.asarray(z['genes']).ravel()
            label = str(z['label']) if 'label' in z else os.path.basename(path)
        except Exception:
            continue
        if not np.isfinite(fit) or fit <= -1e8:
            continue
        if label in have_labels:
            continue
        _key = _npz_shard_key(path)
        if _key is not None and _key in covered_shards:
            continue
        rows.append({
            'timestamp': pd.Timestamp.fromtimestamp(os.path.getmtime(path)),
            'label': label, 'fitness': fit,
            'n_features': int((np.asarray(genes) == 1).sum()),
            'fitness_metric': '', 'trace': os.path.relpath(path, tagdir),
        })
    if rows:
        df = pd.DataFrame(rows)
        df['ts'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Build convergence report")
    ap.add_argument('--tag', required=True)
    ap.add_argument('--run-root', default=os.path.join('runs', 'ga_parallel'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--code-dir', default=None,
                    help='.o log directory (default: this script\'s dir)')
    ap.add_argument('--no-logs', action='store_true',
                    help='skip .o log parsing (traces + npz only)')
    a = ap.parse_args()

    tagdir = os.path.join(a.run_root, a.tag)
    outdir = a.out or os.path.join(tagdir, 'merged', 'convergence')
    os.makedirs(outdir, exist_ok=True)
    code_dir = a.code_dir or os.path.dirname(os.path.abspath(__file__))

    traces = _collect_traces(tagdir)
    have = set(traces['label'].unique().tolist()) if len(traces) else set()
    if a.no_logs:
        logs, covered = pd.DataFrame(), frozenset()
    else:
        logs, covered = _collect_logs(code_dir, a.tag)
    fallback = _collect_npz_fallback(tagdir, have, covered)
    parts = [p for p in (traces, logs, fallback) if len(p)]
    if not parts:
        print("no convergence data found under", tagdir)
        return
    allpts = pd.concat(parts, ignore_index=True)

    allpts = allpts.sort_values(['label', 'ts']).reset_index(drop=True)
    # Best-so-far envelope per label (traces already only record improvements,
    # but npz fallback + multi-shard overlap need the cummax).
    allpts['best_so_far'] = allpts.groupby('label')['fitness'].cummax()

    summary = []
    for label, g in allpts.groupby('label'):
        g = g.sort_values('ts')
        summary.append({
            'label': label,
            'n_points': int(len(g)),
            'first_fitness': float(g['fitness'].iloc[0]),
            'best_fitness': float(g['best_so_far'].iloc[-1]),
            'gain': float(g['best_so_far'].iloc[-1] - g['fitness'].iloc[0]),
            'first_time': str(g['ts'].iloc[0]),
            'best_time': str(g['ts'].iloc[-1]),
            'span_h': round((g['ts'].iloc[-1] - g['ts'].iloc[0])
                            .total_seconds() / 3600.0, 2),
        })
    summ = pd.DataFrame(summary).sort_values('label')
    summ.to_csv(os.path.join(outdir, 'convergence_summary.csv'), index=False)
    print(summ.to_string(index=False))

    # ── Figure: best-so-far step curves ──────────────────────────────────
    # Plot only labels with a real curve (multi-point with movement) so the
    # figure stays readable; endpoint-only labels remain in the CSV.
    labels = summ['label'].tolist()
    curve_labels = [lb for lb in labels
                    if (len(allpts[allpts['label'] == lb]) > 1
                        and float(allpts[allpts['label'] == lb]
                                  ['best_so_far'].iloc[-1]
                                  - allpts[allpts['label'] == lb]
                                  ['fitness'].iloc[0]) != 0.0)]
    skipped = len(labels) - len(curve_labels)
    labels = curve_labels
    n = len(labels)
    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 3.4 * nrows),
                             squeeze=False)
    t0 = allpts['ts'].min()
    for ax, label in zip(axes.flat, labels):
        g = allpts[allpts['label'] == label].sort_values('ts')
        hours = (g['ts'] - t0).dt.total_seconds().values / 3600.0
        ax.step(hours, g['best_so_far'].values, where='post',
                color='#2E86AB', lw=2)
        ax.scatter(hours, g['fitness'].values, color='#2E86AB', s=14,
                   alpha=0.6, zorder=3)
        gain = float(g['best_so_far'].iloc[-1] - g['fitness'].iloc[0])
        ax.set_title(f"{label}  (gain {gain:+.4f}, {len(g)} improvements)",
                     fontsize=9)
        ax.set_xlabel('Wall time since first improvement (h)', fontsize=8)
        ax.set_ylabel('Best fitness so far (−BIC/n, higher better)',
                      fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(f'GA convergence — {a.tag} (higher = better)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figpath = os.path.join(outdir, 'convergence.png')
    fig.savefig(figpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"figure -> {figpath}")
    print(f"summary -> {os.path.join(outdir, 'convergence_summary.csv')}")


if __name__ == '__main__':
    main()
