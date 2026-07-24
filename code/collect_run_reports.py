"""
collect_run_reports.py
======================
Archive the newest run's validation artifacts and embed a curated, fully
self-contained gallery into the model-review deck
(abm_safety/model_review_deck.html), so every pipeline run's diagnostics are
saved in one place and visible as slides.

Two outputs per run:

1. ARCHIVE  — abm_safety/report_assets/<run_name>/
   Copies of the full validation artifacts (HTML reports, PNGs, key CSVs)
   so they survive run-directory cleanups and can be opened directly.

2. DECK EMBED — the gallery slide in model_review_deck.html
   A curated subset of plots, downscaled and inlined as base64 data URIs
   between /*RUN_REPORTS_START*/ ... /*RUN_REPORTS_END*/ sentinels, grouped
   into sections (Validation / Destination & location / Search convergence /
   Accessibility & spatial / Safety). Self-contained: works in the published
   artifact (CSP blocks file references) and when the HTML is opened locally.

Pipeline integration: called non-fatally at the end of pbs_pipeline_mode.pbs
(right after generate_sa2_pattern_slide.py). Manual use:

    python collect_run_reports.py                    # newest run dir
    python collect_run_reports.py --run runs/ga_parallel/sa_bandit_staged__X
"""
from __future__ import annotations

import argparse
import base64
import datetime
import glob
import io
import json
import os
import re
import shutil
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DECK_HTML = os.path.normpath(os.path.join(CODE_DIR, '..', 'abm_safety', 'model_review_deck.html'))
ASSETS_DIR = os.path.normpath(os.path.join(CODE_DIR, '..', 'abm_safety', 'report_assets'))

SENTINEL_RE = re.compile(r'/\*RUN_REPORTS_START\*/.*?/\*RUN_REPORTS_END\*/', re.DOTALL)

# Per-image / total budget for the deck embed (base64 inflates ~4/3)
MAX_IMG_BYTES = 400_000
MAX_TOTAL_BYTES = 6_000_000
MAX_WIDTH_PX = 1200

# Curated gallery: section -> list of (glob relative to run dir, caption, max n)
GALLERY_SPEC = [
    ('Validation (HHTS in-sample)', [
        ('ga_hhts_validation/plots/validation/val_SUMMARY_dashboard.png', 'Summary dashboard — all stages at a glance', 1),
        ('ga_hhts_validation/plots/validation/val_stage1_frequency.png', 'Stage 1 — activity frequency, observed vs predicted', 1),
        ('ga_hhts_validation/plots/validation/val_stage3_duration.png', 'Stage 3 — activity duration distributions', 1),
        ('ga_hhts_validation/plots/validation/val_stage4_periods.png', 'Stage 4 — schedule placement periods', 1),
        ('ga_hhts_validation/plots/validation/val_stage5_mode_shares.png', 'Stage 5 — mode shares, observed vs predicted', 1),
        ('ga_hhts_validation/plots/validation/val_stage5_confusion.png', 'Stage 5 — mode confusion matrix (sampled predictions)', 1),
    ]),
    ('Destination & location choice', [
        ('location_choice/shadow_price_convergence_*.png', 'Shadow-price convergence', 3),
        ('location_choice/shadow_price_validation_*.png', 'Shadow-price validation, observed vs predicted zone counts', 3),
        ('ga_hhts_validation/plots/validation/val_stage5_dest_work_share_part1.png', 'Work destination shares (zones 1–40)', 1),
    ]),
    ('Search convergence', [
        ('plots/ga_all_stages.png', 'SA-bandit BIC/n convergence, all stages', 1),
        ('plots/ga_stage*.png', 'Per-stage search convergence', 6),
        ('stage_reports/ga_stage*.png', 'Per-stage search report', 4),
    ]),
    ('Accessibility & spatial', [
        ('plots/accessibility_distributions.png', 'Zone accessibility measure distributions (371-zone scope)', 1),
        ('plots/accessibility_vs_landuse.png', 'Accessibility vs land-use intensity', 1),
        ('plots/distance_distribution.png', 'Trip distance distribution', 1),
        ('plots/desire_lines.png', 'Desire lines — predicted OD flows', 1),
    ]),
    ('Safety experiment', [
        ('safety_plots/safety_mode_share.png', 'Mode share with vs without crash-risk skims', 1),
        ('safety_plots/safety_crash_coefs.png', 'Crash-exposure coefficients by segment', 1),
        ('plots/safety_*.png', 'Safety comparison', 4),
    ]),
]

# Full artifacts archived (not embedded) — HTML reports and key CSVs
ARCHIVE_SPEC = [
    'ga_hhts_validation/validation_report.html',
    'validation_report.html',
    'location_choice/destination_mode_validation.html',
    'location_choice/location_assignment_validation.html',
    'location_choice/*_coefficients.csv',
    'stage_reports/ga_stage*.html',
    'stage_reports/ga_stage*_table.csv',
    'stage5_trip_mode_coefficients.html',
    'safety_report.html',
    'run_outline.txt',
    'timings.csv',
    'ga_hhts_validation/plots/validation/*.png',
    'location_choice/*.png',
    'plots/*.png',
]


def find_newest_run() -> str | None:
    """Newest run dir under runs/ that actually contains report artifacts."""
    cands = []
    for d in glob.glob(os.path.join(CODE_DIR, 'runs', '*', '*')):
        if not os.path.isdir(d):
            continue
        has_reports = (glob.glob(os.path.join(d, 'plots', '*.png'))
                       or glob.glob(os.path.join(d, 'ga_hhts_validation', 'plots', 'validation', '*.png'))
                       or glob.glob(os.path.join(d, 'stage_reports', '*.png')))
        if has_reports:
            cands.append(d)
    return max(cands, key=os.path.getmtime) if cands else None


def _shrink_png(path: str) -> bytes | None:
    """Return image bytes ready for embedding (downscaled JPEG/PNG), or None."""
    try:
        raw = open(path, 'rb').read()
    except OSError:
        return None
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        if im.width > MAX_WIDTH_PX:
            im = im.resize((MAX_WIDTH_PX, int(im.height * MAX_WIDTH_PX / im.width)))
        buf = io.BytesIO()
        # white background + JPEG: matplotlib PNGs compress 3-8x with no
        # visible loss at slide scale
        if im.mode in ('RGBA', 'P', 'LA'):
            from PIL import Image as _I
            bg = _I.new('RGB', im.size, (255, 255, 255))
            bg.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1])
            im = bg
        im.convert('RGB').save(buf, format='JPEG', quality=78, optimize=True)
        out = buf.getvalue()
        return out if len(out) < len(raw) else raw
    except Exception:
        return raw if len(raw) <= MAX_IMG_BYTES else None


def build_gallery(run_dir: str) -> dict:
    run_name = os.path.basename(run_dir.rstrip('/\\'))
    total = 0
    sections = []
    for section, entries in GALLERY_SPEC:
        imgs = []
        for pat, caption, max_n in entries:
            matches = sorted(glob.glob(os.path.join(run_dir, pat)))[:max_n]
            for m in matches:
                data = _shrink_png(m)
                if data is None or len(data) > MAX_IMG_BYTES:
                    continue
                if total + len(data) > MAX_TOTAL_BYTES:
                    break
                total += len(data)
                mime = 'image/jpeg' if data[:3] == b'\xff\xd8\xff' else 'image/png'
                cap = caption
                if max_n > 1:
                    stem = os.path.splitext(os.path.basename(m))[0]
                    seg = stem.split('_')[-1]
                    cap = f"{caption} — {seg}"
                imgs.append({
                    'caption': cap,
                    'file': os.path.relpath(m, run_dir).replace('\\', '/'),
                    'src': f"data:{mime};base64,{base64.b64encode(data).decode()}",
                })
        if imgs:
            sections.append({'name': section, 'images': imgs})
    return {
        'run': run_name,
        'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'run_mtime': datetime.datetime.fromtimestamp(os.path.getmtime(run_dir)).strftime('%Y-%m-%d %H:%M'),
        'embedded_bytes': total,
        'sections': sections,
    }


def archive_run(run_dir: str) -> str:
    run_name = os.path.basename(run_dir.rstrip('/\\'))
    dest = os.path.join(ASSETS_DIR, run_name)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for pat in ARCHIVE_SPEC:
        for src in glob.glob(os.path.join(run_dir, pat)):
            rel = os.path.relpath(src, run_dir)
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:
                if (not os.path.exists(out)
                        or os.path.getmtime(src) > os.path.getmtime(out)):
                    shutil.copy2(src, out)
                    n += 1
            except OSError as e:
                print(f"[reports] copy failed {rel}: {e}")
    print(f"[reports] archived {n} artifact(s) -> {dest}")
    return dest


def inject(deck_path: str, data: dict) -> bool:
    with open(deck_path, 'r', encoding='utf-8') as fh:
        html = fh.read()
    payload = ('/*RUN_REPORTS_START*/window.RUN_REPORTS = '
               + json.dumps(data, ensure_ascii=False, separators=(',', ':'))
               + ';/*RUN_REPORTS_END*/')
    new_html, cnt = SENTINEL_RE.subn(payload, html)
    if cnt == 0:
        print(f"[reports] ERROR: RUN_REPORTS sentinels not found in {deck_path}")
        return False
    with open(deck_path, 'w', encoding='utf-8') as fh:
        fh.write(new_html)
    print(f"[reports] deck updated: {deck_path} "
          f"({data['embedded_bytes'] / 1e6:.1f} MB embedded, "
          f"{sum(len(s['images']) for s in data['sections'])} plots, "
          f"{len(data['sections'])} sections)")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='run dir (default: newest with reports)')
    ap.add_argument('--deck', default=DECK_HTML)
    ap.add_argument('--no-archive', action='store_true')
    args = ap.parse_args(argv)

    run_dir = args.run or find_newest_run()
    if not run_dir or not os.path.isdir(run_dir):
        print("[reports] no run directory with report artifacts found")
        return 1
    print(f"[reports] collecting from {run_dir}")

    if not args.no_archive:
        archive_run(run_dir)

    data = build_gallery(run_dir)
    if not data['sections']:
        print("[reports] no embeddable plots found — deck left unchanged")
        return 0
    if os.path.exists(args.deck):
        inject(args.deck, data)
    else:
        print(f"[reports] deck not found at {args.deck}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
