# -*- coding: utf-8 -*-
"""Patch model_review_deck.html"""
import re, sys

PATH = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(PATH, encoding="utf-8") as f:
    html = f.read()

# 1. Remove Leaflet CDN links
html = re.sub(r'<link rel="stylesheet" href="https://unpkg\.com/leaflet[^>]*>\s*', '', html)
html = re.sub(r'<script src="https://unpkg\.com/leaflet[^>]*></script>\s*', '', html)
print("Removed Leaflet CDN links")

# 2. Remove Leaflet CSS
html = re.sub(r'<style>\s*\.leaflet-container[^<]*</style>\s*', '', html)
html = re.sub(r'\.leaflet-control-attribution[^}]*}\s*', '', html)
html = re.sub(r'\.map-trip-label[^}]*}\s*', '', html)

# 3. Replace SA2 density slide
old_pat = (r'<section class="slide" id="sa2-density-slide".*?</section>'
           r'\s*<script>\s*\(function\(\)\s*\{.*?var ZONES = \{.*?\};'
           r'\s*window\.ZONES_ALL = ZONES;.*?</script>\s*')
NEW = """\
      <section class="slide" id="sa2-density-slide" style="padding:3.5% 4.5%;">
        <h2 class="slide-kicker">Population &amp; Geography</h2>
        <h1 class="slide-title">Synthetic population: 3.7M persons across 371 SEQ SA2 zones</h1>
        <div class="body-area" style="gap:2vh;">
          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat hero"><div class="value">3,700,000</div><div class="label">synthetic residents</div></div>
            <div class="stat"><div class="value">371</div><div class="label">SA2 zone universe</div></div>
            <div class="stat"><div class="value">2,300,950</div><div class="label">person-day plans</div></div>
            <div class="stat"><div class="value">7,070,358</div><div class="label">trips generated</div></div>
          </div>
          <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
            <div class="stat"><div class="value">330</div><div class="label">SEQ SA2 zones (ABS 2021)</div></div>
            <div class="stat"><div class="value">41</div><div class="label">Tweed Shire + border zones</div></div>
            <div class="stat"><div class="value">~80</div><div class="label">SA3 districts</div></div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.85rem; max-width:none;">
            Each synthetic person has a full activity-travel schedule: CDAP &rarr; activity frequencies &rarr; durations &rarr; timing &rarr; destination &rarr; mode &rarr; trip chain.
            Zone system derived from ABS 2021 SA2/SA3/SA4 boundaries; landuse reconstructed from Census + EMME network model.
          </p>
          <div class="card" style="flex:1;">
            <h3 style="margin-top:0;">How the population feeds the model</h3>
            <ul class="plain">
              <li><b>CDAP (Stage 0):</b> Each person classified as mandatory / non-mandatory / home daily pattern</li>
              <li><b>Activity generation:</b> Ordered logit predicts how many work, edu, social, shopping, escort, maintenance, discretionary, and eating-out tours each person makes</li>
              <li><b>Destination + mode (Stage 5):</b> Nested logit with ACDC agglomeration, crash-frequency skims, and shadow-price calibration</li>
            </ul>
          </div>
        </div>
      </section>"""

html, n = re.subn(old_pat, NEW, html, flags=re.DOTALL)
print(f"Replaced SA2 density slide: {n}")

# 4. Remove Leaflet JS blocks after PERSON_DAY_SLIDE_END
html = re.sub(r'<!-- Leaflet CSS -->\s*<link rel="stylesheet" href="https://unpkg\.com/leaflet[^>]*>\s*'
              r'<style>\s*#person-map \.leaflet-container[^<]*</style>\s*', '', html)
html = re.sub(r'<script src="https://unpkg\.com/leaflet[^>]*></script>\s*', '', html)
html = re.sub(r'<script>\s*\(function\(\)\s*\{\s*var MAP_DATA = \{.*?</script>\s*', '', html, flags=re.DOTALL)
html = re.sub(r'<script>\s*\(function\(\)\s*\{\s*var AGENTS = \{.*?</script>\s*', '', html, flags=re.DOTALL)
html = re.sub(r'<script>\s*var obs = new MutationObserver.*?</script>\s*', '', html, flags=re.DOTALL)
print("Removed Leaflet JS blocks")

# 5. Enlarge fonts
font_map = {
    'font-size:0.72rem': 'font-size:0.88rem',
    'font-size:0.68rem': 'font-size:0.84rem',
    'font-size:0.66rem': 'font-size:0.82rem',
    'font-size:0.64rem': 'font-size:0.80rem',
    'font-size:0.62rem': 'font-size:0.78rem',
    'font-size:0.60rem': 'font-size:0.76rem',
    'font-size:0.58rem': 'font-size:0.74rem',
    'font-size:0.55rem': 'font-size:0.72rem',
    'font-size:0.54rem': 'font-size:0.70rem',
    'font-size:0.53rem': 'font-size:0.69rem',
    'font-size:0.52rem': 'font-size:0.68rem',
    'font-size:0.50rem': 'font-size:0.66rem',
    'font-size:0.48rem': 'font-size:0.64rem',
    'font-size:0.44rem': 'font-size:0.60rem',
    'font-size:0.40rem': 'font-size:0.56rem',
}
for old, new in font_map.items():
    n = html.count(old)
    if n:
        html = html.replace(old, new)
        print(f"  Font: {old} -> {new} ({n}x)")

for old_sz, new_sz in [
    ('font-size:0.7rem', 'font-size:0.86rem'),
    ('font-size:0.75rem', 'font-size:0.90rem'),
    ('font-size:0.78rem', 'font-size:0.92rem'),
    ('font-size:0.80rem', 'font-size:0.94rem'),
    ('font-size:0.82rem', 'font-size:0.96rem'),
    ('font-size:0.85rem', 'font-size:1.00rem'),
    ('font-size:0.86rem', 'font-size:1.00rem'),
    ('font-size:0.90rem', 'font-size:1.04rem'),
    ('font-size:0.95rem', 'font-size:1.08rem'),
]:
    html = html.replace(old_sz, new_sz)

# 6. Update status slides
OLD_PENDING = ('<span class="pill pending">Pending</span>\n'
               '              <ul class="plain" style="margin-top:0.9em;">\n'
               '                <li>Rerun in progress:')
idx = html.find(OLD_PENDING)
if idx >= 0:
    # Find end of pending card
    card_end = html.find('</ul>\n            </div>', idx + len(OLD_PENDING))
    if card_end >= 0:
        card_end += len('</ul>\n            </div>')
        NEW_PENDING = """              <span class="pill pending">Pending &mdash; Next Run</span>
              <ul class="plain" style="margin-top:0.9em;">
                <li><b>guided03 completed all 6 GA stages</b> &mdash; 17h 39m wall, OOM-killed during post-processing at 188GB/194GB (exit 137)</li>
                <li><b>Two-shot destination:</b> STAGE5_DEST_TWO_STAGE=1 &mdash; SA3 district choice with logsum feeding SA2 choice; LL-improvement gate before default</li>
                <li><b>Home-gravity:</b> c_home_pull / c_home_pull_late &mdash; homeward attraction scaled by departure time and remaining day budget; nonmand-only</li>
                <li><b>Themed ACDC:</b> aggl_retail, aggl_office, aggl_edu, aggl_leisure &mdash; mutex-encoded against generic c_aggl; same decay kernel, theme-restricted size weights</li>
                <li><b>Mode pool widened:</b> CBD&times;transit, transit access/egress, micromobility&times;bike, purpose&times;mode dummies &mdash; all with mutex guard encoding groups</li>
                <li><b>Shadow prices:</b> edu 361/371 adjusted (50 fail), nonmand 316/371 (299 fail), work 316/371 (299 fail) &mdash; base form needs the new destination candidates</li>
                <li><b>Memory:</b> Reduce prediction-phase memory via zonal batching to stay under 194GB</li>
              </ul>"""
        old_card = html[idx:card_end]
        html = html.replace(old_card, NEW_PENDING)
        print("Updated pending status")

OLD_DONE = ('<span class="pill done">Done</span>\n'
            '              <ul class="plain" style="margin-top:0.9em;">\n'
            '                <li>All four destination-choice defects diagnosed and corrected in code</li>')
idx = html.find(OLD_DONE)
if idx >= 0:
    card_end = html.find('</ul>\n            </div>', idx + len(OLD_DONE))
    if card_end >= 0:
        card_end += len('</ul>\n            </div>')
        NEW_DONE = """              <span class="pill done">Done</span>
              <ul class="plain" style="margin-top:0.9em;">
                <li>All four destination-choice defects diagnosed and corrected in code</li>
                <li>Zero-gradient estimation failure diagnosed &mdash; numba engine fallback + MXL&rarr;NL guard in place</li>
                <li>Constraint-guided search (SearchLibrium 0.0.138) + random-coefficient policy wired across stages 1&ndash;5</li>
                <li>SEQ + Tweed Shire zone system re-derived from ABS boundaries &mdash; 371 zone universe confirmed</li>
                <li>HPC JAX/GPU environment fixed and reinstall-proofed</li>
                <li><b>guided03 (sa_bandit):</b> all 6 GA stages complete (17h 39m), rho&sup2; edu=0.423/0.605, nonmand=0.549/0.685, work=0.629/0.413 (mode/dest)</li>
              </ul>"""
        old_card = html[idx:card_end]
        html = html.replace(old_card, NEW_DONE)
        print("Updated done status")

# 7. Add flow SVG after pipeline-row
FLOW = """
<div style="display:flex; align-items:center; justify-content:center; padding:0.8em 0;">
<svg viewBox="0 0 900 155" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
  <defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#1C7293"/></marker></defs>
  <rect x="5" y="18" width="95" height="48" rx="6" fill="#1E2761"/><text x="52" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">CDAP</text><text x="52" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 0</text>
  <line x1="100" y1="42" x2="122" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="124" y="18" width="95" height="48" rx="6" fill="#1E2761"/><text x="171" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">Activity Freq</text><text x="171" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 1</text>
  <line x1="219" y1="42" x2="241" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="243" y="18" width="90" height="48" rx="6" fill="#1E2761"/><text x="288" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">Day Budget</text><text x="288" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 2</text>
  <line x1="333" y1="42" x2="355" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="357" y="18" width="90" height="48" rx="6" fill="#1E2761"/><text x="402" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">Duration</text><text x="402" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 3</text>
  <line x1="447" y1="42" x2="469" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="471" y="18" width="90" height="48" rx="6" fill="#1E2761"/><text x="516" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">Schedule</text><text x="516" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 4</text>
  <line x1="561" y1="42" x2="583" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="585" y="6" width="125" height="72" rx="8" fill="#C85A14"/><text x="647" y="28" fill="white" font-size="10" font-weight="bold" text-anchor="middle">Destination</text><text x="647" y="43" fill="white" font-size="10" font-weight="bold" text-anchor="middle">+ Mode Choice</text><text x="647" y="60" fill="#FFE3CE" font-size="7" text-anchor="middle">Stage 5 &middot; NL + ACDC + Crash</text>
  <line x1="710" y1="42" x2="732" y2="42" stroke="#1C7293" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="734" y="18" width="130" height="48" rx="6" fill="#1E2761"/><text x="799" y="38" fill="white" font-size="9" font-weight="bold" text-anchor="middle">Trip Chain</text><text x="799" y="54" fill="#C9D3F5" font-size="7" text-anchor="middle">Stage 6 &rarr; DTA export</text>
  <rect x="190" y="82" width="480" height="26" rx="4" fill="#EBEFFA" stroke="#DCE1F0" stroke-width="0.5"/>
  <text x="430" y="99" fill="#1E2761" font-size="8" text-anchor="middle" font-weight="600">SA-Bandit feature selection across all stages &middot; BIC-guided bandit &times; simulated-annealing</text>
  <path d="M 647 78 L 647 105 L 430 105" fill="none" stroke="#2E7D6B" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="538" y="118" fill="#2E7D6B" font-size="7" text-anchor="middle">skim feedback (network assignment &rarr; travel times)</text>
</svg>
</div>"""

target = '<div class="pipeline-row">'
idx = html.find(target)
if idx >= 0:
    # Find closing of the pipeline-row div
    depth = 0
    for i in range(idx, len(html)):
        if html[i:i+5] == '<div ' or html[i:i+5] == '<div>':
            depth += 1
        elif html[i:i+6] == '</div>':
            depth -= 1
            if depth == 0:
                html = html[:i+6] + FLOW + html[i+6:]
                print("Added pipeline flow SVG")
                break

with open(PATH, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
