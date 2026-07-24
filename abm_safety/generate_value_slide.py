# -*- coding: utf-8 -*-
"""Generate synthetic population value slide with real run data."""
import json, os

DECK_HTML = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"

# Real data from guided03 run
total_pop = 3820161
ptypes = {
    "FullTimeWorker": 1142867, "Retired": 1022488, "PartTimeWorker": 644221,
    "SchoolChild": 595898, "WorkingStudent": 169422, "UniversityStudent": 121338,
    "Preschool": 104204, "NonWorker": 19723,
}
cdap = {"M": 2389073, "N": 884894, "H": 546194}
total_trips = 7070358
total_person_days = 2300950
ga_wall = "17h 39m"
mode_rho2 = {"work": 0.629, "edu": 0.423, "nonmand": 0.549}  # mode
dest_rho2 = {"work": 0.413, "edu": 0.605, "nonmand": 0.685}

ptype_bars = "".join(
    f'<div style="flex:1;min-width:90px;"><div style="font-size:0.56rem;color:var(--text-muted);text-align:center;">{pt}</div>'
    f'<div style="height:4px;background:var(--border);border-radius:2px;margin:2px 0;"><div style="height:100%;width:{c/total_pop*100:.0f}%;background:var(--primary);border-radius:2px;"></div></div>'
    f'<div style="font-size:0.66rem;font-weight:700;text-align:center;font-family:var(--font-mono);">{c:,}</div></div>'
    for pt, c in sorted(ptypes.items(), key=lambda x: -x[1])
)

slide = f"""\
      <section class="slide" id="synth-value-slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Synthetic Population</h2>
        <h1 class="slide-title">What 3.8 million synthetic persons produce</h1>
        <div class="body-area" style="gap:1vh; margin-top:0.3vh;">
          <div class="grid" style="grid-template-columns: repeat(5, 1fr);">
            <div class="stat hero"><div class="value">{total_pop:,}</div><div class="label">synthetic residents</div></div>
            <div class="stat"><div class="value">{total_person_days:,}</div><div class="label">person-day plans</div></div>
            <div class="stat"><div class="value">{total_trips:,}</div><div class="label">trips generated</div></div>
            <div class="stat"><div class="value">371</div><div class="label">SA2 destination zones</div></div>
            <div class="stat"><div class="value">{ga_wall}</div><div class="label">full pipeline wall time</div></div>
          </div>
          <div class="grid" style="grid-template-columns: 0.9fr 1.1fr 1fr; flex:1; min-height:0;">
            <div class="card" style="padding:0.7em 0.9em;">
              <h3 style="margin:0 0 0.4em;font-size:0.84rem;">Person types</h3>
              <div style="display:flex;flex-wrap:wrap;gap:4px;">
                {ptype_bars}
              </div>
            </div>
            <div class="card" style="padding:0.7em 0.9em;">
              <h3 style="margin:0 0 0.4em;font-size:0.84rem;">Daily activity pattern (CDAP)</h3>
              <div style="display:flex;gap:1em;justify-content:center;align-items:flex-end;height:80%;">
                <div style="text-align:center;"><div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;color:var(--primary);">{cdap['M']:,}</div><div style="font-size:0.6rem;color:var(--text-muted);">Mandatory ({cdap['M']/total_pop*100:.0f}%)</div><div style="width:50px;height:{cdap['M']/total_pop*100:.1f}px;background:var(--primary);margin:4px auto 0;border-radius:4px 4px 0 0;"></div></div>
                <div style="text-align:center;"><div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;color:var(--teal);">{cdap['N']:,}</div><div style="font-size:0.6rem;color:var(--text-muted);">Non-mandatory ({cdap['N']/total_pop*100:.0f}%)</div><div style="width:50px;height:{cdap['N']/total_pop*100:.1f}px;background:var(--teal);margin:4px auto 0;border-radius:4px 4px 0 0;"></div></div>
                <div style="text-align:center;"><div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;color:var(--text-faint);">{cdap['H']:,}</div><div style="font-size:0.6rem;color:var(--text-muted);">Home ({cdap['H']/total_pop*100:.0f}%)</div><div style="width:50px;height:{cdap['H']/total_pop*100:.1f}px;background:var(--text-faint);margin:4px auto 0;border-radius:4px 4px 0 0;"></div></div>
              </div>
            </div>
            <div class="card" style="padding:0.7em 0.9em;">
              <h3 style="margin:0 0 0.4em;font-size:0.84rem;">Stage 5 model fit (McFadden &rho;&sup2;)</h3>
              <div style="font-size:0.64rem;">
                <table style="width:100%;border-collapse:collapse;">
                  <tr><th style="text-align:left;padding:2px;">Segment</th><th style="text-align:right;padding:2px;">Mode</th><th style="text-align:right;padding:2px;">Dest</th></tr>
                  <tr><td style="padding:2px;">Work</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{mode_rho2['work']:.3f}</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{dest_rho2['work']:.3f}</td></tr>
                  <tr style="background:var(--surface-tint);"><td style="padding:2px;">Education</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{mode_rho2['edu']:.3f}</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{dest_rho2['edu']:.3f}</td></tr>
                  <tr><td style="padding:2px;">Non-Mandatory</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{mode_rho2['nonmand']:.3f}</td><td style="text-align:right;font-family:var(--font-mono);color:var(--good);padding:2px;">{dest_rho2['nonmand']:.3f}</td></tr>
                </table>
              </div>
            </div>
          </div>
          <div class="card" style="flex:1; background:var(--orange-soft); padding:0.7em 1em;">
            <h3 style="margin:0 0 0.25em; color:var(--orange); font-size:0.82rem;">Why a synthetic population matters</h3>
            <div style="display:flex; gap:1em; font-size:0.68rem; line-height:1.5;">
              <div style="flex:1;"><b>Disaggregate heterogeneity.</b> Every person has their own attributes (age, income, vehicles, licence, employment, study level, home zone). Aggregate models collapse this to zone-level averages. The ABM preserves the full joint distribution.</div>
              <div style="flex:1;"><b>Full activity chains.</b> Each person generates a complete day: CDAP pattern &rarr; activity frequencies &rarr; durations &rarr; timing &rarr; destination &rarr; mode &rarr; trip chain. Aggregate models model trips independently.</div>
              <div style="flex:1;"><b>Policy sensitivity.</b> Because attributes are preserved, the model responds to demographic change (aging population, new housing, vehicle electrification) and infrastructure change (new transit, bike lanes, road pricing) at the person level.</div>
            </div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.66rem;">Data from guided03 (sa_bandit run, {ga_wall} wall time). CDAP = Coordinated Daily Activity Pattern. &rho;&sup2; > 0.2 is acceptable; > 0.4 is excellent for discrete choice.</p>
        </div>
      </section>
<script>
// No interactive JS needed - purely data display
</script>"""

# Inject into deck - replace person-day slide
with open(DECK_HTML, encoding="utf-8") as f:
    html = f.read()

# Find person-day slide boundaries
ps = html.find("<!--PERSON_DAY_SLIDE_START-->")
pe = html.find("<!--PERSON_DAY_SLIDE_END-->")
if ps >= 0 and pe >= 0:
    pe += len("<!--PERSON_DAY_SLIDE_END-->")
    html = html[:ps] + slide + html[pe:]
    print(f"Replaced person-day slide ({pe-ps:,} -> {len(slide):,} bytes)")

with open(DECK_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes. Done.")
