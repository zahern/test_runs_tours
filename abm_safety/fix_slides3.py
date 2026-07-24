# -*- coding: utf-8 -*-
"""Fix crash SVG formula, duration slide spacing, heatmap colors."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Fix the crash slide SVG formula ──
# Change "CRASHij = 0.0091" to just "CRASHij" 
html = html.replace(
    '<text x="598" y="38" fill="#D5F5E3" font-size="5.5" text-anchor="middle">= 0.0091</text>',
    '<text x="598" y="38" fill="#D5F5E3" font-size="6" text-anchor="middle">expected crashes</text>'
)
# Change the "expected on OD pair" label
html = html.replace(
    '<text x="598" y="53" fill="#aaa" font-size="5.5" text-anchor="middle">expected on OD pair</text>',
    '<text x="598" y="53" fill="#aaa" font-size="5.5" text-anchor="middle">on this OD pair</text>'
)
# Make the utility formula clearer
html = html.replace(
    '+ &beta;<sub>crash</sub> &middot; CRASH<sub>ij</sub>',
    'enters utility as &beta;<sub>crash</sub>&middot;CRASH<sub>ij</sub>'
)
print("Fixed crash SVG formulas")

# ── 2. Fix duration slide - spread content across full width ──
idx = html.find("Duration model (Stage 3)")
s = html.rfind("<section ", 0, idx)
depth = 1; i = s + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
e = i - 1

NEW_DUR = """\
      <section class="slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">Model Coefficients &middot; Stages 3 &amp; 5</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">Activity duration &amp; destination-mode fit summary</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid #2E3B8E;">
              <h3 style="margin:0 0 0.4em; font-size:0.84rem; color:#2E3B8E;">Duration model (Stage 3) &mdash; what the coefficients mean</h3>
              <p style="font-size:0.72rem; margin:0 0 0.4em;"><b>Log-normal model:</b> ln(duration<sub>minutes</sub>) ~ N(&mu;, &sigma;&sup2;). Each covariate shifts the mean of the log-duration distribution.</p>
              <div style="font-size:0.7rem;line-height:1.55;">
                <p style="margin:0 0 0.15em;color:var(--good);"><b>Positive coefficient</b> (&rarr; longer activity):</p>
                <p style="margin:0 0 0.35em;"><b>const</b> = baseline. Work ~4.55 (exp(4.55) &asymp; 95 min). &nbsp; <b>ptype_FullTimeWorker</b> = +0.98 (~2.7&times; baseline). &nbsp; <b>ptype_PartTimeWorker</b> = +0.49.</p>
                <p style="margin:0 0 0.15em;color:#c0392b;"><b>Negative coefficient</b> (&rarr; shorter activity):</p>
                <p style="margin:0;"><b>time_consumed</b> = later activities are shorter (budget pressure). &nbsp; <b>age_child / age_teen</b> = younger persons have shorter discretionary activities.</p>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em; border-left:4px solid #1C7293;">
              <h3 style="margin:0 0 0.4em; font-size:0.84rem; color:#1C7293;">Stage 5 fit summary &mdash; what the numbers mean</h3>
              <p style="font-size:0.7rem; margin:0 0 0.4em;"><b>McFadden &rho;&sup2; = 1 &minus; LL<sub>model</sub>/LL<sub>null</sub></b>. Measures improvement over constants-only model. &rho;&sup2; &gt; 0.2 is acceptable; &gt; 0.4 is excellent.</p>
              <table style="width:100%;border-collapse:collapse;font-size:0.62rem;margin:0.3em 0;">
                <tr style="background:var(--surface-tint);"><th style="text-align:left;padding:3px 5px;">Segment</th><th style="text-align:left;padding:3px 5px;">Model</th><th style="text-align:right;padding:3px 5px;">&rho;&sup2;</th><th style="text-align:right;padding:3px 5px;">LL final</th><th style="text-align:right;padding:3px 5px;">n obs</th></tr>
                <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.629</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-3,661</td><td style="text-align:right;padding:2px 5px;">6,091</td></tr>
                <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.413</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">6,091</td></tr>
                <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.423</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-2,783</td><td style="text-align:right;padding:2px 5px;">3,049</td></tr>
                <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.605</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">3,049</td></tr>
                <tr><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.549</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-9,899</td><td style="text-align:right;padding:2px 5px;">10,454</td></tr>
                <tr style="border-bottom:1px solid var(--border);"><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.685</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">10,454</td></tr>
              </table>
              <p style="font-size:0.6rem;color:var(--text-muted);margin:0;"><b>BIC</b> = k&middot;ln(n) &minus; 2&middot;LL. Lower is better. SA search minimises BIC across all stages.</p>
            </div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.6rem;">guided03 results. Duration: MetaCount SA-Guided, 57s. Stage 5: Larch NL/MNL, 37m30s. All &rho;&sup2; from in-sample HHTS estimation.</p>
        </div>
      </section>"""

html = html[:s] + NEW_DUR + html[e:]
print(f"Fixed duration slide ({e-s:,} -> {len(NEW_DUR):,} bytes)")

# ── 3. Fix heatColor to white-to-dark-green ──
old_hc = "function heatColor(t) {"
hc_idx = html.find(old_hc)
if hc_idx > 0:
    depth = 1; j = html.find("{", hc_idx) + 1
    while depth > 0 and j < len(html):
        if html[j] == "{": depth += 1
        elif html[j] == "}": depth -= 1
        j += 1
    hc_end = j - 1
    
    new_hc = """function heatColor(t) {
        t = Math.max(0, Math.min(1, t));
        // White (0) -> light green -> dark green (1)
        var r = Math.round(245 - 225*t), g = Math.round(248 - 200*t), b = Math.round(244 - 220*t);
        return [r, g, b];
      }"""
    
    html = html[:hc_idx] + new_hc + html[hc_end+1:]
    
    # Update legend gradient
    html = html.replace(
        "background:linear-gradient(90deg, rgb(34,139,34), rgb(224,199,34), rgb(229,59,54))",
        "background:linear-gradient(90deg, rgb(245,248,244), rgb(130,168,110), rgb(20,48,24))"
    )
    html = html.replace(
        "background:linear-gradient(90deg, rgb(36,80,130), rgb(186,180,130), rgb(216,100,30), rgb(236,40,40))",
        "background:linear-gradient(90deg, rgb(245,248,244), rgb(130,168,110), rgb(20,48,24))"
    )
    print("Fixed heatColor to white-to-dark-green")

# ── 4. Fix heatmap axis label overlap ──
# The xtab heatmap has column/row labels that may overlap. Add better spacing.
old_xtab_grid = 'display:grid; grid-template-columns: auto repeat('
if old_xtab_grid in html:
    html = html.replace(
        old_xtab_grid,
        'display:grid; grid-template-columns: 80px repeat('
    )
    html = html.replace(
        "gap:3px; font-size:0.6rem;",
        "gap:2px; font-size:0.62rem;"
    )
    # Fix row label width
    html = html.replace(
        "color:var(--text-muted);display:flex;align-items:center;padding-right:4px;white-space:nowrap;",
        "color:var(--text-muted);display:flex;align-items:center;padding-right:4px;white-space:nowrap;font-size:0.58rem;"
    )
    # Fix column header
    html = html.replace(
        "text-align:center;color:var(--text-muted);padding:2px;white-space:nowrap;",
        "text-align:center;color:var(--text-muted);padding:2px 1px;white-space:nowrap;font-size:0.56rem;"
    )
    print("Fixed heatmap axis spacing")

# ── 5. Update heatmap legend text to match new colors ──
html = html.replace("Low</span> &rarr; <span style=\"color:#ec2828;\">High", "Low</span> &rarr; <span style=\"color:#143018;\">High")
html = html.replace("Low</span> &rarr; <span style=\"color:#ec2828;\">High", "Low</span> &rarr; <span style=\"color:#143018;\">High")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
