# -*- coding: utf-8 -*-
"""Replace destination coefficients slide with highlighted version."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

idx = html.find("Destination choice: ACDC size terms")
s = html.rfind("<section ", 0, idx)
depth = 1; i = s + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
e = i - 1

NEW = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Model Coefficients &middot; Stage 5</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">Destination choice: what drives zone attractiveness</h1>
        <div class="body-area" style="gap:0.7vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div style="display:flex;flex-direction:column;gap:0.4em;">
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid #e74c3c; background:#fef0f0;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem; color:#c0392b;">&#9888; Crash &amp; safety <span style="font-size:0.6rem;font-weight:400;">(force-included, never dropped)</span></h3>
                <div style="font-size:0.72rem;line-height:1.5;">
                  <b style="font-family:var(--font-mono);">c_crash = -0.100</b> &mdash; safety disutility. The only coefficient the constraint architecture guarantees is present in every destination utility. CRASH skim enters as log(CRASH + 1). The negative sign means zones with higher crash exposure on the route are systematically less likely to be chosen, all else equal.
                </div>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--primary); flex:1;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem;">Distance &amp; impedance</h3>
                <table style="width:100%;font-size:0.7rem;border-collapse:collapse;">
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_log_dist</td><td style="text-align:right;padding:2px 4px;color:var(--primary);font-weight:700;">&minus;</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">primary deterrent</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_dist</td><td style="text-align:right;padding:2px 4px;color:var(--primary);font-weight:700;">&minus;</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">linear distance penalty</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_logsum</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">mode accessibility elasticity</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_intrazonal</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">intra-zone trip preference</td></tr>
                </table>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--good); flex:1;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem;">Spatial structure (ACDC)</h3>
                <table style="width:100%;font-size:0.7rem;border-collapse:collapse;">
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_aggl</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+0.150</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">agglomeration bonus</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_comp</td><td style="text-align:right;padding:2px 4px;color:var(--primary);font-weight:700;">&minus;0.060</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">competition penalty</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_walk_feas</td><td style="text-align:right;padding:2px 4px;">0.000</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">walkable zone indicator</td></tr>
                </table>
              </div>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.4em;">
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid #2E3B8E; flex:1;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem;">Work attractors <span style="font-size:0.6rem;font-weight:400;color:var(--text-muted);">(conditioned on industry)</span></h3>
                <table style="width:100%;font-size:0.7rem;border-collapse:collapse;">
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_indp_finance</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">FPSEMPN &middot; indp 10-13</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_indp_health</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">HEREMPN &middot; indp 16-17</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_indp_retail</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">RETEMPN &middot; indp 7-8</td></tr>
                </table>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid #1C7293; flex:1;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem;">Education attractors <span style="font-size:0.6rem;font-weight:400;color:var(--text-muted);">(conditioned on study level)</span></h3>
                <table style="width:100%;font-size:0.7rem;border-collapse:collapse;">
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_edu_post</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">tertiary &middot; STUDYING=3,4</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_edu_primary</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">primary &middot; STUDYING=1</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_edu_secondary</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">secondary &middot; STUDYING=2</td></tr>
                </table>
              </div>
              <div class="card" style="padding:0.5em 0.7em; border-left:4px solid var(--orange); flex:1;">
                <h3 style="margin:0 0 0.2em; font-size:0.82rem;">Person &times; distance interactions</h3>
                <table style="width:100%;font-size:0.7rem;border-collapse:collapse;">
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_incp_dist</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">higher income = less deterred by distance</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_age_dist</td><td style="text-align:right;padding:2px 4px;color:var(--primary);font-weight:700;">&minus;</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">older = more deterred by distance</td></tr>
                  <tr><td style="padding:2px 4px;font-family:var(--font-mono);font-weight:600;">c_veh_dist</td><td style="text-align:right;padding:2px 4px;color:var(--good);font-weight:700;">+</td><td style="font-size:0.62rem;color:var(--text-muted);padding:2px 4px;">more vehicles = less deterred</td></tr>
                </table>
              </div>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.4em 0.7em;font-size:0.66rem;">
            <b style="color:var(--orange);">Key:</b> <span style="color:var(--good);font-weight:600;">Green +</span> = attracts trips to zone &nbsp;|&nbsp; <span style="color:var(--primary);font-weight:600;">Blue &minus;</span> = deters trips &nbsp;|&nbsp; <span style="background:#fce4e4;padding:1px 5px;border-radius:2px;font-weight:600;color:#c0392b;">Pink card</span> = <b>crash/safety component</b> &mdash; the only coefficient guaranteed present in every destination utility specification. All other coefficients selected by SA search from 30&ndash;50 candidates per segment.
          </div>
        </div>
      </section>"""

html = html[:s] + NEW + html[e:]
print(f"Destination coefs slide: {e-s:,} -> {len(NEW):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
