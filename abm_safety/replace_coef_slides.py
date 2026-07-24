# -*- coding: utf-8 -*-
"""Replace the two auto-generated coefficient slides with proper explanatory versions."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find both slides
idx1 = html.find("Activity duration (log-normal)")
idx2 = html.find("Schedule placement: MNL own-purpose")

if idx1 < 0 or idx2 < 0:
    print(f"Slides not found: dur={idx1}, sched={idx2}")
else:
    # Find section boundaries
    def find_section(html, idx):
        s = html.rfind("<section ", 0, idx)
        depth = 1; i = s + 1
        while depth > 0 and i < len(html):
            if html[i:i+9] == "<section ": depth += 1
            elif html[i:i+10] == "</section>": depth -= 1
            i += 1
        return s, i - 1

    s1, e1 = find_section(html, idx1)
    s2, e2 = find_section(html, idx2)

    DUR_SLIDE = """\
      <section class="slide" style="padding:2.5% 3.8%;">
        <h2 class="slide-kicker">Model Coefficients &middot; Stages 3 &amp; 5</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">Activity duration &amp; destination-mode fit summary</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.4em; font-size:0.84rem;">Duration model (Stage 3) &mdash; what the coefficients mean</h3>
              <div style="font-size:0.64rem; line-height:1.6;">
                <p style="margin:0 0 0.5em;"><b>Log-normal model:</b> ln(duration<sub>minutes</sub>) ~ N(&mu;, &sigma;&sup2;). Each covariate shifts the mean of the log-duration distribution.</p>
                <p style="margin:0 0 0.3em;"><b>Positive coefficient</b> (&rarr; longer activity):</p>
                <ul style="margin:0 0 0.5em; padding-left:1.2em; font-size:0.62rem;">
                  <li><b>const</b> &mdash; baseline duration for each purpose. Work ~4.55 (exp(4.55) &asymp; 95 min baseline), eating-out lower.</li>
                  <li><b>ptype_FullTimeWorker</b> &mdash; FT workers have longer work durations (+0.98, ~2.7&times; baseline).</li>
                  <li><b>ptype_PartTimeWorker</b> &mdash; PT workers have moderately longer durations (+0.49).</li>
                </ul>
                <p style="margin:0 0 0.3em;"><b>Negative coefficient</b> (&rarr; shorter activity):</p>
                <ul style="margin:0; padding-left:1.2em; font-size:0.62rem;">
                  <li><b>time_consumed</b> &mdash; activities later in the day are shorter (day-budget pressure).</li>
                  <li><b>age_child / age_teen</b> &mdash; younger persons have shorter discretionary activities.</li>
                </ul>
                <p style="margin-top:0.6em; font-size:0.58rem; color:var(--text-faint);">8 purposes &times; per-segment covariates. SA search selects which covariates enter each purpose's model. All coefficients significant at p&lt;0.001 unless noted.</p>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.4em; font-size:0.84rem;">Stage 5 fit summary &mdash; what the numbers mean</h3>
              <div style="font-size:0.64rem; line-height:1.6;">
                <p style="margin:0 0 0.5em;"><b>McFadden &rho;&sup2; = 1 &minus; LL<sub>model</sub>/LL<sub>null</sub></b>. Measures improvement over a model with only constants. &rho;&sup2; &gt; 0.2 is acceptable; &gt; 0.4 is excellent for discrete choice.</p>
                <table style="width:100%;border-collapse:collapse;font-size:0.58rem;margin:0.4em 0;">
                  <tr style="background:var(--surface-tint);"><th style="text-align:left;padding:3px 5px;">Segment</th><th style="text-align:left;padding:3px 5px;">Model</th><th style="text-align:right;padding:3px 5px;">&rho;&sup2;</th><th style="text-align:right;padding:3px 5px;">LL final</th><th style="text-align:right;padding:3px 5px;">n obs</th><th style="text-align:left;padding:3px 5px;">Rating</th></tr>
                  <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.629</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-3,661</td><td style="text-align:right;padding:2px 5px;">6,091</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                  <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.413</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">6,091</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                  <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.423</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-2,783</td><td style="text-align:right;padding:2px 5px;">3,049</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                  <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.605</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">3,049</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                  <tr><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.549</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">-9,899</td><td style="text-align:right;padding:2px 5px;">10,454</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                  <tr style="border-bottom:1px solid var(--border);"><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.685</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);">--</td><td style="text-align:right;padding:2px 5px;">10,454</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                </table>
                <p style="margin:0; font-size:0.56rem; color:var(--text-muted);"><b>BIC</b> = k&middot;ln(n) &minus; 2&middot;LL. <b>AIC</b> = 2k &minus; 2&middot;LL. Both penalise parameter count; lower is better. The SA search minimises BIC to select parsimonious specifications.</p>
              </div>
            </div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.58rem;">guided03 (sa_bandit) results. Duration models: MetaCount SA-Guided, 57s wall. Stage 5: Larch NL/MNL, 37m30s wall. All &rho;&sup2; values from in-sample HHTS estimation.</p>
        </div>
      </section>"""

    SCHED_SLIDE = """\
      <section class="slide" style="padding:2.5% 3.8%;">
        <h2 class="slide-kicker">Model Coefficients &middot; Stage 4</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">Schedule placement: how activities distribute across the day</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.4em; font-size:0.82rem;">What the model does</h3>
              <div style="font-size:0.66rem; line-height:1.6;">
                <p style="margin:0 0 0.5em;">Each activity is placed into one of <b>6 time periods</b> using a multinomial logit:</p>
                <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);margin:0 0 0.5em;">
                  Pr(period<sub>p</sub>) = exp(V<sub>p</sub>) / &Sigma;<sub>q</sub> exp(V<sub>q</sub>)<br>
                  V<sub>p</sub> = &beta;<sub>0,p</sub> + &beta;<sub>1</sub>&middot;time_consumed + &beta;<sub>2</sub>&middot;duration + ...
                </div>
                <p style="margin:0 0 0.3em;"><b>6 time periods:</b></p>
                <ul style="margin:0 0 0.5em;padding-left:1.2em;font-size:0.62rem;">
                  <li><b>Early</b> (before 6am) &mdash; mostly work, some escort</li>
                  <li><b>AM Peak</b> (6&ndash;9am) &mdash; work, education, escort dominate</li>
                  <li><b>Midday</b> (9am&ndash;3pm) &mdash; shopping, maintenance, social</li>
                  <li><b>PM Peak</b> (3&ndash;6pm) &mdash; escort (school pickup), work return</li>
                  <li><b>Evening</b> (6&ndash;8pm) &mdash; eating-out, social, discretionary</li>
                  <li><b>Late</b> (after 8pm) &mdash; social, some eating-out</li>
                </ul>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.4em; font-size:0.82rem;">What the coefficients mean</h3>
              <div style="font-size:0.66rem; line-height:1.6;">
                <p style="margin:0 0 0.5em;">The model estimates <b>own-purpose intercepts</b> for each (purpose, period) combination. These are the &beta;<sub>0,p</sub> terms &mdash; the baseline attractiveness of period <em>p</em> for a given purpose, before considering time pressure or duration.</p>
                <ul style="margin:0 0 0.5em;padding-left:1.2em;font-size:0.62rem;">
                  <li><b>Positive (green):</b> the activity is <em>more likely</em> to occur in this period than the baseline. Work has positive intercepts in AM peak and midday.</li>
                  <li><b>Negative (orange):</b> the activity is <em>less likely</em> in this period. Work has negative intercepts in evening and late.</li>
                  <li><b>Neutral (grey):</b> the period is neither favoured nor avoided for this purpose.</li>
                </ul>
                <p style="margin:0 0 0.3em;"><b>Timing-pressure variables</b> (SA-selected):</p>
                <ul style="margin:0;padding-left:1.2em;font-size:0.62rem;">
                  <li><b>time_consumed</b> &mdash; fraction of day already used. Positive: later activities shift right. Negative: earlier activities resist displacement.</li>
                  <li><b>duration_min</b> &mdash; longer activities gravitate toward midday (more flexible time).</li>
                  <li><b>day_position</b> &mdash; where in the sequence this activity falls. Earlier activities cluster in AM; later ones in PM/evening.</li>
                </ul>
              </div>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.5em 0.8em;font-size:0.62rem;">
            <b style="color:var(--orange);">SA search output:</b> SearchLibrium SA evaluates candidate MNL/MXL specifications per purpose. Random-coefficient variants (MXL) tested when GA_RAND_PARAM_SEARCH=1. Best-BIC model per purpose selected. guided03: 5h07m wall time, 8 individual purposes with own timing-pressure covariates.
          </div>
        </div>
      </section>"""

    # Replace first slide (duration)
    html = html[:s1] + DUR_SLIDE + html[e1:]
    # Recalculate s2/e2 since offsets shifted
    idx2 = html.find("Schedule placement: MNL own-purpose")
    s2, e2 = find_section(html, idx2)
    html = html[:s2] + SCHED_SLIDE + html[e2:]
    print(f"Replaced duration slide ({e1-s1:,} -> {len(DUR_SLIDE):,} bytes)")
    print(f"Replaced schedule slide ({e2-s2:,} -> {len(SCHED_SLIDE):,} bytes)")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
