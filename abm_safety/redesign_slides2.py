# -*- coding: utf-8 -*-
"""Redesign pipeline, model explainers, and population/landuse slides."""
import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. REDESIGN PIPELINE SLIDE (3) - sequential reveal, highlight boxes ──
# Find slide 3 section
s3_comment = html.find("<!-- 3. PIPELINE -->")
s3_sec = html.find("<section ", s3_comment)
depth = 1; i = s3_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
s3_end = i - 1

# Also remove the old <style> and <script> that follow
trail = html[s3_end:s3_end+2000]
extra_end = s3_end
if "</style>" in trail[:500]:
    extra_end = s3_end + trail.index("</style>") + len("</style>")
if "<script>" in trail[:1000]:
    sc_idx = trail.index("</script>")
    extra_end = max(extra_end, s3_end + sc_idx + len("</script>"))

NEW_PIPELINE = """\
      <section class="slide" id="pipeline-slide" style="padding:2.5% 4%;">
        <h2 class="slide-kicker">Architecture</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.5vw,2rem);">Seven stages: from person attributes to full trip chains</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div style="display:flex; align-items:center; gap:3px; flex-wrap:wrap; font-size:0.6rem;">
            <div class="pipe-box stage-box" data-stage="0" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>0. CDAP</b><br><small>MNL: M / N / H</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="1" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>1. Frequency</b><br><small>Count models</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="2" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>2. Budget</b><br><small>Day duration</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="3" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>3. Duration</b><br><small>Log-normal</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="4" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>4. Schedule</b><br><small>6-period MNL</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="5" style="flex:1;min-width:100px;padding:0.4em 0.3em;transition:all 0.4s;"><b>5. Dest + Mode</b><br><small>NL + ACDC + Crash</small></div><div class="pipe-arrow" style="width:10px;"></div>
            <div class="pipe-box stage-box" data-stage="6" style="flex:1;min-width:80px;padding:0.4em 0.3em;transition:all 0.4s;"><b>6. Trip Chain</b><br><small>DTA-ready</small></div>
          </div>
          <div style="flex:1; display:flex; align-items:center; justify-content:center; min-height:15vh;">
            <div id="stage-desc" style="font-size:1.05rem; line-height:1.65; max-width:75ch; text-align:center; color:var(--text); transition:all 0.4s; padding:1em;">
              <span style="color:var(--text-faint);">Use &larr; &rarr; keys or click a stage box above</span>
            </div>
          </div>
          <div style="display:flex; gap:0.8em; justify-content:center; font-size:0.66rem; color:var(--text-muted);">
            <span id="stage-counter"></span>
          </div>
        </div>
      </section>

<style>
.stage-box{opacity:0.4;border:2px solid var(--border);}
.stage-box.active{opacity:1;border-color:var(--primary);background:var(--surface-tint);}
.stage-box.hot-box.active{border-color:#C85A14;background:var(--orange-soft);}
</style>
<script>
(function(){
  var descs=[
    "<b>Stage 0 &mdash; Coordinated Daily Activity Pattern.</b><br>Classifies every one of the 3.8 million synthetic persons into a daily pattern: <b>Mandatory</b> (has at least one work or education tour), <b>Non-mandatory</b> (discretionary activities only), or <b>Home</b> (stays home all day). A 3-alternative multinomial logit model with person-level covariates selected by SA search. This stage determines whether the person even leaves the house.",
    "<b>Stage 1 &mdash; Activity frequency.</b><br>For each of 8 purposes (work, education, social, escort, shopping, maintenance, discretionary, eating-out), predicts <b>how many tours</b> the person makes. Count models (Poisson / Negative Binomial) with SA-selected covariates per purpose. A full-time worker might make 1 work tour + 1 shopping tour; a retiree might make 2 social tours + 1 eating-out tour.",
    "<b>Stage 2 &mdash; Day budget.</b><br>Predicts the person's <b>total out-of-home time</b> (B), start-of-day, and end-of-day. This creates the temporal container that all activities must fit within. A person with B=8 hours has a very different feasible schedule from someone with B=2 hours.",
    "<b>Stage 3 &mdash; Activity duration.</b><br>For each activity the person makes, predicts <b>how many minutes</b> it lasts. Log-normal models per purpose, conditioned on person type, day position, and remaining budget. Work tours average ~8 hours; shopping tours ~40 minutes.",
    "<b>Stage 4 &mdash; Schedule placement.</b><br>Places each activity into one of <b>6 time periods</b> (early, AM peak, midday, PM peak, evening, late). Multinomial logit with timing-pressure variables. Work goes in AM/midday; eating-out in evening; escort in AM/PM peaks. Respects the budget and duration constraints from Stages 2&ndash;3.",
    "<b>Stage 5 &mdash; Destination + Mode choice.</b><br>The spatial core of the model. <b>Nested logit</b> over 371 SA2 zones and 5 modes (Drive Alone, Shared Ride, Walk, Bike, Transit). Crash-frequency skims enter every utility. ACDC agglomeration and competition terms capture spatial structure. Purpose-specific size terms weight zone attractiveness. Shadow prices calibrate zone-level totals to observed shares.",
    "<b>Stage 6 &mdash; Trip chain assembly.</b><br>Connects activities into <b>full trip sequences</b> with intermediate stops. Validates schedules (no time travel: durations must be non-negative, returns must reach home). Exports to vehicle.dat format ready for dynamic traffic assignment and network performance evaluation."
  ];
  var boxes=document.querySelectorAll('.stage-box');
  function setStage(n){
    boxes.forEach(function(b,i){
      b.classList.toggle('active',i<=n);
      if(b.dataset.stage=='5')b.classList.toggle('hot-box',i<=n);
    });
    document.getElementById('stage-desc').innerHTML=descs[n]||'';
    document.getElementById('stage-counter').textContent='Stage '+(n)+' of 6';
  }
  boxes.forEach(function(b){b.addEventListener('click',function(){setStage(parseInt(b.dataset.stage));});});
  setStage(0);
  document.addEventListener('keydown',function(e){
    var cur=0;boxes.forEach(function(b,i){if(b.classList.contains('active'))cur=i;});
    if(e.key==='ArrowRight')setStage(Math.min(cur+1,6));
    else if(e.key==='ArrowLeft')setStage(Math.max(cur-1,0));
  });
})();
</script>"""

html = html[:s3_sec] + NEW_PIPELINE + html[extra_end:]
print(f"  Pipeline slide: {extra_end-s3_sec:,} -> {len(NEW_PIPELINE):,} bytes")

# ── 2. REDESIGN MODEL EXPLAINERS (3b) - sequential reveal ──
s3b_comment = html.find("<!-- 3b. MODEL EXPLAINERS -->")
s3b_sec = html.find("<section ", s3b_comment)
depth = 1; i = s3b_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
s3b_end = i - 1

NEW_EXPLAINERS = """\
      <section class="slide" id="explainer-slide" style="padding:2.5% 4%;">
        <h2 class="slide-kicker">Architecture &middot; estimation details</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.5vw,2rem);">What each stage estimates &mdash; and what the SA search decides</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div style="display:flex; flex:1; gap:0.8vw;">
            <div style="flex:1; display:flex; flex-direction:column; gap:0.3em; overflow-y:auto; padding-right:0.3em;">
              <div class="exp-item" data-step="0"><b>0. CDAP &mdash; Daily pattern.</b> MNL over H/N/M. <em>Search decides:</em> which person-type, age, income, household, and accessibility variables enter the utilities. Structural: 3-alternative MNL. <span style="color:var(--teal);font-size:0.64rem;">Larch MNL &middot; 18 features selected &middot; 23,189 HHTS persons</span></div>
              <div class="exp-item" data-step="1"><b>1. Activity frequency.</b> Count models (Poisson/NB) per purpose (8 purposes &times; ptype segments). <em>Search decides:</em> which demographics and accessibility measures predict each purpose's frequency. Structural: ordered-logit baseline. <span style="color:var(--teal);font-size:0.64rem;">MetaCount SA-Guided &middot; ~16 features per segment &middot; 11h49m wall</span></div>
              <div class="exp-item" data-step="2"><b>2. Day budget.</b> Regression on start-of-day, end-of-day, total out-of-home time B. <em>Search decides:</em> which person attributes shape the daily time container. Structural: OLS with lognormal LL. <span style="color:var(--teal);font-size:0.64rem;">MetaCount SA-Guided &middot; 1m44s wall</span></div>
              <div class="exp-item" data-step="3"><b>3. Activity duration.</b> Log-normal per purpose. <em>Search decides:</em> which covariates shift the duration distribution. Structural: log-normal likelihood. <span style="color:var(--teal);font-size:0.64rem;">MetaCount SA-Guided &middot; 57s wall</span></div>
              <div class="exp-item" data-step="4"><b>4. Schedule placement.</b> MNL over 6 time periods per purpose. <em>Search decides:</em> timing-pressure variables; which coefficients get random parameters (MXL). Structural: 6-period MNL. <span style="color:var(--teal);font-size:0.64rem;">SearchLibrium SA &middot; 5h07m wall</span></div>
              <div class="exp-item" data-step="5"><b>5a. Tour mode choice.</b> Nested logit over 5 modes. <em>Search decides:</em> LOS interactions, person-mode affinities, CBD/transit terms. Structural: Mu_Root=1, ASCs, CRASH force-included. <span style="color:var(--teal);font-size:0.64rem;">Larch NL &middot; &rho;&sup2;=0.42&ndash;0.63 &middot; JAX/numba engine guard</span></div>
              <div class="exp-item" data-step="6"><b>5b. Destination choice.</b> MNL over 371 SA2 zones. <em>Search decides:</em> attractor types, person&times;distance interactions, ACDC form (generic vs themed), home-pull gravity. Structural: size_term, logsum, c_crash, c_log_dist. <span style="color:var(--teal);font-size:0.64rem;">Larch flat-CA &middot; &rho;&sup2;=0.41&ndash;0.69 &middot; shadow-price calibration</span></div>
              <div class="exp-item" data-step="7"><b>6. Trip chain + export.</b> Assembly only &mdash; no estimation. Intermediate stops on tour detours. Validates schedules, repairs negative durations, exports vehicle.dat. <span style="color:var(--teal);font-size:0.64rem;">Predict-only &middot; 371 zones &times; 2.3M person-days</span></div>
            </div>
            <div style="width:2px;background:var(--border);align-self:stretch;"></div>
            <div id="exp-detail" style="flex:0.9; display:flex; align-items:center; justify-content:center; font-size:0.76rem; color:var(--text-faint); text-align:center; padding:1em;">
              Click a stage to see what the search decides
            </div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.62rem;">Every stage keeps a structural backbone the search cannot remove. The SA search only decides the <em>enrichment</em> around it &mdash; never the core model form.</p>
        </div>
      </section>

<style>
.exp-item{padding:0.35em 0.5em;border-left:3px solid var(--border);cursor:pointer;transition:all 0.3s;font-size:0.7rem;line-height:1.4;opacity:0.5;}
.exp-item.show{opacity:1;border-left-color:var(--teal);}
.exp-item b{color:var(--primary);}
</style>
<script>
(function(){
  var items=document.querySelectorAll('.exp-item');
  items.forEach(function(el,i){
    el.addEventListener('click',function(){
      items.forEach(function(e){e.classList.remove('show');});
      for(var j=0;j<=i;j++)items[j].classList.add('show');
    });
  });
  // Show first by default
  items[0].classList.add('show');
})();
</script>"""

html = html[:s3b_sec] + NEW_EXPLAINERS + html[s3b_end:]
print(f"  Explainers slide: {s3b_end-s3b_sec:,} -> {len(NEW_EXPLAINERS):,} bytes")

# ── 3. REDESIGN POPULATION/LANDUSE SLIDE (4) ──
s4_comment = html.find("<!-- 4. POPULATION + ZONES -->")
s4_sec = html.find("<section ", s4_comment)
depth = 1; i = s4_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
s4_end = i - 1

NEW_POP = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Population &amp; Geography</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">Land-use data: what each of the 371 SA2 zones contains</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat hero"><div class="value">3.82M</div><div class="label">synthetic residents</div></div>
            <div class="stat"><div class="value">371</div><div class="label">SA2 zones (ABS 2021 boundaries)</div></div>
            <div class="stat"><div class="value">8</div><div class="label">person types synthesised</div></div>
            <div class="stat"><div class="value">2.3M</div><div class="label">person-day plans generated</div></div>
          </div>
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.3em; font-size:0.8rem;">Employment (by industry, per zone)</h3>
              <div style="font-size:0.66rem; line-height:1.55;">
                <b>TOTEMP</b> &mdash; total employment (sum of sectoral below)<br>
                <b>FPSEMPN</b> &mdash; finance &amp; professional services<br>
                <b>HEREMPN</b> &mdash; health, education, public admin<br>
                <b>RETEMPN</b> &mdash; retail trade &amp; hospitality<br>
                <b>AGREMPN</b> &mdash; agriculture, forestry, fishing<br>
                <b>MWTEMPN</b> &mdash; manufacturing, wholesale trade<br>
                <b>OTHEMP</b> &mdash; other services &amp; industries<br>
                <span style="color:var(--text-faint);">Source: ABS Census 2021 employment by industry by SA2</span>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.3em; font-size:0.8rem;">Education (enrolment places, per zone)</h3>
              <div style="font-size:0.66rem; line-height:1.55;">
                <b>Primary</b> &mdash; primary school enrolment<br>
                <b>Secondary</b> &mdash; secondary school enrolment<br>
                <b>Education</b> &mdash; tertiary (TAFE + university)<br>
                <b>HSENROLL</b> &mdash; high school supplementary count<br>
                <b>Primary_total_priv</b> &mdash; private primary places<br>
                <b>Secondary_total_priv</b> &mdash; private secondary places<br>
                <span style="color:var(--text-faint);">Source: ABS Census + Queensland education department</span>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.3em; font-size:0.8rem;">Land-use (OSM features, per zone)</h3>
              <div style="font-size:0.66rem; line-height:1.55;">
                <b>Residential</b> &mdash; residential parcel acreage<br>
                <b>Commercial</b> &mdash; commercial/industrial floorspace<br>
                <b>Retail</b> &mdash; retail &amp; shopping precincts<br>
                <b>Entertainment</b> &mdash; entertainment &amp; recreation<br>
                <b>EATOUT</b> &mdash; restaurants, cafes, pubs, bars<br>
                <b>Healthcare</b> &mdash; hospitals, clinics, pharmacies<br>
                <b>Recreation / sport / parks</b> &mdash; OSM POI counts<br>
                <b>Kindergarten / childcare</b> &mdash; OSM POI counts<br>
                <span style="color:var(--text-faint);">Source: OSM POI extraction + EMME network land-use model + ABS Census apportionment</span>
              </div>
            </div>
            <div class="card" style="padding:0.6em 0.8em;">
              <h3 style="margin:0 0 0.3em; font-size:0.8rem;">Demographics &amp; accessibility</h3>
              <div style="font-size:0.66rem; line-height:1.55;">
                <b>TOTPOP</b> &mdash; total population<br>
                <b>TOTHH</b> &mdash; total households<br>
                <b>AGE0519</b> &mdash; school-age population (5&ndash;19)<br>
                <b>SA2 area</b> &mdash; zone area (km&sup2;)<br>
                <b>Auto / walk / bike / transit accessibility</b> &mdash; cumulative opportunity measures within 30 min<br>
                <b>Crash accessibility</b> &mdash; home-zone crash exposure<br>
                <span style="color:var(--text-faint);">Source: ABS Census + network skims + SPF predictions</span>
              </div>
            </div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.62rem;">Landuse rebuilt from real data: EMME 2062-zone network model where available, ABS Census apportionment for zones outside it. No nearest-neighbour guessing. 371 zones covering all SEQ SA2s from ABS 2021 boundaries.</p>
        </div>
      </section>"""

html = html[:s4_sec] + NEW_POP + html[s4_end:]
print(f"  Pop/landuse slide: {s4_end-s4_sec:,} -> {len(NEW_POP):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
