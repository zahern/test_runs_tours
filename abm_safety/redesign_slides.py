# -*- coding: utf-8 -*-
"""Redesign slide 2 (Why safety) and slide 3 (Pipeline) - reveal-based, bigger text, clear stages, remove bandit."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Redesign Slide 2: WHY THIS MATTERS ──
old_s2_start = html.find("<!-- 2. WHY THIS MATTERS -->")
old_s2_sec = html.find("<section ", old_s2_start)
depth = 1; i = old_s2_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
old_s2_end = i - 1

NEW_S2 = """\
      <section class="slide" style="padding:3.5% 5%;">
        <h2 class="slide-kicker">Motivation</h2>
        <h1 class="slide-title" style="font-size:clamp(1.6rem,3vw,2.4rem);">Safety shapes travel demand &mdash; it should enter the model that way</h1>
        <div class="body-area" style="gap:2vh; justify-content:center;">
          <div class="reveal-list" style="font-size:0.95rem; line-height:1.7; max-width:80ch;">
            <div class="reveal-item" data-step="1"><b>Conventional ABMs</b> compute crash risk <em>after</em> demand is assigned to the network. Safety is a network performance metric, not a behavioural input. Mode choice, destination choice, and route choice are blind to crash exposure.</div>
            <div class="reveal-item" data-step="2"><b>This framework</b> embeds expected crash frequency directly into mode-choice and destination-choice utilities. An origin&ndash;destination crash-frequency matrix (CRASH) enters every tour's utility function <em>alongside</em> travel time, distance, and land-use attractiveness.</div>
            <div class="reveal-item" data-step="3"><b>Why it matters.</b> Empirical evidence consistently shows that crash risk shifts active-mode uptake, route selection, escort behaviour, and departure-time choice. Children, the elderly, and low-income workers are most affected. Omitting safety systematically <em>misestimates</em> demand for the travellers least able to absorb unsafe infrastructure.</div>
            <div class="reveal-item" data-step="4"><b>How it works.</b> Safety Performance Functions (SPFs) fitted by metacountregressor predict segment-level expected crashes. The SA search over 6&times;5 road-stereotype&times;crash-type cells evaluates ~780&ndash;5,011 candidate count models per cell, selecting the best-BIC specification automatically. Empirical-Bayes blending ensures expected totals reproduce observed totals. Crashes are accumulated along inter-zonal shortest paths into the OD CRASH skim.</div>
            <div class="reveal-item" data-step="5"><b>The search decides.</b> CRASH is force-included in every Stage-5 utility specification by the constraint architecture. The SA search decides its coefficient sign, magnitude, and interactions &mdash; not the analyst. If the data does not support a safety effect, the coefficient shrinks toward zero. If it does, the model captures it.</div>
          </div>
          <div class="reveal-buttons" style="display:flex; gap:0.5em; justify-content:center; align-self:center;">
            <button onclick="revealStep(0)" style="cursor:pointer;padding:6px 16px;background:var(--primary);color:#fff;border:none;border-radius:4px;font-size:0.74rem;">Reset</button>
            <button onclick="revealNext()" style="cursor:pointer;padding:6px 16px;background:var(--teal);color:#fff;border:none;border-radius:4px;font-size:0.74rem;">Reveal next</button>
            <span id="reveal-counter" style="font-size:0.64rem;color:var(--text-muted);align-self:center;"></span>
          </div>
        </div>
      </section>

<style>
.reveal-item{opacity:0.12;transition:opacity 0.5s;padding:0.3em 0;border-left:3px solid var(--border);padding-left:1em;}
.reveal-item.show{opacity:1;border-left-color:var(--teal);}
.reveal-item b{color:var(--primary);}
</style>
<script>
var revealStep=function(n){
  document.querySelectorAll('.reveal-item').forEach(function(el,i){
    el.classList.toggle('show', i<n);
  });
  document.getElementById('reveal-counter').textContent=n+' / '+document.querySelectorAll('.reveal-item').length;
};
var revealNext=function(){
  var items=document.querySelectorAll('.reveal-item'), n=0;
  items.forEach(function(el,i){if(el.classList.contains('show'))n=i+1;});
  revealStep(Math.min(n+1,items.length));
};
revealStep(1);
</script>"""

html = html[:old_s2_sec] + NEW_S2 + html[old_s2_end:]
print(f"  Slide 2: {old_s2_end-old_s2_sec:,} -> {len(NEW_S2):,} bytes")

# ── 2. Redesign Slide 3: PIPELINE - clear stages, remove bandit ──
old_s3_start = html.find("<!-- 3. PIPELINE -->")
old_s3_sec = html.find("<section ", old_s3_start)
depth = 1; i = old_s3_sec + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
old_s3_end = i - 1

NEW_S3 = """\
      <section class="slide" style="padding:2.8% 4%;">
        <h2 class="slide-kicker">Architecture</h2>
        <h1 class="slide-title" style="font-size:clamp(1.5rem,2.8vw,2.2rem);">What each stage does: from person to trip chain</h1>
        <div class="body-area" style="gap:1.2vh;">
          <div class="pipeline-row">
            <div class="pipe-box"><b>STAGE 0</b>Daily pattern<br><small>MNL: Mandatory / Non-mandatory / Home</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box"><b>STAGE 1</b>Activity frequency<br><small>Count models: how many tours per purpose</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box"><b>STAGE 2</b>Day budget<br><small>Total out-of-home time &amp; start/end</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box"><b>STAGE 3</b>Duration<br><small>Log-normal: minutes per activity</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box"><b>STAGE 4</b>Schedule<br><small>MNL over 6 time periods</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box hot"><b>STAGE 5</b>Destination + Mode<br><small>NL over 371 zones &times; 5 modes + crash skim</small></div><div class="pipe-arrow"></div>
            <div class="pipe-box"><b>STAGE 6</b>Trip chain<br><small>Assemble trips &rarr; DTA export</small></div>
          </div>
          <div style="display:flex; gap:1vw;">
            <div class="reveal-list" style="flex:1; font-size:0.78rem; line-height:1.55;">
              <div class="reveal-item2" data-step="0"><b>Stage 0 &mdash; CDAP.</b> Classifies every person as Mandatory (work/study), Non-mandatory (discretionary), or Home. 3-alternative MNL. A homebody generates zero tours; a mandatory pattern generates at least one work or education tour.</div>
              <div class="reveal-item2" data-step="1"><b>Stage 1 &mdash; Activity frequency.</b> For each of 8 purposes (work, edu, social, escort, shopping, maintenance, discretionary, eating-out), predicts how many tours the person makes. Ordered logit / count models with SA-selected covariates.</div>
              <div class="reveal-item2" data-step="2"><b>Stage 2 &mdash; Day budget.</b> Predicts the person's total out-of-home time (B), start-of-day, and end-of-day. Constrains the schedule: activities must fit within the available budget.</div>
              <div class="reveal-item2" data-step="3"><b>Stage 3 &mdash; Duration.</b> Log-normal model per purpose predicts how many minutes each activity lasts. Conditioned on person type, day position, and schedule constraints.</div>
              <div class="reveal-item2" data-step="4"><b>Stage 4 &mdash; Schedule placement.</b> MNL over 6 time periods (early, AM peak, midday, PM peak, evening, late). Places each activity in the day, respecting the time budget and duration constraints from Stages 2&ndash;3.</div>
              <div class="reveal-item2" data-step="5"><b>Stage 5 &mdash; Destination + Mode.</b> The core spatial choice. Nested logit over 371 SA2 zones and 5 modes (DA, SR, Walk, Bike, Transit). CRASH skim enters every utility. ACDC agglomeration and competition terms capture spatial structure. Shadow prices calibrate zone-level totals to observed shares.</div>
              <div class="reveal-item2" data-step="6"><b>Stage 6 &mdash; Trip chain.</b> Assembles tours into full trip sequences with intermediate stops. Validates schedules (no negative durations, home returns). Exports to vehicle.dat for dynamic traffic assignment.</div>
            </div>
          </div>
          <div style="display:flex; gap:1vw;">
            <div class="card" style="flex:1; padding:0.6em 0.9em;"><b style="color:var(--primary);">SA feature selection across all stages.</b> A simulated-annealing search over candidate feature masks, guided by a penalised-BIC objective. The analyst declares constraints (what is forbidden) through ga_constraints.py; the search decides the enrichment (what is included). Per-stage fitness evaluations use the same objective function for uniform selection pressure.</div>
            <div class="card" style="flex:1; background:var(--orange-soft); padding:0.6em 0.9em;"><b style="color:var(--orange);">Safety skims enter Stage 5.</b> Empirical-Bayes crash-prediction functions feed a crash-frequency OD skim used directly in mode- and destination-choice utilities. CRASH is force-included by the constraint architecture; the SA search decides its coefficient and interactions.</div>
          </div>
          <p class="sub" style="text-align:center; font-size:0.7rem;">Estimated on the Queensland Household Travel Survey (HHTS). Applied to the full SEQ synthetic population (3.8M persons, 371 SA2 zones).</p>
        </div>
      </section>

<style>
.reveal-item2{opacity:0.15;transition:opacity 0.45s;padding:0.25em 0;border-left:3px solid var(--border);padding-left:0.8em;margin-bottom:0.15em;}
.reveal-item2.show{opacity:1;border-left-color:var(--teal);}
.reveal-item2 b{color:var(--primary);}
</style>
<script>
(function(){
  var ri=document.querySelectorAll('.reveal-item2'), step=0;
  function show(n){ri.forEach(function(el,i){el.classList.toggle('show',i<n);});step=n;}
  show(0);
  document.addEventListener('keydown',function(e){if(e.key==='ArrowRight')show(Math.min(step+1,ri.length));else if(e.key==='ArrowLeft')show(Math.max(step-1,0));});
  var obs=new MutationObserver(function(){var el=document.querySelector('.reveal-item2');if(el&&el.offsetParent!==null)show(step||0);});
  var d=document.getElementById('deck');if(d)obs.observe(d,{attributes:true,attributeFilter:['style']});
})();
</script>"""

html = html[:old_s3_sec] + NEW_S3 + html[old_s3_end:]
print(f"  Slide 3: {old_s3_end-old_s3_sec:,} -> {len(NEW_S3):,} bytes")

# ── 3. Remove "bandit" mentions from slide 3b and elsewhere ──
html = html.replace("SA-bandit feature selection", "SA feature selection")
html = html.replace("SA-Bandit feature selection", "SA feature selection")
html = html.replace("SA-Bandit search", "SA search")
html = html.replace("SA-bandit selected", "SA-selected")
html = html.replace("SA-bandit", "SA")
html = html.replace("BanditGuidedSA", "SA-Guided")
html = html.replace("BanditSA", "SA")
html = html.replace("BIC-guided bandit", "BIC-guided SA")
html = html.replace("bandit outer loop", "SA outer loop")
html = html.replace("bandit search", "SA search")
html = html.replace("bandit", "SA")
# Fix the SVG text
html = html.replace("SA-Bandit feature selection across all stages", "SA feature selection across all stages")
html = html.replace("SA &times; simulated-annealing", "SA &times; simulated-annealing")

# ── 4. Also update slide 10c (SEARCH ARCHITECTURE) bandit mentions ──
html = html.replace("BANDIT OUTER LOOP", "SA OUTER LOOP")
html = html.replace("bandit outer loop", "SA outer loop")
html = html.replace("Bandit-based", "SA-based")

# ── 5. Clean "bandit" from remaining model explainer text ──
html = html.replace("· BanditGuidedSA", "· SA-Guided")
html = html.replace("· SA-GuidedSA", "· SA-Guided")

print(f"  Remaining 'bandit': {html.count('bandit')}")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
