# -*- coding: utf-8 -*-
"""Redesign pipeline slide with visual diagram + better text spacing."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find pipeline slide
idx = html.find("Seven stages: from person")
s = html.rfind("<section ", 0, idx)
depth = 1; i = s + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
slide_end = i - 1

# Also clean up trailing style/script from the previous version
trail = html[slide_end:slide_end+3000]
clean_end = slide_end
# Remove any orphaned </style> or <script> blocks
if "<style>" not in html[s:slide_end] and "</style>" in trail[:500]:
    clean_end += trail.index("</style>") + len("</style>")
if "<script>" not in html[s:slide_end] and "</script>" in trail[:1500]:
    si = trail.index("</script>")
    clean_end = max(clean_end, slide_end + si + len("</script>"))

NEW = """\
      <section class="slide" id="pipeline-slide" style="padding:2% 3.5%;">
        <h2 class="slide-kicker">Architecture</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">Seven stages: from person attributes to full trip chains</h1>
        <div class="body-area" style="gap:0.6vh; margin-top:0.3vh;">
          <div class="grid" style="grid-template-columns: 0.48fr 0.52fr; gap:0.8vw; flex:1; min-height:0;">
            <!-- Left: pipeline diagram -->
            <div style="display:flex;flex-direction:column;gap:0.3vh;justify-content:center;">
              <div style="display:flex;align-items:center;gap:1px;flex-wrap:wrap;">
                <div class="pipe-box stage-box" data-stage="0" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>0. CDAP</b><br>MNL: M/N/H</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="1" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>1. Freq</b><br>Count</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="2" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>2. Budget</b><br>Duration</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="3" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>3. Dur</b><br>Log-normal</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="4" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>4. Sched</b><br>6-period</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="5" style="flex:1.2;min-width:70px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>5. Dest+Mode</b><br>NL + ACDC</div><div class="pipe-arrow" style="width:6px;"></div>
                <div class="pipe-box stage-box" data-stage="6" style="flex:1;min-width:55px;padding:0.35em 0.2em;font-size:0.55rem;transition:all 0.4s;cursor:pointer;"><b>6. Chain</b><br>DTA-ready</div>
              </div>
              <!-- Visual diagram showing the flow -->
              <svg viewBox="0 0 360 280" style="width:100%;flex:1;min-height:0;" xmlns="http://www.w3.org/2000/svg">
                <defs><marker id="a5" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#888"/></marker></defs>
                <!-- Person -->
                <rect x="10" y="5" width="70" height="30" rx="5" fill="#1E2761"/>
                <text x="45" y="18" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Person</text>
                <text x="45" y="29" fill="#C9D3F5" font-size="5" text-anchor="middle">age, inc, veh, HH</text>
                <line x1="80" y1="20" x2="105" y2="20" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <!-- CDAP -->
                <rect x="108" y="0" width="55" height="40" rx="5" fill="#2E3B8E" opacity="0.85"/>
                <text x="135" y="14" fill="white" font-size="6" font-weight="bold" text-anchor="middle">CDAP</text>
                <text x="135" y="24" fill="#C9D3F5" font-size="5" text-anchor="middle">M / N / H</text>
                <text x="135" y="35" fill="#aaa" font-size="4.5" text-anchor="middle">3-alt MNL</text>
                <line x1="163" y1="20" x2="178" y2="20" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <!-- Stage 1 -->
                <rect x="180" y="0" width="55" height="40" rx="5" fill="#2E3B8E" opacity="0.85"/>
                <text x="207" y="14" fill="white" font-size="6" font-weight="bold" text-anchor="middle">n tours</text>
                <text x="207" y="24" fill="#C9D3F5" font-size="5" text-anchor="middle">8 purposes</text>
                <text x="207" y="35" fill="#aaa" font-size="4.5" text-anchor="middle">Poisson/NB</text>
                <line x1="235" y1="20" x2="250" y2="20" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <!-- Stage 2+3 -->
                <rect x="252" y="0" width="55" height="40" rx="5" fill="#2E3B8E" opacity="0.85"/>
                <text x="279" y="14" fill="white" font-size="6" font-weight="bold" text-anchor="middle">When +</text>
                <text x="279" y="24" fill="#C9D3F5" font-size="5" text-anchor="middle">how long</text>
                <text x="279" y="35" fill="#aaa" font-size="4.5" text-anchor="middle">Budget+dur</text>
                <!-- Arrow down to Stage 4,5 -->
                <line x1="279" y1="42" x2="279" y2="55" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="230" y="56" width="100" height="45" rx="5" fill="#C85A14" opacity="0.9"/>
                <text x="280" y="70" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Schedule</text>
                <text x="280" y="80" fill="#FFE3CE" font-size="5" text-anchor="middle">6 periods &middot; MNL</text>
                <text x="280" y="92" fill="#FFE3CE" font-size="5" text-anchor="middle">early/AM/mid/PM/eve/late</text>
                <line x1="330" y1="78" x2="345" y2="78" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="270" y="106" width="80" height="50" rx="6" fill="#C85A14" opacity="0.95"/>
                <text x="310" y="120" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Dest +</text>
                <text x="310" y="132" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Mode</text>
                <text x="310" y="145" fill="#FFE3CE" font-size="5" text-anchor="middle">371 zones &times; 5 modes</text>
                <text x="310" y="155" fill="#FFE3CE" font-size="4.5" text-anchor="middle">+ CRASH + ACDC</text>
                <!-- Arrow to chain -->
                <path d="M 270 131 L 255 131 L 255 175 L 200 175" fill="none" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="110" y="160" width="88" height="35" rx="5" fill="#2E7D6B" opacity="0.85"/>
                <text x="154" y="174" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Trip chain</text>
                <text x="154" y="186" fill="#D5F5E3" font-size="5" text-anchor="middle">home&rarr;act&rarr;home</text>
                <!-- Output arrow -->
                <line x1="110" y1="177" x2="85" y2="177" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="5" y="162" width="78" height="30" rx="5" fill="#1E2761"/>
                <text x="44" y="181" fill="white" font-size="6" font-weight="bold" text-anchor="middle">vehicle.dat</text>
                <!-- Feedback -->
                <path d="M 44 192 L 44 240 L 310 240 L 310 155" fill="none" stroke="#2E7D6B" stroke-width="0.8" stroke-dasharray="3,2"/>
                <text x="177" y="252" fill="#2E7D6B" font-size="5" text-anchor="middle">skim feedback (network times &rarr; demand)</text>
                <!-- SA annotation -->
                <rect x="180" y="210" width="130" height="18" rx="3" fill="#EBEFFA" stroke="#DCE1F0" stroke-width="0.5"/>
                <text x="245" y="222" fill="#1E2761" font-size="5" font-weight="600" text-anchor="middle">SA search across all stages</text>
              </svg>
              <div style="text-align:center;font-size:0.5rem;color:var(--text-faint);">Click a stage box or use &larr; &rarr; keys</div>
            </div>
            <!-- Right: stage descriptions -->
            <div style="display:flex;flex-direction:column;gap:0.4em;overflow-y:auto;">
              <div id="stage-desc" style="font-size:0.8rem;line-height:1.55;color:var(--text);padding:0.3em 0;">
                <span style="color:var(--text-faint);">Click a stage above to see what it does</span>
              </div>
              <div class="card" style="padding:0.4em 0.7em;font-size:0.58rem;background:var(--orange-soft);">
                <b style="color:var(--orange);">SA feature selection:</b> simulated-annealing search over candidate feature masks, guided by penalised-BIC. Analyst declares constraints (ga_constraints.py); search decides the enrichment. Uniform objective across all stages.
              </div>
            </div>
          </div>
        </div>
      </section>

<style>
.stage-box{opacity:0.4;border:2px solid var(--border);cursor:pointer;}
.stage-box.active{opacity:1;border-color:var(--primary);background:var(--surface-tint);}
.stage-box.hot-box.active{border-color:#C85A14;background:var(--orange-soft);}
</style>
<script>
(function(){
  var descs=[
    "<b>Stage 0 &mdash; CDAP.</b> Classifies every person as Mandatory, Non-mandatory, or Home. A 3-alternative MNL with person-level covariates. Determines whether the person leaves the house at all. <span style=\"color:var(--teal);font-size:0.58rem;\">Larch MNL &middot; 18 features &middot; 23k HHTS persons</span>",
    "<b>Stage 1 &mdash; Activity frequency.</b> For 8 purposes (work, edu, social, escort, shopping, maint, discr, eat), predicts how many tours the person makes. Count models (Poisson/NB) with SA-selected covariates per purpose. <span style=\"color:var(--teal);font-size:0.58rem;\">MetaCount SA &middot; ~16 features/segment &middot; 11h49m</span>",
    "<b>Stage 2 &mdash; Day budget.</b> Predicts total out-of-home time (B), start-of-day, end-of-day. Creates the temporal container all activities must fit within. <span style=\"color:var(--teal);font-size:0.58rem;\">MetaCount SA &middot; 1m44s</span>",
    "<b>Stage 3 &mdash; Activity duration.</b> Log-normal model per purpose predicts minutes per activity. Conditioned on person type and schedule constraints. Work ~8h; shopping ~40min. <span style=\"color:var(--teal);font-size:0.58rem;\">MetaCount SA &middot; 57s</span>",
    "<b>Stage 4 &mdash; Schedule placement.</b> MNL over 6 time periods places each activity in the day. Timing-pressure variables; optional random coefficients (MXL). <span style=\"color:var(--teal);font-size:0.58rem;\">SearchLibrium SA &middot; 5h07m</span>",
    "<b>Stage 5 &mdash; Destination + Mode.</b> Nested logit over 371 zones and 5 modes. CRASH skim enters every utility. ACDC agglomeration captures spatial structure. Shadow prices calibrate zone totals. <span style=\"color:var(--teal);font-size:0.58rem;\">Larch NL &middot; &rho;&sup2;=0.42-0.69 &middot; 37m30s</span>",
    "<b>Stage 6 &mdash; Trip chain.</b> Assembles tours into full trip sequences with stops. Validates schedules. Exports vehicle.dat for dynamic traffic assignment. <span style=\"color:var(--teal);font-size:0.58rem;\">Predict-only &middot; 371 zones &times; 2.3M person-days</span>"
  ];
  var boxes=document.querySelectorAll('.stage-box');
  function setStage(n){
    boxes.forEach(function(b,i){
      b.classList.toggle('active',i<=n);
      if(b.dataset.stage=='5')b.classList.toggle('hot-box',i<=n);
    });
    document.getElementById('stage-desc').innerHTML=descs[n]||'';
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

html = html[:s] + NEW + html[clean_end:]
print(f"Pipeline slide: {clean_end-s:,} -> {len(NEW):,} bytes")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
