# -*- coding: utf-8 -*-
"""Remove gallery slide, fix pipeline SVG arrows, fix explainers right panel."""
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── 1. Remove gallery slide ──
g = html.find("Validation &amp; diagnostics gallery")
if g > 0:
    s = html.rfind("<section ", 0, g)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    html = html[:s] + html[i:]
    print(f"Removed gallery slide ({i-s} bytes)")

# ── 2. Fix pipeline slide SVG - add clear schedule→dest arrow ──
p_idx = html.find("Seven stages: from person")
if p_idx > 0:
    s = html.rfind("<section ", 0, p_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    # Fix the SVG: add clear arrow from schedule to dest, and make dest→mode clearer
    slide = html[s:e]
    
    # Fix the schedule-to-dest path: "Stage 5" should have a clear incoming arrow from Stage 4 (schedule)
    # Currently the SVG has schedule at y=56 and dest at y=106, but no connecting arrow
    # The schedule box is: <rect x="230" y="56" ...> and dest is at <rect x="270" y="106" ...>
    # Add a vertical arrow from schedule to dest
    
    old_sched_rect = '<rect x="230" y="56" width="100" height="45" rx="5" fill="#C85A14" opacity="0.9"/>'
    new_sched_block = '''<line x1="252" y1="20" x2="252" y2="40" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="230" y="41" width="100" height="28" rx="5" fill="#1E2761" opacity="0.8"/><text x="280" y="55" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Schedule</text><text x="280" y="65" fill="#C9D3F5" font-size="5" text-anchor="middle">6 periods MNL</text>
                <line x1="280" y1="69" x2="280" y2="82" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>
                <rect x="250" y="83" width="120" height="50" rx="6" fill="#C85A14" opacity="0.95"/><text x="310" y="98" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Dest + Mode</text><text x="310" y="110" fill="#FFE3CE" font-size="5" text-anchor="middle">371 zones &times; 5 modes</text><text x="310" y="122" fill="#FFE3CE" font-size="4.5" text-anchor="middle">+ CRASH + ACDC</text>'''
    
    old_block = '<line x1="279" y1="42" x2="279" y2="55" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>\n                <rect x="230" y="56" width="100" height="45" rx="5" fill="#C85A14" opacity="0.9"/>\n                <text x="280" y="70" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Schedule</text>\n                <text x="280" y="80" fill="#FFE3CE" font-size="5" text-anchor="middle">6 periods &middot; MNL</text>\n                <text x="280" y="92" fill="#FFE3CE" font-size="5" text-anchor="middle">early/AM/mid/PM/eve/late</text>\n                <line x1="330" y1="78" x2="345" y2="78" stroke="#1C7293" stroke-width="1" marker-end="url(#a5)"/>\n                <rect x="270" y="106" width="80" height="50" rx="6" fill="#C85A14" opacity="0.95"/>\n                <text x="310" y="120" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Dest +</text>\n                <text x="310" y="132" fill="white" font-size="6" font-weight="bold" text-anchor="middle">Mode</text>\n                <text x="310" y="145" fill="#FFE3CE" font-size="5" text-anchor="middle">371 zones &times; 5 modes</text>\n                <text x="310" y="155" fill="#FFE3CE" font-size="4.5" text-anchor="middle">+ CRASH + ACDC</text>\n                <!-- Arrow to chain -->\n                <path d="M 270 131 L 255 131 L 255 175 L 200 175"'
    
    new_path = '''<path d="M 250 108 L 235 108 L 235 175 L 200 175"'''
    
    slide = slide.replace(old_block, new_sched_block + '\n                ' + new_path)
    
    html = html[:s] + slide + html[e:]
    print(f"Fixed pipeline SVG arrows")

# ── 3. Fix explainers slide - use right panel with detail content ──
ex_idx = html.find("What each stage estimates")
if ex_idx > 0:
    s = html.rfind("<section ", 0, ex_idx)
    depth = 1; i = s + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e = i - 1
    
    slide = html[s:e]
    
    # Replace the empty right panel with a dynamic detail panel
    old_right = '<div id="exp-detail" style="flex:0.9; display:flex; align-items:center; justify-content:center; font-size:0.76rem; color:var(--text-faint); text-align:center; padding:1em;">\n              Click a stage to see what the search decides\n            </div>'
    
    new_right = '''<div id="exp-detail" style="flex:1; display:flex; flex-direction:column; gap:0.4em; padding:0.2em 0.5em; overflow-y:auto;">
              <div class="card" style="padding:0.4em 0.6em;font-size:0.72rem;background:var(--surface-tint);"><b style="color:var(--primary);">Model form:</b> <span id="ed-form">-</span></div>
              <div class="card" style="padding:0.4em 0.6em;font-size:0.72rem;background:var(--surface-tint);"><b style="color:var(--orange);">Search decides:</b> <span id="ed-search">-</span></div>
              <div class="card" style="padding:0.4em 0.6em;font-size:0.72rem;background:var(--surface-tint);"><b>Structural:</b> <span id="ed-struct">-</span></div>
              <div class="card" style="padding:0.4em 0.6em;font-size:0.66rem;background:var(--surface-tint);"><b>Engine &amp; wall:</b> <span id="ed-engine">-</span></div>
            </div>'''
    
    slide = slide.replace(old_right, new_right)
    
    # Update the JS to populate the detail panel on click
    old_js = '''items.forEach(function(el,i){
      el.addEventListener('click',function(){
        items.forEach(function(e){e.classList.remove('show');});
        for(var j=0;j<=i;j++)items[j].classList.add('show');
      });
    });'''
    
    new_js = '''var details=[
      {f:'MNL over H/N/M',s:'Which person-type, age, income, household variables enter day-pattern utilities',t:'3-alternative MNL',e:'Larch MNL &middot; 18 features &middot; 23k HHTS persons'},
      {f:'Count models (Poisson/NB) per purpose',s:'Which demographics and accessibility measures predict each purpose frequency',t:'Ordered-logit baseline',e:'MetaCount SA &middot; ~16 feat/seg &middot; 11h49m'},
      {f:'Regression on start, end, total B',s:'Which person attributes shape the daily time container',t:'OLS with lognormal LL',e:'MetaCount SA &middot; 1m44s'},
      {f:'Log-normal per purpose',s:'Which covariates shift the duration distribution',t:'Log-normal likelihood',e:'MetaCount SA &middot; 57s'},
      {f:'MNL over 6 time periods',s:'Timing-pressure variables; which coefficients get random parameters',t:'6-period MNL',e:'SearchLibrium SA &middot; 5h07m'},
      {f:'Nested logit over 5 modes',s:'LOS interactions, person-mode affinities, CBD/transit terms',t:'Mu_Root=1, ASCs, CRASH force-included',e:'Larch NL &middot; &rho;&sup2;=0.42-0.63'},
      {f:'MNL over 371 SA2 zones',s:'Attractor types, person&times;distance, ACDC form, home-pull',t:'size_term, logsum, c_crash, c_log_dist',e:'Larch flat-CA &middot; &rho;&sup2;=0.41-0.69'},
      {f:'Assembly only - no estimation',s:'Intermediate stops on detours. Validates, exports vehicle.dat',t:'Predict-only',e:'371 zones &times; 2.3M person-days'}
    ];
    items.forEach(function(el,i){
      el.addEventListener('click',function(){
        items.forEach(function(e){e.classList.remove('show');});
        for(var j=0;j<=i;j++)items[j].classList.add('show');
        var d=details[i]||{};
        document.getElementById('ed-form').innerHTML=d.f||'-';
        document.getElementById('ed-search').innerHTML=d.s||'-';
        document.getElementById('ed-struct').innerHTML=d.t||'-';
        document.getElementById('ed-engine').innerHTML=d.e||'-';
      });
    });'''
    
    slide = slide.replace(old_js, new_js)
    
    # Also initialize the first detail
    old_init = 'items[0].classList.add(\'show\');'
    new_init = '''items[0].classList.add('show');
    var d0=details[0];document.getElementById('ed-form').innerHTML=d0.f;document.getElementById('ed-search').innerHTML=d0.s;document.getElementById('ed-struct').innerHTML=d0.t;document.getElementById('ed-engine').innerHTML=d0.e;'''
    
    slide = slide.replace(old_init, new_init)
    
    html = html[:s] + slide + html[e:]
    print(f"Fixed explainers right panel")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes.")
