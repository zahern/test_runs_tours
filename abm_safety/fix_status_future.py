html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Remove slide 34 "Where this stands"
idx34=3986938
s34=html.rfind('<section ',0,idx34)
depth=1;i=s34+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e34=i-1
html=html[:s34]+html[e34:]
print(f'Removed slide 34 ({e34-s34} bytes)')

# 2. Fix nested slide 39 inside 38
idx38=html.find('Model performance at a glance')
s38=html.rfind('<section ',0,idx38)
depth=1;i=s38+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e38=i-1
# Find nested
nested_s=html.find('<section ',s38+1)
if nested_s>0 and nested_s<e38:
    depth=1;j=nested_s+1
    while depth>0 and j<e38:
        if html[j:j+9]=='<section ':depth+=1
        elif html[j:j+10]=='</section>':depth-=1
        j+=1
    nested_e=j-1
    nested=html[nested_s:nested_e]
    html=html[:nested_s]+html[nested_e:]
    e38_new=e38-(nested_e-nested_s)
    html=html[:e38_new]+'\n\n'+nested+html[e38_new:]
    print(f'Fixed nested slide 39')

# 3. Find closing/Questions slide and add future work slide before it
closing=html.find('Questions &amp; Discussion')
if closing<0:
    closing=html.find('13. CLOSING')
if closing>0:
    # Find the section before closing
    prev_sec=html.rfind('</section>',0,closing)
    
    FUTURE='''\
      <section class="slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">Future Work</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">Counterfactual testing becomes possible</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.7vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--good);">
              <h3 style="margin:0 0 0.3em; font-size:0.84rem;">1. Safety counterfactuals</h3>
              <p style="font-size:0.7rem;line-height:1.5;margin:0;">Because crash risk is an <b>explicit demand input</b>, the model can answer "what if" questions. Run baseline (CRASH ON) and counterfactual (CRASH OFF) with identical coefficients, then compare: mode shares, destination patterns, trip distances, crash exposure per person-day. The nosafety run (3h54m, 4.4M trips) provides the counterfactual arm.</p>
            </div>
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--orange);">
              <h3 style="margin:0 0 0.3em; font-size:0.84rem;">2. Infrastructure scenarios</h3>
              <p style="font-size:0.7rem;line-height:1.5;margin:0;">Modify the CRASH skim to reflect <b>proposed safety treatments</b>: reduced speeds, protected bike lanes, intersection upgrades, pedestrian crossings. Re-run the model with the modified skim. Quantify: which demographic groups benefit most? How do mode shares shift? Which corridors show the largest crash-exposure reduction?</p>
            </div>
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--primary);">
              <h3 style="margin:0 0 0.3em; font-size:0.84rem;">3. Destination model improvements</h3>
              <p style="font-size:0.7rem;line-height:1.5;margin:0;">The new destination candidates (home-pull gravity, themed ACDC, two-stage SA3&rarr;SA2) are coded and ready for estimation. Home-pull adds late-day homeward attraction. Themed ACDC provides purpose-specific agglomeration. Two-stage destination connects district choice to zone choice through utility.</p>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.6em 0.9em;font-size:0.68rem;">
            <b style="color:var(--orange);">The key advance:</b> Traditional ABMs treat safety as a network outcome computed after assignment. This framework makes it a demand input &mdash; the same mechanism as travel time and land use. The next step is using this capability for <b>counterfactual policy evaluation</b>: quantifying how safety investments reshape travel demand, identifying which groups benefit, and providing evidence for infrastructure prioritisation.
          </div>
        </div>
      </section>'''
    
    html=html[:closing]+FUTURE+'\n\n      '+html[closing:]
    print('Added future work slide')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print(f'Wrote {len(html):,} bytes')
