html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
idx=html.find('A day in the life: simulated activity plans')
s=html.rfind('<section ',0,idx)
depth=1;i=s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e=i-1

future='''\
      <section class="slide dark-slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">Future Work</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem); color:#FFFFFF;">What comes next</h1>
        <div class="body-area" style="gap:0.8vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.7vw; flex:1; min-height:0;">
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <h3 style="font-size:0.9rem;">Safety counterfactuals</h3>
              <p style="font-size:0.72rem;line-height:1.5;margin:0;">Because crash risk is an explicit demand input, the model can answer "what if" questions. Run baseline (CRASH ON) and counterfactual (CRASH OFF) with identical coefficients, then compare mode shares, destination patterns, and crash exposure per person-day.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <h3 style="font-size:0.9rem;">Infrastructure scenarios</h3>
              <p style="font-size:0.72rem;line-height:1.5;margin:0;">Modify the CRASH skim to reflect proposed safety treatments: reduced speeds, protected bike lanes, intersection upgrades. Quantify which groups benefit most and how mode shares shift.</p>
            </div>
            <div class="card on-primary" style="padding:0.7em 0.9em;">
              <h3 style="font-size:0.9rem;">Destination model improvements</h3>
              <p style="font-size:0.72rem;line-height:1.5;margin:0;">New destination candidates coded and ready: home-pull gravity (late-day homeward attraction), themed ACDC (purpose-specific agglomeration), two-stage SA3\u2192SA2 destination choice.</p>
            </div>
          </div>
          <div class="card on-primary" style="padding:0.6em 0.9em; background:rgba(200,90,20,0.25);">
            <p style="font-size:0.72rem;margin:0;text-align:center;">The key advance: traditional ABMs treat safety as a network outcome. This framework makes it a demand input. The next step is counterfactual policy evaluation.</p>
          </div>
        </div>
      </section>'''

html=html[:s]+future+html[e:]
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print(f'Done. Wrote {len(html):,} bytes')
