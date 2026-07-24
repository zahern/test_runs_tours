html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
idx=html.find('Mode choice: nested logit with crash-frequency skims')
s=html.rfind('<section ',0,idx)
depth=1;i=s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e=i-1

SLIDE='''\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Mode Choice &middot; Key Findings</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">What the mode choice model tells us about travel behaviour</h1>
        <div class="body-area" style="gap:0.6vh;">
          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid var(--primary);">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem;">Travel time deters all modes</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">c_ivt_auto, c_ivt_transit, c_walk_time, and c_bike_time are all negative and significant. Every additional minute of travel time reduces mode probability. Effect strongest for walk and bike (distance-sensitive), then transit, then auto. Travellers trade off time against convenience.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid var(--teal);">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem;">Income and vehicles drive car use</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">c_incp_da and c_veh_da are positive: higher income and more vehicles increase drive-alone probability. c_noveh_transit and c_noveh_walk are positive for zero-car households. Vehicle ownership is the single strongest predictor of mode choice after travel time.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid var(--orange); background:var(--orange-soft);">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem; color:var(--orange);">Crash risk reduces motorised mode use</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">c_crash_veh is negative: higher crash exposure makes DA, SR, and Transit less attractive. Most significant for education tours (p&lt;0.001), consistent with parents avoiding high-crash routes for school travel. Walk and bike are affected through the destination utility's c_crash term.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid #2E3B8E;">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem;">Age and gender shape active travel</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">c_age_walk and c_age_bike are negative: older travellers less likely to walk or cycle. c_young_bike is positive for younger groups. c_female_transit and c_female_walk are positive: women more likely to use transit and walk than men, controlling for other factors.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid var(--good);">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem;">Work status structures mode availability</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">c_ftwork_da is positive: full-time workers more likely to drive alone (scheduling pressure). c_notwork_transit is positive: non-workers more likely to use transit. c_ptwork_sr is positive for part-time workers on shared ride.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em; border-top:4px solid #9B59B6;">
              <h3 style="margin:0 0 0.2em; font-size:0.8rem;">Nest structure confirms mode grouping</h3>
              <p style="font-size:0.64rem;line-height:1.5;margin:0;">Mu_Car (DA+SR) and Mu_Motor (Car+Transit) are between 0.1 and 1.0, confirming motorised modes share unobserved preferences. Mu_NonMotor similarly groups active modes. The NL structure is statistically justified: travellers substitute within nests more readily than across them.</p>
            </div>
          </div>
        </div>
      </section>'''

html=html[:e]+'\n\n'+SLIDE+html[e:]
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print(f'Done. Wrote {len(html):,} bytes')
