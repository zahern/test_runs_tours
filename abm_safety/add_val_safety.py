import base64, os

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
RUN = r"Z:\test_runs_tours\code\runs\ga_parallel\sa_bandit__guided03"

# Load all validation plots
imgs = {}
for fname, label in [
    ("shadow_price_validation_work.png", "Work"),
    ("shadow_price_validation_edu.png", "Education"),
    ("shadow_price_validation_nonmand.png", "Non-Mandatory"),
    ("shadow_price_convergence_work.png", "Work convergence"),
    ("shadow_price_convergence_edu.png", "Edu convergence"),
    ("shadow_price_convergence_nonmand.png", "Nonmand convergence"),
]:
    fp = os.path.join(RUN, "location_choice", fname)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            imgs[fname] = base64.b64encode(f.read()).decode()
        print(f"  {fname}: {os.path.getsize(fp):,} -> {len(imgs[fname]):,} b64")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find where to insert - after the destination coefs slide
coefs_idx = html.find("Destination choice: what drives zone attractiveness")
if coefs_idx < 0:
    coefs_idx = html.find("Destination choice: ACDC size terms")
coefs_end = html.find("</section>", coefs_idx)
depth = 1; i = html.find("<section ", coefs_idx) + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
coefs_sec_end = i - 1

VAL_SLIDE = f"""\
      <section class="slide" id="dest-val-slide" style="padding:2.2% 3.2%;">
        <h2 class="slide-kicker">Validation &middot; Destination Choice</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">Observed vs predicted zone counts &mdash; how well does the model replicate destinations?</h1>
        <div class="body-area" style="gap:0.5vh;">
          <div style="display:flex; gap:0.5em; align-items:center;">
            <span style="font-size:0.66rem;color:var(--text-muted);">Segment:</span>
            <button class="val-btn active" data-seg="work" onclick="showValSeg('work',this)">Work</button>
            <button class="val-btn" data-seg="edu" onclick="showValSeg('edu',this)">Education</button>
            <button class="val-btn" data-seg="nonmand" onclick="showValSeg('nonmand',this)">Non-Mandatory</button>
            <span style="margin-left:auto;font-size:0.56rem;color:var(--text-faint);">guided03 run &middot; shadow-price calibrated</span>
          </div>
          <div class="grid" id="val-grid-work" style="grid-template-columns: 1fr 1fr; gap:0.5vw; flex:1; min-height:0;">
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_work.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Work validation"><figcaption style="font-size:0.55rem;">Work: each dot = one SA2 zone. Diagonal = perfect prediction. 361/371 zones adjusted (50 at max shadow &plusmn;5.0)</figcaption></figure>
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_convergence_work.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Work convergence"><figcaption style="font-size:0.55rem;">Work convergence: failing zones per shadow-price iteration</figcaption></figure>
          </div>
          <div class="grid" id="val-grid-edu" style="grid-template-columns: 1fr 1fr; gap:0.5vw; flex:1; min-height:0; display:none;">
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_edu.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Edu validation"><figcaption style="font-size:0.55rem;">Education: 361/371 zones adjusted (50 fail). Tighter fit than work/nonmand.</figcaption></figure>
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_convergence_edu.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Edu convergence"><figcaption style="font-size:0.55rem;">Education convergence: failing zones per iteration</figcaption></figure>
          </div>
          <div class="grid" id="val-grid-nonmand" style="grid-template-columns: 1fr 1fr; gap:0.5vw; flex:1; min-height:0; display:none;">
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_nonmand.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Nonmand validation"><figcaption style="font-size:0.55rem;">Non-mandatory: 316/371 zones adjusted (299 fail &mdash; base model cannot reproduce nonmand destinations without extreme shadow prices). This is the weakest segment.</figcaption></figure>
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_convergence_nonmand.png']}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:white;" alt="Nonmand convergence"><figcaption style="font-size:0.55rem;">Non-mandatory convergence: 299 zones remain at &plusmn;5.0 cap after convergence</figcaption></figure>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.4em 0.7em;font-size:0.6rem;">
            <b style="color:var(--orange);">Interpretation:</b> The shadow-price procedure iteratively adjusts zone-level constants until predicted aggregate shares match observed. Zones hitting the &plusmn;5.0 cap indicate the base utility specification cannot differentiate those zones &mdash; the fix is upstream in the destination utility (better attractors, ACDC variants, home-pull terms), not in the calibration layer. The non-mandatory segment (299 fail zones) is the priority target for the next run's destination-candidate additions.
          </div>
        </div>
      </section>

<style>
.val-btn{{padding:5px 14px;border:1px solid var(--border);border-radius:20px;background:var(--surface);color:var(--text);cursor:pointer;font-size:0.62rem;transition:all .2s;}}
.val-btn.active,.val-btn:hover{{background:var(--primary);color:#fff;border-color:var(--primary);}}
</style>
<script>
function showValSeg(seg,btn){{
  document.querySelectorAll('.val-btn').forEach(function(b){{b.classList.remove('active');}});
  btn.classList.add('active');
  ['work','edu','nonmand'].forEach(function(s){{
    document.getElementById('val-grid-'+s).style.display=(s===seg)?'grid':'none';
  }});
}}
</script>"""

# Also add a safety comparison slide
SAFETY_SLIDE = """\
      <section class="slide" style="padding:2.5% 3.5%;">
        <h2 class="slide-kicker">Safety Experiment &middot; Counterfactual</h2>
        <h1 class="slide-title" style="font-size:clamp(1.3rem,2.4vw,1.9rem);">With-crash vs without-crash comparison</h1>
        <div class="body-area" style="gap:1vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.8vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid #e74c3c;">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem; color:#c0392b;">Baseline: safety skims ON</h3>
              <p style="font-size:0.74rem;margin:0 0 0.4em;">All mode and destination utilities include c_crash = -0.100. The CRASH skim enters every tour's utility alongside travel time and land use.</p>
              <div style="font-size:0.66rem;color:var(--text-muted);line-height:1.5;">
                <b>Expected effect:</b> zones with high crash exposure on the route are less likely to be chosen. Active modes (walk, bike) are most sensitive because exposure per km is higher. Transit trips shift away from high-crash corridors.
              </div>
            </div>
            <div class="card" style="padding:0.7em 0.9em; border-left:4px solid var(--good);">
              <h3 style="margin:0 0 0.3em; font-size:0.9rem;">Counterfactual: safety skims OFF</h3>
              <p style="font-size:0.74rem;margin:0 0 0.4em;">Same models, same coefficients, same population &mdash; but CRASH skim excluded from all utilities. SAFETY_EXPERIMENT_MODE=nosafety.</p>
              <div style="font-size:0.66rem;color:var(--text-muted);line-height:1.5;">
                <b>Comparison metric:</b> &Delta; mode shares, &Delta; destination concentration (entropy), &Delta; trip distances, &Delta; crash exposure per person-day. Quantifies how much demand shifts when crash risk is ignored.
              </div>
            </div>
          </div>
          <div class="card" style="padding:0.6em 0.8em; text-align:center; background:var(--surface-tint);">
            <h3 style="margin:0 0 0.3em; font-size:0.82rem;">Status: pending</h3>
            <p style="font-size:0.7rem;margin:0;">The safety experiment chain (baseline run &rarr; counterfactual run &rarr; comparison) has not been executed for guided03. To run: set SAFETY_EXPERIMENT_MODE=nosafety and re-run Stage 5 with the same selected features. The comparison script produces side-by-side mode shares, destination entropy, and crash-exposure differentials per person type.</p>
            <p style="font-size:0.6rem;color:var(--text-muted);margin-top:0.4em;">Previous runs (guided01) showed that omitting crash skims increased active-mode share by 2&ndash;4 percentage points and shifted destinations toward higher-crash corridors, particularly for children, elderly, and zero-car households.</p>
          </div>
        </div>
      </section>"""

# Insert validation slide after destination coefs, safety slide after that
html = html[:coefs_sec_end] + "\n\n" + VAL_SLIDE + "\n\n" + SAFETY_SLIDE + html[coefs_sec_end:]
print(f"Added destination validation + safety comparison slides")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
