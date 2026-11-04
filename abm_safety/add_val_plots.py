import base64, os
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
FIGS = r"Z:\test_runs_tours\code\runs\ga_parallel\sa_bandit__guided03\location_choice"

imgs = {}
for f in ["shadow_price_validation_work.png", "shadow_price_validation_edu.png", "shadow_price_validation_nonmand.png"]:
    fp = os.path.join(FIGS, f)
    with open(fp, "rb") as ff:
        imgs[f] = base64.b64encode(ff.read()).decode()

with open(DECK, encoding="utf-8") as f:
    html = f.read()

idx = html.find("Destination choice: what drives zone attractiveness")
s = html.rfind("<section ", 0, idx)
depth = 1; i = s + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
e = i - 1

slide = html[s:e]

legend = slide.find('<b style="color:var(--orange);">Key:</b>')
if legend > 0:
    plots = f"""          <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap:0.4vw; margin-top:0.3vh;">
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_work.png']}" style="max-width:100%;max-height:13vh;object-fit:contain;border-radius:4px;border:1px solid var(--border);" alt="Work validation"><figcaption style="font-size:0.5rem;">Work: observed vs predicted zone counts</figcaption></figure>
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_edu.png']}" style="max-width:100%;max-height:13vh;object-fit:contain;border-radius:4px;border:1px solid var(--border);" alt="Education validation"><figcaption style="font-size:0.5rem;">Education: observed vs predicted zone counts</figcaption></figure>
            <figure class="evidence" style="margin:0;"><img src="data:image/png;base64,{imgs['shadow_price_validation_nonmand.png']}" style="max-width:100%;max-height:13vh;object-fit:contain;border-radius:4px;border:1px solid var(--border);" alt="Non-mandatory validation"><figcaption style="font-size:0.5rem;">Non-mandatory: observed vs predicted zone counts</figcaption></figure>
          </div>
"""
    legend_div = slide.rfind('<div class="card"', 0, legend)
    new_slide = slide[:legend_div] + plots + "\n          " + slide[legend_div:]
    html = html[:s] + new_slide + html[e:]
    print("Added validation plots to destination coefs slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
