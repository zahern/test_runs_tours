import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Extract QUT logo base64
m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)"', html)
qut_b64 = m.group(0) if m else ""
print(f"QUT logo found: {len(qut_b64)} chars")

# Clean title slide - just title + author + logo
TITLE = f"""      <section class="slide dark-slide">
        <div style="flex:1; display:flex; flex-direction:column; justify-content:center; max-width:72%;">
          <div class="eyebrow">Integrated Model Review</div>
          <h1 class="slide-title" style="font-size:clamp(1.8rem,3.6vw,3rem);">A Safety-Augmented Activity-Based Model<br>for South-East Queensland</h1>
          <p style="font-family:var(--font-mono); font-size:0.8rem; color:#9FADE0; margin-top:2em;">Zeke Ahern &middot; Amalan Mahendran &middot; Alexander Paz &mdash; Queensland University of Technology</p>
        </div>
        <img src="{qut_b64}" style="position:absolute;top:3%;right:4%;height:5vh;opacity:0.9;" alt="QUT">
      </section>"""

# Find title slide start and Why Matters start
title_start = html.find("<!-- 1. TITLE -->")
title_section = html.find("<section ", title_start)
wm = html.find("<!-- 2. WHY THIS MATTERS -->")

html = html[:title_section] + TITLE + "\n\n      " + html[wm:]
print(f"Cleaned title slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
