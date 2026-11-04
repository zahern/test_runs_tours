import base64, os
DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
LOGO = r"Z:\test_runs_tours\abm_safety\qut.png"

with open(LOGO, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
print(f"Logo: {os.path.getsize(LOGO):,} bytes -> {len(logo_b64):,} b64")

with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find title slide
idx = html.find("<!-- 1. TITLE -->")
s = html.find("<section ", idx)
depth = 1; i = s + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
e = i - 1

slide = html[s:e]

# Find the author paragraph and insert logo before it
author_tag = "Zeke Ahern"
ai = slide.find(author_tag)
if ai > 0:
    # Find the start of the <p> containing the author
    p_start = slide.rfind("<p ", 0, ai)
    if p_start < 0: p_start = slide.rfind("<p>", 0, ai)
    
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="position:absolute;top:3%;right:4%;height:5vh;opacity:0.9;" alt="QUT">'
    
    new_slide = slide[:p_start] + logo_html + slide[p_start:]
    html = html[:s] + new_slide + html[e:]
    print(f"Added QUT logo to title slide")
else:
    print("Author line not found")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
