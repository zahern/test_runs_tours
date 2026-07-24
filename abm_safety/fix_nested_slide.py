import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# Find the MetaCount paper slide
paper_start = html.find("MetaCountRegressor: a major motivation")
s8 = html.rfind("<section ", 0, paper_start)

# Find its true end (after all nested sections)
depth = 1; i = s8 + 1
while depth > 0 and i < len(html):
    if html[i:i+9] == "<section ": depth += 1
    elif html[i:i+10] == "</section>": depth -= 1
    i += 1
paper_end = i - 1
print(f"Paper slide: {s8}-{paper_end} ({paper_end-s8:,} bytes)")

# Check if the estimation slide is nested inside
est_idx = html.find("The core problem: for any model", s8, paper_end)
if est_idx > 0:
    print(f"Estimation slide is NESTED inside paper slide at offset {est_idx}")
    # Extract the estimation slide
    est_s = html.rfind("<section ", 0, est_idx)
    depth = 1; j = est_s + 1
    while depth > 0 and j < len(html):
        if html[j:j+9] == "<section ": depth += 1
        elif html[j:j+10] == "</section>": depth -= 1
        j += 1
    est_e = j - 1
    est_slide = html[est_s:est_e]
    print(f"  Estimation slide: {est_s}-{est_e} ({est_e-est_s:,} bytes)")
    
    # Remove the estimation slide from inside the paper slide
    html = html[:est_s] + html[est_e:]
    paper_end -= (est_e - est_s)  # Adjust paper end
    
    # Re-insert the estimation slide AFTER the paper slide
    html = html[:paper_end] + "\n\n" + est_slide + html[paper_end:]
    print(f"  Moved estimation slide after paper slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
