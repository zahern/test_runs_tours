import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# The problem: slide 6 (MetaCount paper) is 552KB - it swallowed slides 7 onwards
# Also slide 1 has extra content merged in
# Let me find the actual boundaries

# Find "<!-- 1. TITLE -->" through all slide comments
comments = [(m.start(), m.group()) for m in re.finditer(r'<!-- \d+[a-z]?\. [A-Z]', html)]
print("Slide comments:")
for pos, txt in comments[:20]:
    print(f"  {pos}: {txt.strip()}")

# Find where slide 6 ends abnormally
slide6_start = 85366
slide6_end = 637370  # Should be much smaller

# Check what's between slide 6 end and slide 8 (landuse)
between = html[637370:637441]
print(f"\nBetween slide 6 and 8: {len(between)} chars: {repr(between[:100])}")

# The ABM needs cards and detail slides are missing. Let me find them
abm_cards = html.find("Six ingredients every ABM must bring together")
print(f"\nABM cards: {abm_cards if abm_cards>0 else 'NOT FOUND'}")
abm_detail = html.find("Disaggregate persons &mdash;")
print(f"ABM detail list: {abm_detail if abm_detail>0 else 'NOT FOUND'}")
