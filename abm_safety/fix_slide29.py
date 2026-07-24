# -*- coding: utf-8 -*-
"""Fix slide 29 nesting, add validation summary slide."""
import re

DECK = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
with open(DECK, encoding="utf-8") as f:
    html = f.read()

# ── Find and fix slide 29 nesting ──
# Slide 29 is "Key takeaways" which has slide 30 (validation) nested inside
take_idx = html.find("Key takeaways from this review")
if take_idx > 0:
    s29 = html.rfind("<section ", 0, take_idx)
    depth = 1; i = s29 + 1
    while depth > 0 and i < len(html):
        if html[i:i+9] == "<section ": depth += 1
        elif html[i:i+10] == "</section>": depth -= 1
        i += 1
    e29 = i - 1
    print(f"Slide 29 true end: {e29} (was 4798127)")
    
    # Check for nested slides within slide 29
    nested_count = 0
    j = s29 + 1
    while j < e29:
        if html[j:j+9] == "<section ":
            nested_count += 1
            # Find this nested section's end
            nd = 1; k = j + 1
            while nd > 0 and k < e29:
                if html[k:k+9] == "<section ": nd += 1
                elif html[k:k+10] == "</section>": nd -= 1
                k += 1
            nested_end = k - 1
            nested = html[j:nested_end]
            h1 = re.search(r'<h1[^>]*>(.*?)</h1>', nested, re.DOTALL)
            title = re.sub(r'<[^>]+>', '', h1.group(1) if h1 else '?')[:80]
            print(f"  Nested at {j}-{nested_end}: {title}")
            
            # Extract nested slide and place it after slide 29
            html = html[:j] + html[nested_end:]
            e29 -= (nested_end - j)
            # Insert after e29
            html = html[:e29] + "\n\n" + nested + html[e29:]
            e29 += len(nested) + 2
            
            j = e29  # break the loop
        j += 1
    
    print(f"Fixed {nested_count} nested slides")

# ── Add validation summary slide before the closing ──
closing = html.find("Questions &amp; Discussion")
if closing > 0:
    prev_sec = html.rfind("</section>", 0, closing)
    
    VAL_SUMMARY = """\
      <section class="slide" style="padding:2.2% 3.5%;">
        <h2 class="slide-kicker">Validation Summary &middot; guided03</h2>
        <h1 class="slide-title" style="font-size:clamp(1.2rem,2.3vw,1.8rem);">Model performance at a glance</h1>
        <div class="body-area" style="gap:0.6vh;">
          <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat hero"><div class="value">17h 39m</div><div class="label">total GA wall time (all 6 stages)</div></div>
            <div class="stat"><div class="value">3.82M</div><div class="label">synthetic persons predicted</div></div>
            <div class="stat"><div class="value">7.07M</div><div class="label">trips generated</div></div>
            <div class="stat"><div class="value">188GB</div><div class="label">peak memory (OOM during reports)</div></div>
          </div>
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.6vw; flex:1; min-height:0;">
            <div class="card" style="padding:0.5em 0.7em;">
              <h3 style="margin:0 0 0.3em; font-size:0.82rem;">Stage 5 model fit (McFadden &rho;&sup2;)</h3>
              <table style="width:100%;font-size:0.66rem;border-collapse:collapse;">
                <tr style="background:var(--surface-tint);"><th style="text-align:left;padding:3px 5px;">Segment</th><th style="text-align:left;padding:3px 5px;">Model</th><th style="text-align:right;padding:3px 5px;">&rho;&sup2;</th><th style="text-align:left;padding:3px 5px;">Rating</th></tr>
                <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.629</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                <tr><td style="padding:2px 5px;">Work</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.413</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.423</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                <tr><td style="padding:2px 5px;">Edu</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.605</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                <tr><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Mode (NL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.549</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
                <tr><td style="padding:2px 5px;">Nonmand</td><td style="padding:2px 5px;">Dest (MNL)</td><td style="text-align:right;padding:2px 5px;font-family:var(--font-mono);color:var(--good);">0.685</td><td style="padding:2px 5px;color:var(--good);">Excellent</td></tr>
              </table>
              <p style="font-size:0.56rem;color:var(--text-muted);margin:0.3em 0 0;">&rho;&sup2; &gt; 0.2 acceptable, &gt; 0.4 excellent for discrete choice. BIC = k&middot;ln(n) - 2&middot;LL. Lower = better.</p>
            </div>
            <div class="card" style="padding:0.5em 0.7em;">
              <h3 style="margin:0 0 0.3em; font-size:0.82rem;">Destination shadow-price calibration</h3>
              <table style="width:100%;font-size:0.66rem;border-collapse:collapse;">
                <tr style="background:var(--surface-tint);"><th style="text-align:left;padding:3px 5px;">Segment</th><th style="text-align:right;padding:3px 5px;">Zones adj.</th><th style="text-align:right;padding:3px 5px;">Fail zones</th><th style="text-align:right;padding:3px 5px;">Max |sp|</th></tr>
                <tr><td style="padding:2px 5px;">Education</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">361/371</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;color:var(--orange);">50</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">5.0</td></tr>
                <tr><td style="padding:2px 5px;">Non-mandatory</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">316/371</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;color:#e74c3c;">299</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">5.0</td></tr>
                <tr><td style="padding:2px 5px;">Work</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">316/371</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;color:#e74c3c;">299</td><td style="text-align:right;font-family:var(--font-mono);padding:2px 5px;">5.0</td></tr>
              </table>
              <p style="font-size:0.56rem;color:var(--text-muted);margin:0.3em 0 0;">Zones hitting &plusmn;5.0 cap indicate the base utility cannot differentiate them. Nonmand (299 fail) is the priority target for home-pull + themed ACDC.</p>
            </div>
          </div>
          <div class="card" style="background:var(--orange-soft);padding:0.4em 0.7em;font-size:0.6rem;">
            <b style="color:var(--orange);">Next run priorities:</b> (1) Home-gravity (c_home_pull, c_home_pull_late) for nonmand destination differentiation. (2) Themed ACDC variants (retail, office, edu, leisure) mutex-encoded against generic c_aggl. (3) Two-stage destination (SA3 &rarr; SA2) with STAGE5_DEST_TWO_STAGE=1. (4) Mode choice pool widened with CBD&times;transit, transit access, micromobility, purpose&times;mode dummies.
          </div>
        </div>
      </section>"""
    
    html = html[:closing] + VAL_SUMMARY + "\n\n      " + html[closing:]
    print("Added validation summary slide")

with open(DECK, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes")
