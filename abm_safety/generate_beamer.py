#!/usr/bin/env python3
"""
generate_beamer.py — Prepares assets for model_review_beamer.tex

Usage: python generate_beamer.py

This script:
  1. Extracts base64 images from the HTML deck → beamer/figures/*.png
  2. Creates the standalone population choropleth map → beamer/interactive/map.html
  3. The Beamer .tex file itself is hand-crafted and committed separately.
"""

import os, re, sys, base64, hashlib
from pathlib import Path
from bs4 import BeautifulSoup

SRC_HTML = Path(r"Z:\test_runs_tours\abm_safety\model_review_deck.html")
OUT_DIR  = Path(r"Z:\test_runs_tours\abm_safety\beamer")
FIG_DIR  = OUT_DIR / "figures"
INT_DIR  = OUT_DIR / "interactive"


def extract_base64_images(soup):
    """Find all base64 images in HTML, save as PNG, return updated soup."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src.startswith("data:image/"):
            m = re.match(r"data:image/(\w+);base64,(.+)", src, re.DOTALL)
            if m:
                fmt = m.group(1)
                data = m.group(2)
                h = hashlib.md5(data.encode()[:100]).hexdigest()[:8]
                fname = f"img_{count:03d}_{h}.{fmt}"
                count += 1
                try:
                    raw = base64.b64decode(data)
                    (FIG_DIR / fname).write_bytes(raw)
                    img["src"] = f"figures/{fname}"
                    print(f"  Extracted: {fname} ({len(raw)} bytes)")
                except Exception as e:
                    print(f"  [WARN] Failed: {fname}: {e}")
    return soup


def create_map_html(soup):
    """Create standalone interactive choropleth map HTML."""
    INT_DIR.mkdir(parents=True, exist_ok=True)

    # Find the synth-pop slide
    map_slide = soup.find("section", id="synth-pop-slide")
    if not map_slide:
        print("  [WARN] synth-pop-slide not found!")
        return None

    # Get styles
    style_blocks = "\n".join(str(s) for s in soup.find_all("style"))

    # Find all scripts after/inside the slide
    scripts = []
    current = map_slide
    for sibling in map_slide.find_next_siblings():
        if sibling.name == "script" and sibling.string:
            scripts.append(sibling.string.strip())
        elif sibling.name == "section":
            break
    for script in map_slide.find_all("script"):
        if script.string:
            scripts.append(script.string.strip())

    # Also need the data vars — find the script that defines ZD, PB64
    for script in soup.find_all("script"):
        if script.string and "var ZD={" in script.string:
            scripts.insert(0, script.string.strip())

    scripts_block = "<script>\n" + "\n\n".join(scripts) + "\n</" + "script>"

    # Get pbtn CSS
    pbtn_css = ""
    for style in soup.find_all("style"):
        if style.string and ".pbtn{" in style.string:
            pbtn_css = style.string
            break

    # Build standalone HTML
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive Population Map — SEQ ABM</title>
{style_blocks}
<style>
  body {{ margin: 0; padding: 0; background: var(--bg); height: 100vh; width: 100vw; overflow: hidden; font-family: var(--font-body); }}
  .stage {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }}
  .deck {{ position: relative; width: min(98vw, 174.2vh); height: min(98vh, 174.2vw); overflow: hidden; }}
  .nav, .counter, .arrow-btn, .nav-arrows, .theme-toggle, .nav-dots, .track {{ display: none !important; }}
  {pbtn_css}
</style>
</head>
<body>
<div class="stage">
  <div class="deck">
    {str(map_slide)}
  </div>
</div>
{scripts_block}
<script>
  // Override init to run immediately instead of via MutationObserver
  (function(){{
    var origInit = window.PM && window.PM.set;
    function forceInit() {{
      var cnv = document.getElementById('pmap');
      if (!cnv) {{ setTimeout(forceInit, 200); return; }}
      var ctx = cnv.getContext('2d');
      if (!ctx) {{ setTimeout(forceInit, 200); return; }}
      function rs() {{
        var r = cnv.parentElement.getBoundingClientRect();
        cnv.width = r.width * (window.devicePixelRatio || 1);
        cnv.height = r.height * (window.devicePixelRatio || 1);
        ctx.setTransform(1,0,0,1,0,0);
        ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
        cnv.style.width = r.width + 'px';
        cnv.style.height = r.height + 'px';
        if (typeof draw === 'function') draw();
      }}
      rs();
      window.addEventListener('resize', rs);
      if (typeof draw === 'function') draw();
    }}
    setTimeout(forceInit, 300);
  }})();
</script>
</body>
</html>"""

    outpath = INT_DIR / "map.html"
    outpath.write_text(full_html, encoding="utf-8")
    print(f"  Created: {outpath}")
    return outpath


def main():
    print("=" * 50)
    print("  Beamer Asset Generator")
    print("=" * 50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    INT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Parsing HTML...")
    raw = SRC_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "lxml")

    print("[2/3] Extracting images...")
    extract_base64_images(soup)

    print("[3/3] Creating interactive map...")
    create_map_html(soup)

    print("\nDone. Compile with:")
    print(f"  cd {OUT_DIR}")
    print(f"  pdflatex model_review_beamer.tex\n")


if __name__ == "__main__":
    main()
