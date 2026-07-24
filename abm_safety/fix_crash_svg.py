html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
svg_start=html.find('<svg viewBox="0 0 780 95"')
svg_end=html.find('</svg>',svg_start)+6

NEW='''<svg viewBox="0 0 780 95" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a3" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L6,2.5 L0,5 Z" fill="#1C7293"/></marker></defs>
              <rect x="5" y="8" width="55" height="22" rx="4" fill="#2E3B8E"/><text x="32" y="22" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Zone A</text>
              <line x1="60" y1="19" x2="80" y2="19" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="83" y="13" width="90" height="30" rx="4" fill="#C85A14" opacity="0.85"/><text x="128" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment s<sub>1</sub></text><text x="128" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0032</text>
              <line x1="173" y1="28" x2="193" y2="28" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="195" y="13" width="90" height="30" rx="4" fill="#C85A14" opacity="0.65"/><text x="240" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment s<sub>2</sub></text><text x="240" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0018</text>
              <line x1="285" y1="28" x2="305" y2="28" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="307" y="13" width="90" height="30" rx="4" fill="#C85A14" opacity="0.45"/><text x="352" y="25" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Segment s<sub>3</sub></text><text x="352" y="37" fill="#FFE3CE" font-size="5.5" text-anchor="middle">E[crash] = 0.0041</text>
              <line x1="397" y1="28" x2="417" y2="28" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="419" y="8" width="55" height="22" rx="4" fill="#2E3B8E"/><text x="446" y="22" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Zone B</text>
              <line x1="474" y1="19" x2="500" y2="19" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="503" y="6" width="105" height="46" rx="5" fill="#2E7D6B" opacity="0.9"/><text x="555" y="22" fill="white" font-size="7" font-weight="bold" text-anchor="middle">CRASH<sub>AB</sub></text><text x="555" y="36" fill="#D5F5E3" font-size="6" text-anchor="middle">= &Sigma; E[crash<sub>s</sub>]</text><text x="555" y="48" fill="#aaa" font-size="5.5" text-anchor="middle">sum along path</text>
              <line x1="608" y1="29" x2="630" y2="29" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="633" y="6" width="142" height="46" rx="5" fill="url(#bg3)"/><text x="704" y="22" fill="white" font-size="7" font-weight="bold" text-anchor="middle">V<sub>dest(B)</sub></text><text x="704" y="36" fill="#D5F5E3" font-size="5.5" text-anchor="middle">= ... + &beta;<sub>crash</sub>&middot;CRASH<sub>AB</sub></text><text x="704" y="48" fill="#D5F5E3" font-size="5.5" text-anchor="middle">enters destination utility</text>
              <defs><linearGradient id="bg3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>
              <text x="390" y="68" fill="#888" font-size="6" text-anchor="middle">Zone A &rarr; road segments s<sub>1</sub>, s<sub>2</sub>, s<sub>3</sub> &rarr; Zone B &rarr; CRASH<sub>AB</sub> stored in OMX &rarr; enters destination utility for every (person, zone) pair</text>
            </svg>'''

html=html[:svg_start]+NEW+html[svg_end:]
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print(f'Done. Wrote {len(html):,} bytes')
