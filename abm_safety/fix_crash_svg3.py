html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

svg_start=html.find('<svg viewBox="0 0 780 80"')
svg_end=html.find('</svg>',svg_start)+6

NEW='''<svg viewBox="0 0 780 72" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a3" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#1C7293"/></marker></defs>
              <rect x="5" y="10" width="60" height="20" rx="4" fill="#2E3B8E"/><text x="35" y="24" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Zone A</text>
              <line x1="65" y1="20" x2="78" y2="20" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="80" y="7" width="100" height="26" rx="3" fill="#C85A14" opacity="0.85"/><text x="130" y="19" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Road segments</text><text x="130" y="30" fill="#FFE3CE" font-size="5.5" text-anchor="middle">each with E[crash] from SPF</text>
              <line x1="180" y1="20" x2="193" y2="20" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="195" y="10" width="60" height="20" rx="4" fill="#2E3B8E"/><text x="225" y="24" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Zone B</text>
              <line x1="255" y1="20" x2="272" y2="20" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="274" y="6" width="105" height="28" rx="5" fill="#2E7D6B" opacity="0.9"/><text x="326" y="19" fill="white" font-size="7" font-weight="bold" text-anchor="middle">CRASH<sub>AB</sub></text><text x="326" y="31" fill="#D5F5E3" font-size="5.5" text-anchor="middle">sum of E[crash] along path</text>
              <line x1="379" y1="20" x2="396" y2="20" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="398" y="6" width="200" height="28" rx="5" fill="url(#bg3)"/><text x="498" y="19" fill="white" font-size="7" font-weight="bold" text-anchor="middle">V<sub>dest</sub> adds &beta;<sub>crash</sub> &middot; CRASH<sub>AB</sub></text><text x="498" y="31" fill="#D5F5E3" font-size="5.5" text-anchor="middle">enters destination utility</text>
              <defs><linearGradient id="bg3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>
              <rect x="5" y="46" width="770" height="24" rx="4" fill="#EBEFFA" stroke="#DCE1F0" stroke-width="0.5"/>
              <text x="390" y="58" fill="#1E2761" font-size="6" text-anchor="middle">371x371 OD pairs: shortest path -> sum E[crash] per segment -> CRASH skim in OMX -> Stage 5 reads CRASH<sub>ij</sub> for every (person, zone) pair</text>
              <text x="390" y="68" fill="#888" font-size="5" text-anchor="middle">SA search decides coefficient. Analyst only decides CRASH must be present (force-included by ga_constraints.py). Same mechanism as travel-time skims.</text>
            </svg>'''

html=html[:svg_start]+NEW+html[svg_end:]

# Also update the bottom explanation paragraph
old_para='Each road segment\'s predicted crash frequency is <b>summed along the shortest path</b>'
new_para='Each 100m road segment\'s predicted crash frequency is <b>summed along the shortest path</b>'
html=html.replace(old_para, new_para)

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
