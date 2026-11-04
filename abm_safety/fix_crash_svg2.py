html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
svg_start=html.find('<svg viewBox="0 0 780 95"')
svg_end=html.find('</svg>',svg_start)+6

# Replace with cleaner 2-row SVG
NEW='''<svg viewBox="0 0 780 80" style="max-width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
              <defs><marker id="a3" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4 Z" fill="#1C7293"/></marker></defs>
              <rect x="5" y="10" width="60" height="22" rx="4" fill="#2E3B8E"/><text x="35" y="24" fill="white" font-size="8" font-weight="bold" text-anchor="middle">Zone A</text>
              <line x1="65" y1="21" x2="80" y2="21" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="82" y="8" width="100" height="26" rx="3" fill="#C85A14" opacity="0.85"/><text x="132" y="20" fill="white" font-size="7" font-weight="bold" text-anchor="middle">Road segments</text><text x="132" y="31" fill="#FFE3CE" font-size="5.5" text-anchor="middle">s1, s2, ..., sn</text>
              <line x1="182" y1="21" x2="197" y2="21" stroke="#1C7293" stroke-width="1" marker-end="url(#a3)"/>
              <rect x="199" y="10" width="60" height="22" rx="4" fill="#2E3B8E"/><text x="229" y="24" fill="white" font-size="8" font-weight="bold" text-anchor="middle">Zone B</text>
              <line x1="259" y1="21" x2="278" y2="21" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="281" y="6" width="105" height="30" rx="5" fill="#2E7D6B" opacity="0.9"/><text x="333" y="20" fill="white" font-size="7" font-weight="bold" text-anchor="middle">CRASH<sub>AB</sub></text><text x="333" y="32" fill="#D5F5E3" font-size="5.5" text-anchor="middle">sum of E[crash] along path</text>
              <line x1="386" y1="21" x2="404" y2="21" stroke="#1C7293" stroke-width="1.2" marker-end="url(#a3)"/>
              <rect x="407" y="6" width="185" height="30" rx="5" fill="url(#bg3)"/><text x="499" y="20" fill="white" font-size="7" font-weight="bold" text-anchor="middle">V<sub>dest</sub> = ... + &beta;<sub>crash</sub> &middot; CRASH<sub>AB</sub></text><text x="499" y="32" fill="#D5F5E3" font-size="5.5" text-anchor="middle">enters destination utility for (Zone A, Zone B)</text>
              <defs><linearGradient id="bg3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1E2761"/><stop offset="100%" stop-color="#2E3B8E"/></linearGradient></defs>
              <!-- Bottom annotation -->
              <rect x="5" y="48" width="770" height="28" rx="4" fill="#EBEFFA" stroke="#DCE1F0" stroke-width="0.5"/>
              <text x="390" y="60" fill="#1E2761" font-size="6.5" text-anchor="middle">For every OD pair (371x371): shortest path through network -> sum expected crashes of every segment along path -> store as CRASH skim in OMX</text>
              <text x="390" y="73" fill="#888" font-size="5.5" text-anchor="middle">Stage 5 reads CRASH<sub>ij</sub> for every (person, zone) pair. The SA search decides the coefficient. The analyst only decides it must be present.</text>
            </svg>'''

html=html[:svg_start]+NEW+html[svg_end:]
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
