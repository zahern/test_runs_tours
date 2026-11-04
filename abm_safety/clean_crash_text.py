html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Remove SVG bottom annotation bar
old='<rect x="5" y="46" width="770" height="24" rx="4" fill="#EBEFFA" stroke="#DCE1F0" stroke-width="0.5"/>'
new='<!-- annotation removed -->'
html=html.replace(old,new)
html=html.replace('<text x="390" y="58" fill="#1E2761" font-size="6" text-anchor="middle">371x371 OD pairs: shortest path -> sum E[crash] per segment -> CRASH skim in OMX -> Stage 5 reads CRASH<sub>ij</sub> for every (person, zone) pair</text>','')
html=html.replace('<text x="390" y="68" fill="#888" font-size="5" text-anchor="middle">SA search decides coefficient. Analyst only decides CRASH must be present (force-included by ga_constraints.py). Same mechanism as travel-time skims.</text>','')
print('Removed SVG annotation text')

# 2. Remove the long paragraph
para='Each 100m road segment'
idx=html.find(para)
if idx>0:
    # Find the enclosing div/paragraph
    tag_start=html.rfind('<',idx-200,idx)
    # Find end of this block
    end=html.find('</div>',html.find('</div>',idx)+6)
    if end<0: end=html.find('</p>',idx)+4
    # Find actual start
    if html[tag_start:tag_start+4]=='<div':
        # Find matching </div>
        depth=1;j=tag_start
        while depth>0 and j<len(html):
            j+=1
            if html[j:j+4]=='<div':depth+=1
            elif html[j:j+6]=='</div>':depth-=1
        block_end=j+6
        html=html[:tag_start]+html[block_end:]
        print('Removed paragraph block')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
