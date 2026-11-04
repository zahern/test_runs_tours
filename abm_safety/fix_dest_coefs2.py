html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Remove validation plots from dest coefs slide
idx=html.find('Destination choice: what drives zone attractiveness')
if idx>0:
    s=html.rfind('<section ',0,idx)
    depth=1;i=s+1
    while depth>0 and i<len(html):
        if html[i:i+9]=='<section ':depth+=1
        elif html[i:i+10]=='</section>':depth-=1
        i+=1
    e=i-1
    slide=html[s:e]
    # Find and remove the validation plot grid
    val=slide.find('shadow_price_validation')
    if val>0:
        grid_start=slide.rfind('<div class="grid"',0,val)
        depth2=1;j=grid_start
        while depth2>0 and j<len(slide):
            j+=1
            if slide[j:j+4]=='<div':depth2+=1
            elif slide[j:j+6]=='</div>':depth2-=1
        grid_end=j+6
        slide=slide[:grid_start]+slide[grid_end:]
        html=html[:s]+slide+html[e:]
        print('Removed validation plots from dest coefs slide')

# 2. Update crash coefficient to estimated value
replacements=[
    ('c_crash = -0.100','c_crash = -0.127'),
    ('-0.100 (fixed, force-included)','-0.127 (SA-estimated, force-included)'),
    ('-0.100 in destination utilities','-0.127 in destination utilities'),
    ('c_crash = -0.100','c_crash = -0.127'),
    ('crash = -0.100','crash = -0.127'),
    ('crash=-0.100','crash=-0.127'),
    ('(fixed, force-included)','(SA-estimated, force-included)'),
    ('-0.1000</td>','-0.127</td>'),
]
for old,new in replacements:
    old_count=html.count(old)
    if old_count:
        html=html.replace(old,new)
        print(f'  {old} -> {new} ({old_count}x)')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
