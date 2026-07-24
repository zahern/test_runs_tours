html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Remove remaining crash text fragments
html=html.replace('sum of E[crash] along path\n','')
html=html.replace('Vdest adds \u03b2crash \u00b7 CRASHABenters destination utility ','')
# Remove any line with SA search decides
import re
html=re.sub(r'.*SA search decides.*\n?','',html)
print('Cleaned remaining crash text')

# 2. Fix nested questions slide
fut_idx=html.find('What comes next')
fut_s=html.rfind('<section ',0,fut_idx)
depth=1;i=fut_s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
fut_e=i-1

# Find nested questions slide
que_idx=html.find('Questions &amp; Discussion',fut_s,fut_e)
if que_idx>0:
    que_s=html.rfind('<section ',0,que_idx)
    depth=1;j=que_s+1
    while depth>0 and j<fut_e:
        if html[j:j+9]=='<section ':depth+=1
        elif html[j:j+10]=='</section>':depth-=1
        j+=1
    que_e=j-1
    questions=html[que_s:que_e]
    html=html[:que_s]+html[que_e:]
    fut_e_new=fut_e-(que_e-que_s)
    html=html[:fut_e_new]+'\n\n'+questions+html[fut_e_new:]
    print('Extracted questions slide')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
