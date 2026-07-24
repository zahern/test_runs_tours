html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# Find value slide
v_idx=html.find('3.8 million synthetic persons')
v_s=html.rfind('<section ',0,v_idx)
depth=1;i=v_s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
v_e=i-1
value_slide=html[v_s:v_e]

# Find map slide
m_idx=html.find('id="synth-pop-slide"')
m_s=html.rfind('<section ',0,m_idx)
depth=1;i=m_s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
m_e=i-1

# Remove value slide from current position
html=html[:v_s]+html[v_e:]
# Recalculate map end
m_idx2=html.find('id="synth-pop-slide"')
m_s2=html.rfind('<section ',0,m_idx2)
depth=1;i=m_s2+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
m_e2=i-1
# Insert after map
html=html[:m_e2]+'\n\n'+value_slide+html[m_e2:]
print('Done')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
