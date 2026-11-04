html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# Find slide 11 (iterative) end
s11=992957
depth=1;e11=s11+1
while depth>0 and e11<len(html):
    if html[e11:e11+9]=='<section ':depth+=1
    elif html[e11:e11+10]=='</section>':depth-=1
    e11+=1
e11-=1

# Find slide 18 (constraint-guided)
s18=2888753
depth=1;e18=s18+1
while depth>0 and e18<len(html):
    if html[e18:e18+9]=='<section ':depth+=1
    elif html[e18:e18+10]=='</section>':depth-=1
    e18+=1
e18-=1

slide18=html[s18:e18]
# Remove slide 18
html=html[:s18]+html[e18:]
# Recalculate slide 11 end after removal
depth=1;e11=s11+1
while depth>0 and e11<len(html):
    if html[e11:e11+9]=='<section ':depth+=1
    elif html[e11:e11+10]=='</section>':depth-=1
    e11+=1
e11-=1
# Insert after slide 11
html=html[:e11]+'\n\n'+slide18+html[e11:]

# Fix title
html=html.replace('The iterative specification process \u2014 and why it doesn\u2019t scale','The iterative specification process')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
