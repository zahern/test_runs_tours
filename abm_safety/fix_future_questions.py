html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Add surveying to future work
old_bottom='The key advance: traditional ABMs treat safety as a network outcome. This framework makes it a demand input. The next step is counterfactual policy evaluation.'
new_bottom='The key advance: traditional ABMs treat safety as a network outcome. This framework makes it a demand input. The next step is counterfactual policy evaluation.<br><br><b>Surveying with safety:</b> Future household travel surveys should include stated-preference questions on crash risk perception, route avoidance behaviour, and willingness to pay for safer infrastructure. This would allow the model to move from objective crash exposure (SPF-predicted) to perceived safety, strengthening the behavioural foundation of the CRASH coefficients.'
html=html.replace(old_bottom,new_bottom)
print('Added surveying to future work')

# 2. Find future work slide end and add questions slide after it
idx=html.find('What comes next')
s=html.rfind('<section ',0,idx)
depth=1;i=s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e=i-1

questions='''\
      <section class="slide dark-slide">
        <div style="flex:1; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
          <div class="eyebrow">Thank you</div>
          <h1 class="slide-title" style="font-size:clamp(1.8rem,3.4vw,2.8rem); color:#FFFFFF;">Questions &amp; Discussion</h1>
          <p style="font-family:var(--font-mono); font-size:0.8rem; color:#9FADE0; margin-top:1.2em;">z.ahern@qut.edu.au</p>
        </div>
      </section>'''

html=html[:e]+'\n\n'+questions+html[e:]
print('Added questions slide')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
