html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
idx=html.find('id="sa2-slide"')
s=html.rfind('<section ',0,idx)
depth=1;i=s+1
while depth>0 and i<len(html):
    if html[i:i+9]=='<section ':depth+=1
    elif html[i:i+10]=='</section>':depth-=1
    i+=1
e=i-1
trail=html[e:e+5000]
extra_end=e
if '<style>' in trail[:1000] and '</style>' in trail[:2000]:
    extra_end=e+trail.index('</style>')+len('</style>')
if '<script>' in trail[:2000] and '</script>' in trail[:3000]:
    si=trail.index('</script>')
    extra_end=max(extra_end,e+si+len('</script>'))
html=html[:s]+html[extra_end:]
print(f'Removed SA2 patterns slide ({extra_end-s} bytes)')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
