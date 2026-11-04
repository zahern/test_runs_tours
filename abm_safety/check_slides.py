import re
html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
sections=list(re.finditer(r'<section class=.slide',html))
for i,m in enumerate(sections[:12]):
    s=m.start()
    e=html.find('</section>',s)
    if e<0: e=s+200
    content=html[s:e]
    h1=re.search(r'<h1[^>]*>(.*?)</h1>',content,re.DOTALL)
    title=re.sub(r'<[^>]+>','',h1.group(1) if h1 else '(no h1)')[:120]
    title=title.replace('\n',' ').strip()
    print(f'Slide {i+1} ({s}-{e}): {title}')
