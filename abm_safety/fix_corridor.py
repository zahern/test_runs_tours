html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
html=html.replace('gravitate toward corridors','gravitate toward OD pairs with higher crash exposure')
html=html.replace('high-crash corridors','higher-crash OD pairs')
html=html.replace('safer corridors','safer OD pairs')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done - replaced corridor with OD pairs')
