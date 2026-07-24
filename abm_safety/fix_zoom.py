html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()

# 1. Fix zoom function
old_zoom='    var deckZoom=1;\n    function zoomDeck(delta){\n      deckZoom=Math.max(0.5,Math.min(2,deckZoom+delta));\n      document.getElementById(\'deck\').style.transform=\'scale(\'+deckZoom+\')\';\n      document.getElementById(\'deck\').style.transformOrigin=\'top center\';\n    }'
new_zoom='    var deckZoom=1;\n    function zoomDeck(delta){\n      deckZoom=Math.max(0.5,Math.min(2,deckZoom+delta));\n      var el=document.querySelector(\'.stage\');\n      if(el){el.style.transform=\'scale(\'+deckZoom+\')\';el.style.transformOrigin=\'top center\';}\n    }'
html=html.replace(old_zoom,new_zoom)
print('Fixed zoom function')

# 2. Fix future work slide - make it dark
idx=html.find('Counterfactual testing becomes possible')
s=html.rfind('<section ',0,idx)
slide_tag_end=html.find('>',s)+1
old_tag=html[s:slide_tag_end]
if 'dark-slide' not in old_tag:
    new_tag=old_tag.replace('class="slide"','class="slide dark-slide"')
    html=html[:s]+new_tag+html[slide_tag_end:]
    print('Made future work slide dark')
else:
    print('Already dark')

with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
