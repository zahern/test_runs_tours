html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
# Find the si-provenance element with source info
idx=html.find('id="si-provenance"')
if idx>0:
    end=html.find('</p>',idx)+4
    content=html[idx:end]
    if 'safety_nosafety' in content or 'Source:' in content or 'stage6_output' in content:
        html=html[:idx]+html[end:]
        print('Removed source provenance line')
    else:
        print(f'Line content: {content[:120]}...')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
