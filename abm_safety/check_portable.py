import re
html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
urls=set(re.findall(r'(https?://[^\s"\'<>]+)',html))
if urls:
    print(f'External URLs ({len(urls)}):')
    for u in sorted(urls):
        print(f'  {u}')
else:
    print('No external URLs found')

cdns=['unpkg','cdn','cloudflare','googleapis','jsdelivr','cartocdn','openstreetmap']
found=False
for c in cdns:
    if c in html:
        print(f'WARNING: {c} found')
        found=True

local=re.findall(r'(src|href)="[A-Za-z]:\\\\',html)
if local:
    print(f'WARNING: {len(local)} Windows paths found')

if not found and not local:
    print('Fully self-contained - safe to move')
