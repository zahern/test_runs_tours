html=open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html',encoding='utf-8').read()
idx=html.find('showFunc')
sec_end=html.find('</section>',idx)
if sec_end>0:
    script='''

<style>
.func-btn{padding:3px 8px;border:1px solid var(--border);border-radius:12px;background:var(--surface);color:var(--text);cursor:pointer;font-size:0.55rem;transition:all .2s;}
.func-btn.active,.func-btn:hover{background:var(--primary);color:#fff;border-color:var(--primary);}
</style>
<script>
var funcDetails=[
  "<b>CDAP (Stage 0)</b><br><span style=\\"font-family:var(--font-mono);\\">MNL over H / N / M</span><br>P(pattern) \u221d exp(\u03b2\u2032x<sub>person</sub>)<br><span style=\\"color:var(--text-faint);\\">3 alternatives \u00b7 ~40 candidates \u00b7 18 selected</span>",
  "<b>Activity frequency (Stage 1)</b><br><span style=\\"font-family:var(--font-mono);\\">Count: Poisson / NB / Ordered Logit</span><br>E[n<sub>tours</sub>] = exp(\u03b2\u2032x)<br><span style=\\"color:var(--text-faint);\\">8 purposes \u00d7 ptype segments \u00b7 ~30 candidates \u00b7 11h49m</span>",
  "<b>Day budget (Stage 2)</b><br><span style=\\"font-family:var(--font-mono);\\">OLS / Lognormal</span><br>B<sub>day</sub> = \u03b2\u2032x + \u03b5<br><span style=\\"color:var(--text-faint);\\">Start, end, total B \u00b7 ~25 candidates \u00b7 1m44s</span>",
  "<b>Activity duration (Stage 3)</b><br><span style=\\"font-family:var(--font-mono);\\">Log-normal per purpose</span><br>ln(dur) ~ N(\u03b2\u2032x, \u03c3\u00b2)<br><span style=\\"color:var(--text-faint);\\">8 purposes \u00b7 ~25 candidates \u00b7 57s</span>",
  "<b>Schedule placement (Stage 4)</b><br><span style=\\"font-family:var(--font-mono);\\">MNL / MXL over 6 periods</span><br>P(period<sub>p</sub>) \u221d exp(\u03b2\u2032x + \u03b7<sub>k</sub>\u03be<sub>k</sub>)<br><span style=\\"color:var(--text-faint);\\">Random coefficients optional \u00b7 ~20 candidates \u00b7 5h07m</span>",
  "<b>Tour mode choice (Stage 5a)</b><br><span style=\\"font-family:var(--font-mono);\\">Nested Logit: Motorised / Non-Motorised</span><br>U<sub>m</sub> = \u03b2<sub>tt</sub>\u00b7TT + \u03b2<sub>crash</sub>\u00b7CRASH + ...<br><span style=\\"color:var(--text-faint);\\">5 modes, 2 nests \u00b7 ~40 candidates \u00b7 \u03c1\u00b2=0.42-0.63</span>",
  "<b>Destination choice (Stage 5b)</b><br><span style=\\"font-family:var(--font-mono);\\">MNL over 371 SA2 zones</span><br>V<sub>j</sub> = \u03b2<sub>size</sub>\u00b7size<sub>j</sub> + \u03b2<sub>dist</sub>\u00b7log(DIST+1) + ... + \u03b2<sub>crash</sub>\u00b7CRASH<sub>ij</sub><br><span style=\\"color:var(--text-faint);\\">371 alternatives \u00b7 ~50 candidates \u00b7 \u03c1\u00b2=0.41-0.69</span>"
];
function showFunc(n,btn){
  document.querySelectorAll('.func-btn').forEach(function(b){b.classList.remove('active');});
  btn.classList.add('active');
  document.getElementById('func-detail').innerHTML=funcDetails[n];
}
</script>'''
    html=html[:sec_end]+script+html[sec_end:]
    print('Added stage buttons JS')
with open(r'Z:\test_runs_tours\abm_safety\model_review_deck.html','w',encoding='utf-8') as f:
    f.write(html)
print('Done')
