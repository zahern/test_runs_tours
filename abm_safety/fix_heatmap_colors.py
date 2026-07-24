import re
path = r'Z:\test_runs_tours\abm_safety\model_review_deck.html'
with open(path, encoding='utf-8') as f:
    html = f.read()

old_func = 'function primaryRGB()'
idx = html.find(old_func)
if idx < 0:
    print('primaryRGB not found'); exit()

end_marker = 'paintTabs(xtabTabsEl, curX);'
end_idx = html.find(end_marker, idx)
close_idx = html.find('}', end_idx)
next_end = html.find('\n      xtabTabsEl.innerHTML', close_idx)
if next_end < 0:
    next_end = html.find('xtabTabsEl.innerHTML', close_idx)
replace_end = next_end if next_end > close_idx else close_idx + 2

new_block = '''function heatColor(t) {
        t = Math.max(0, Math.min(1, t));
        if (t < 0.33) { var s = t / 0.33; return [Math.round(36+150*s), Math.round(80+100*s), Math.round(160-30*s)]; }
        if (t < 0.66) { var s = (t-0.33)/0.33; return [Math.round(186+30*s), Math.round(180-80*s), Math.round(130-100*s)]; }
        var s = (t-0.66)/0.34; return [Math.round(216+20*s), Math.round(100-60*s), Math.round(30+10*s)];
      }

      function renderXtab() {
        const cellByKey = {};
        xt.cells.forEach(c => { cellByKey[c.income + '|' + c.vehicles] = c; });
        const maxV = Math.max(...xt.cells.map(c => c[curX] || 0), 1);

        let html = '<div style="display:grid; grid-template-columns: auto repeat(' +
          xt.col_order.length + ', 1fr); gap:3px; font-size:0.6rem;">';
        html += '<div></div>';
        xt.col_order.forEach(v => {
          html += '<div style="text-align:center;color:var(--text-muted);padding:2px;white-space:nowrap;">' +
            v.replace(' vehicles', 'v').replace(' vehicle', 'v') + '</div>';
        });
        xt.row_order.forEach(inc => {
          html += '<div style="color:var(--text-muted);display:flex;align-items:center;padding-right:4px;white-space:nowrap;">' +
            inc.replace('Bands ', '').replace(' (lowest)', '').replace(' (highest)', '') + '</div>';
          xt.col_order.forEach(veh => {
            const c = cellByKey[inc + '|' + veh];
            const val = c ? (c[curX] || 0) : null;
            const t = val == null ? -1 : Math.min(1, val / maxV);
            const [rr, gg, bb] = t < 0 ? [200, 200, 200] : heatColor(t);
            const title = c
              ? (XMETRIC_LABEL[curX] + ': ' + val + '% \u00b7 ' + fmt(c.trips) + ' trips')
              : 'No data';
            html += '<div title="' + title + '" style="aspect-ratio:1; border-radius:4px; background: rgb(' +
              rr + ',' + gg + ',' + bb + '); border:1px solid var(--border);' +
              (t < 0 ? 'opacity:0.35;' : '') + '"></div>';
          });
        });
        html += '</div>';
        html += '<div style="display:flex;align-items:center;gap:0.4em;margin-top:0.7em;font-size:0.6rem;color:var(--text-muted);">' +
          '<span>Low</span><div style="flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg, rgb(36,80,130), rgb(186,180,130), rgb(216,100,30), rgb(236,40,40));"></div><span>High</span></div>';
        xtabEl.innerHTML = html;
        paintTabs(xtabTabsEl, curX);
      }'''

html = html[:idx] + new_block + html[replace_end:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'OK - replaced heatmap colors ({len(new_block)} chars)')
