"""Patch plot_activity_plans.py to add distance labels to trip bars and totals to titles."""

with open('plot_activity_plans.py', encoding='utf-8') as fh:
    content = fh.read()

# --- 1. Add distance label to trip bars ---
old_trip = '''                if t_end > t_start:
                    ax.barh(y=-0.3, width=t_end-t_start, left=t_start,
                            height=0.25, color=color, alpha=0.7, zorder=2)
                    if (t_end - t_start) > 0.15 and mode_n:
                        ax.text(t_start + (t_end - t_start)/2, -0.3,
                                mode_n, ha='center', va='center',
                                fontsize=6, color='white', fontweight='bold',
                                zorder=3)'''

new_trip = '''                # Distance for this trip leg (prefer haversine dist_km over skim dist)
                _raw_d = tr.get('dist_km') if 'dist_km' in tr.index else tr.get('dist')
                _has_d = (_raw_d is not None and
                          not (isinstance(_raw_d, float) and _math_module.isnan(float(_raw_d))))
                _dist_str = f' {float(_raw_d):.1f}km' if _has_d else ''
                if t_end > t_start:
                    ax.barh(y=-0.3, width=t_end-t_start, left=t_start,
                            height=0.25, color=color, alpha=0.7, zorder=2)
                    if (t_end - t_start) > 0.15 and (mode_n or _dist_str):
                        ax.text(t_start + (t_end - t_start)/2, -0.3,
                                mode_n + _dist_str, ha='center', va='center',
                                fontsize=6, color='white', fontweight='bold',
                                zorder=3)'''

assert old_trip in content, 'trip label section not found'
content = content.replace(old_trip, new_trip, 1)
print('Trip distance label patch applied.')

# --- 2. Add total distance to chart title ---
old_title = '''        # Title with person summary
        ptype = str(person['ptype'].iloc[0]) if 'ptype' in person.columns else ''
        vehs  = int(person['vehrd'].iloc[0]) if 'vehrd' in person.columns else 0
        n_a   = len(acts_p)
        t_start_str = _hm(person['STARTIME'].min()) if 'STARTIME' in person.columns else '?'
        t_end_str   = _hm(person['ENDTIME'].max())  if 'ENDTIME'  in person.columns else '?'

        # Mode summary from trip rows
        if not trips_p.empty and 'mode_name' in trips_p.columns:
            mc = trips_p['mode_name'].value_counts()
            mode_summary = '  '.join(f'{m}x{c}' for m, c in mc.items() if m)
        else:
            mode_summary = ''

        ax.set_title(
            f'Person {pid}  \\u00b7  {ptype}  \\u00b7  vehicles: {vehs}  \\u00b7  '
            f'{n_a} activities  \\u00b7  {t_start_str}\\u2013{t_end_str}'
            + (f'  \\u00b7  modes: {mode_summary}' if mode_summary else ''),
            fontsize=8, pad=4
        )'''

new_title = '''        # Title with person summary
        ptype = str(person['ptype'].iloc[0]) if 'ptype' in person.columns else ''
        vehs  = int(person['vehrd'].iloc[0]) if 'vehrd' in person.columns else 0
        n_a   = len(acts_p)
        t_start_str = _hm(person['STARTIME'].min()) if 'STARTIME' in person.columns else '?'
        t_end_str   = _hm(person['ENDTIME'].max())  if 'ENDTIME'  in person.columns else '?'

        # Total daily distance (prefer haversine dist_km)
        _total_km = None
        if not trips_p.empty:
            _d_col = 'dist_km' if 'dist_km' in trips_p.columns else \
                     ('dist' if 'dist' in trips_p.columns else None)
            if _d_col is not None:
                _vals = trips_p[_d_col].dropna()
                if len(_vals) > 0:
                    _total_km = float(_vals.sum())
        _dist_title = f'  \u00b7  {_total_km:.1f}km total' if _total_km is not None else ''

        # Mode summary from trip rows
        if not trips_p.empty and 'mode_name' in trips_p.columns:
            mc = trips_p['mode_name'].value_counts()
            mode_summary = '  '.join(f'{m}x{c}' for m, c in mc.items() if m)
        else:
            mode_summary = ''

        ax.set_title(
            f'Person {pid}  \u00b7  {ptype}  \u00b7  vehicles: {vehs}  \u00b7  '
            f'{n_a} activities  \u00b7  {t_start_str}\u2013{t_end_str}'
            + _dist_title
            + (f'  \u00b7  modes: {mode_summary}' if mode_summary else ''),
            fontsize=8, pad=4
        )'''

# Try with the actual Unicode escapes as written in file (which may use actual chars)
if old_title not in content:
    # The file may use actual Unicode characters instead of escape sequences
    old_title2 = old_title.replace('\\u00b7', '\u00b7').replace('\\u2013', '\u2013')
    new_title2 = new_title.replace('\\u00b7', '\u00b7').replace('\\u2013', '\u2013')
    if old_title2 in content:
        content = content.replace(old_title2, new_title2, 1)
        print('Title patch applied (unicode chars).')
    else:
        print('WARNING: title section not found, trying substring search...')
        # Just find the ax.set_title block and work with it
        idx = content.find("        ax.set_title(\n            f'Person {pid}")
        if idx == -1:
            idx = content.find('        ax.set_title(\n            f\'Person {pid}')
        print(f'ax.set_title found at index: {idx}')
        # Find what's actually there
        snippet_start = content.rfind('        # Title with person summary', 0, idx)
        print('Snippet:', repr(content[snippet_start:snippet_start+600]))
else:
    content = content.replace(old_title, new_title, 1)
    print('Title patch applied (escaped).')

with open('plot_activity_plans.py', 'w', encoding='utf-8') as fh:
    fh.write(content)
print('DONE')
