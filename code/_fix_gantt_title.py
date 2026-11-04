"""Patch the title section to add total distance."""

with open('plot_activity_plans.py', encoding='utf-8') as fh:
    content = fh.read()

# Identify the exact title block using a unique anchor that won't need unicode
anchor = "        # Mode summary from trip rows\n        if not trips_p.empty and 'mode_name' in trips_p.columns:\n            mc = trips_p['mode_name'].value_counts()\n            mode_summary = '  '.join(f'{m}"

idx = content.find(anchor)
print(f'Anchor found at index: {idx}')
if idx == -1:
    raise RuntimeError('Anchor not found')

# Insert total distance block just before the "# Mode summary" comment
insert_before = "        # Mode summary from trip rows"
total_dist_block = """        # Total daily distance (prefer haversine dist_km)
        _total_km = None
        if not trips_p.empty:
            _d_col = 'dist_km' if 'dist_km' in trips_p.columns else \\
                     ('dist' if 'dist' in trips_p.columns else None)
            if _d_col is not None:
                _vals = trips_p[_d_col].dropna()
                if len(_vals) > 0:
                    _total_km = float(_vals.sum())
        _dist_title = f'  \u00b7  {_total_km:.1f}km total' if _total_km is not None else ''

        """

assert insert_before in content, f'{insert_before!r} not found'
content = content.replace(insert_before, total_dist_block + insert_before, 1)
print('Total distance block inserted.')

# Now update the ax.set_title call to include _dist_title.
# Find the exact ax.set_title(...) block after the anchor
# We need to add _dist_title between the activity line and the modes line.
# The current structure (using actual unicode chars in the file) is:
#   f'Person {pid}  ·  {ptype}  ·  vehicles: {vehs}  ·  '
#   f'{n_a} activities  ·  {t_start_str}–{t_end_str}'
#   + (f'  ·  modes: {mode_summary}' if mode_summary else ''),

# Find ax.set_title after our anchor
title_search = "        ax.set_title(\n"
title_idx = content.find(title_search, idx)
print(f'ax.set_title found at index: {title_idx}')

# Read ~15 lines after ax.set_title to see what's there
end_of_title = content.find('\n        )\n', title_idx)
title_block = content[title_idx:end_of_title + len('\n        )\n')]
print('Current title block:')
print(repr(title_block))

# Replace the concat to add _dist_title
# Current: + (f'  ·  modes: {mode_summary}' if mode_summary else ''),
# New:     + _dist_title\n            + (f'  ·  modes: {mode_summary}' if mode_summary else ''),
old_concat = "            + (f'  \u00b7  modes: {mode_summary}' if mode_summary else ''),"
new_concat = "            + _dist_title\n            + (f'  \u00b7  modes: {mode_summary}' if mode_summary else ''),"

if old_concat in content:
    content = content.replace(old_concat, new_concat, 1)
    print('Title concat patched.')
else:
    print(f'WARNING: old_concat not found: {repr(old_concat)}')
    # Show context
    snippet = content[title_idx:title_idx+400]
    print('Context:')
    print(repr(snippet))

with open('plot_activity_plans.py', 'w', encoding='utf-8') as fh:
    fh.write(content)
print('DONE')
