"""One-shot patch: replace the broken person-movement-plots section."""
import re

with open('pipeline_logger.py', encoding='utf-8') as fh:
    content = fh.read()

# Marker lines that bracket the section to replace
START = '    # Sample person-movement plots'
END   = '    sys.stdout.flush()\n    return plan_coords'

i = content.index(START)
j = content.index(END, i) + len(END)

old_section = content[i:j]
print('=== OLD SECTION (first 100 chars) ===')
print(repr(old_section[:200]))

new_section = '''    # Sample person-movement plots — representative travellers (not outliers)
    try:
        if 'person_id' in plan_coords.columns:
            rt_col = 'row_type' if 'row_type' in plan_coords.columns else None
            trip_rows = plan_coords[plan_coords[rt_col] == 'trip'] if rt_col else plan_coords
            # Use haversine dist_km for ranking (always computed; skim dist may have
            # artefacts for edge OD pairs that produce unrealistically long trips).
            _dist_col = 'dist_km' if 'dist_km' in trip_rows.columns \
                        else ('dist' if 'dist' in trip_rows.columns else None)
            if _dist_col and len(trip_rows) > 0:
                _dist_per_person = (trip_rows.groupby('person_id')[_dist_col]
                                    .sum().dropna())
                _p10 = float(_dist_per_person.quantile(0.10))
                _p90 = float(_dist_per_person.quantile(0.90))
                # Select from the bulk of travellers, not extreme outliers
                _repr = _dist_per_person[
                    (_dist_per_person >= _p10) & (_dist_per_person <= _p90)
                ]
                if len(_repr) == 0:
                    _repr = _dist_per_person
                sample_pids = _repr.sample(
                    min(4, len(_repr)), random_state=42
                ).index.tolist()
                logger.info(
                    f"  Movement plots: {len(sample_pids)} representative travellers "
                    f"(daily dist_km p10={_p10:.1f}-p90={_p90:.1f} km)"
                )
            else:
                sample_pids = []
            for pid in sample_pids:
                try:
                    _sp = os.path.join(_pdir, f'person_{pid}_movement.png')
                    plot_person_movement(plan_coords, pid, save_path=_sp)
                    _saved.append(f'person_{pid}_movement.png')
                    logger.info(f"  Movement plot -> {_sp}")
                except Exception as _mpe:
                    logger.warning(f"  Movement plot for {pid} failed: {_mpe}")
    except Exception as _e:
        logger.warning(f"  Person movement plots failed: {_e}")

    # Distance-linked sample plans: re-draw Gantt charts with dist_km annotations
    try:
        from plot_activity_plans import plot_sample_plans as _psp
        _spdist_path = os.path.join(_pdir, 'sample_plans_with_distances.png')
        _psp(plan_coords, n=8, save_path=_spdist_path, seed=42)
        _saved.append('sample_plans_with_distances.png')
        logger.info(f"  Distance-linked sample plans -> {_spdist_path}")
    except Exception as _spe:
        logger.warning(f"  sample_plans_with_distances failed: {_spe}")

    logger.info(f"  Spatial plots complete: {len(_saved)} files in {_pdir}/")
    sys.stdout.flush()
    return plan_coords'''

patched = content[:i] + new_section + content[j:]
with open('pipeline_logger.py', 'w', encoding='utf-8') as fh:
    fh.write(patched)
print('DONE — section replaced successfully')
