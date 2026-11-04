"""
gen_pipeline_diagram.py — generate the ABM pipeline workflow SVG from a spec
============================================================================
Regenerate the pipeline diagram yourself:  ``python gen_pipeline_diagram.py``
→ writes ``pipeline_model_workflow.svg`` (open in any browser / Word / Inkscape).

To EDIT the diagram, change the SPEC section below and re-run — the layout
(box positions, phase bands, GA bus, logsum rail, arrows) is computed
automatically, so you never touch coordinates:

  • add / remove / reorder a model → edit a phase's ``boxes`` list
  • move a model to another phase   → move its tuple between phases
  • put a model on the GA bus        → 4th tuple field ``ga=True``
  • feed it the accessibility logsum → 5th field ``logsum=True``
  • rename / recolor a phase         → edit ``name`` / ``color``
  • change a data input or its target→ edit ``DATA_INPUTS``
Everything is pure-Python string building — no dependencies.

Each box tuple is:  (title, subtitle, ga, logsum)
  ga=True     → connected to the red GA hypothesis-testing bus (searched model)
  logsum=True → receives the green accessibility-logsum rail
"""
from __future__ import annotations

# ============================================================================
# SPEC — edit this, then re-run.
# ============================================================================
PHASES = [
    {"name": "1 · pre-day", "color": "#8e7cc3", "boxes": [
        ("Accessibility (lr2)",        "computes mode / dest logsums", True,  False),
        ("Mobility attributes",        "mobility_models.py",           True,  True),
        ("Usual locations (lr3)",      "usual_location.py · work/school", True, True),
        ("WFH / telecommute (lr4)",    "wfh_telecommute.py",           True,  True),
    ]},
    {"name": "2 · day pattern", "color": "#3aa6a0", "boxes": [
        ("Stage 0 · CDAP + household", "cdap.py · cdap_households.py",  True,  True),
        ("Tour-purpose sequencing",    "tour_sequence.py",             True,  False),
    ]},
    {"name": "3 · tour & trip demand", "color": "#d1495b", "boxes": [
        ("Stage 1 · Activity frequency", "stage1_activity_frequency.py", True, True),
        ("Stage 2 · Day budget B",       "stage2_day_duration.py",       True, True),
        ("Stage 3 · Activity duration",  "MDCEV / AFT / RRM",            True, True),
        ("Stage 4 · Timing",             "hybrid RUM–RRM",               True, True),
        ("Stage 5 · Destination + mode", "logsum · crash · GP smooth",   True, True),
        ("Stage 5b · Trip stops",        "stage5b_trip_destination.py",  True, False),
        ("Stage 6 · Trip mode",          "nested logit · tour-mode consistent", True, False),
    ]},
    {"name": "4 · outputs", "color": "#8a94a6", "boxes": [
        ("Outputs + saved models",  "*_models.pkl · *_coefficients.csv", False, False),
        ("DTA / assignment",        "run_full_pipeline_dta_tested.py",   False, False),
    ]},
]

# (title, subtitle, feeds_target_title)  — target must match a box title above
DATA_INPUTS = [
    ("HHTS survey",      "estimation data",       "Stage 0 · CDAP + household"),
    ("Synthetic pop",    "build_synth_population", "Mobility attributes"),
    ("Land use",         "employment · ACDC",     "Accessibility (lr2)"),
    ("Skims (OMX)",      "time · cost · safety",  "Stage 5 · Destination + mode"),
]

GA_LABEL = ("GA hypothesis testing", "ga_hypothesis_testing.py",
            "ga_constraints.py", "portfolio: Bandit-SA / GA")

# ---- geometry (tweak if you want more/less spacing) ------------------------
BOX_W, BOX_H, GAP = 360, 52, 14          # spine box size + vertical gap
SPINE_X = 360                            # left edge of the spine column
TOP = 110                                # y of the first spine box
DATA_Y = 20                              # y of the data-input row

C = {"box": "#ffffff", "stroke": "#c8cdd6", "text": "#1c2330", "muted": "#5b6472",
     "arrow": "#8a94a6", "feed": "#5b8def", "ga": "#d1495b", "ls": "#2a9d8f",
     "data": "#5b8def"}


# ============================================================================
# RENDERER — you shouldn't need to edit below here.
# ============================================================================
def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg() -> str:
    # flatten boxes with computed y + phase membership
    flat = []
    y = TOP
    for pi, ph in enumerate(PHASES):
        for (title, sub, ga, ls) in ph["boxes"]:
            flat.append(dict(title=title, sub=sub, ga=ga, ls=ls, phase=pi, y=y))
        y += 0  # phases are contiguous; extra inter-phase gap added below
    # assign y with a slightly larger gap between phases
    y = TOP
    prev_phase = None
    for b in flat:
        if prev_phase is not None and b["phase"] != prev_phase:
            y += 10
        b["y"] = y
        y += BOX_H + GAP
        prev_phase = b["phase"]
    by_title = {b["title"]: b for b in flat}
    cx = SPINE_X + BOX_W / 2
    right = SPINE_X + BOX_W
    total_h = flat[-1]["y"] + BOX_H

    S = []
    W = 1160
    H = total_h + 150
    S.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'role="img" aria-label="ABM pipeline workflow" '
             f'font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">')
    S.append(f'''<style>
  .bx{{fill:{C['box']};stroke:{C['stroke']};stroke-width:1.5}}
  .t{{fill:{C['text']};font-weight:650;font-size:15px}}
  .s{{fill:{C['muted']};font-size:11.5px;font-family:ui-monospace,Menlo,Consolas,monospace}}
  .blab{{font-weight:700;font-size:13px}}
  .a{{stroke:{C['arrow']};stroke-width:2;fill:none;marker-end:url(#ah)}}
  .f{{stroke:{C['feed']};stroke-width:1.6;stroke-dasharray:5 4;fill:none;marker-end:url(#af)}}
  .bus{{stroke:{C['ga']};stroke-width:1.8;stroke-dasharray:5 4;fill:none}}
  .ls{{stroke:{C['ls']};stroke-width:1.8;fill:none;marker-end:url(#al)}}
  .lsrail{{stroke:{C['ls']};stroke-width:2.4;fill:none}}
</style>
<defs>
  <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{C['arrow']}"/></marker>
  <marker id="af" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{C['feed']}"/></marker>
  <marker id="al" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{C['ls']}"/></marker>
</defs>''')

    # phase bands + rotated labels
    for pi, ph in enumerate(PHASES):
        ys = [b["y"] for b in flat if b["phase"] == pi]
        y0, y1 = min(ys) - 6, max(ys) + BOX_H + 6
        S.append(f'<rect x="{SPINE_X-30}" y="{y0}" width="{BOX_W+60}" '
                 f'height="{y1-y0}" rx="10" fill="{ph["color"]}" opacity="0.10"/>')
        S.append(f'<text class="blab" fill="{ph["color"]}" '
                 f'transform="translate({SPINE_X-38},{(y0+y1)/2}) rotate(-90)" '
                 f'text-anchor="middle">{_esc(ph["name"])}</text>')

    # data inputs (top row) + feeds
    n = len(DATA_INPUTS)
    dw = 220
    step = (W - 60) / n
    for i, (title, sub, tgt) in enumerate(DATA_INPUTS):
        dx = 30 + i * step
        S.append(f'<rect class="bx" x="{dx:.0f}" y="{DATA_Y}" width="{dw:.0f}" height="52" rx="8" stroke="{C["data"]}"/>')
        S.append(f'<text class="t" x="{dx+dw/2:.0f}" y="{DATA_Y+22}" text-anchor="middle">{_esc(title)}</text>')
        S.append(f'<text class="s" x="{dx+dw/2:.0f}" y="{DATA_Y+40}" text-anchor="middle">{_esc(sub)}</text>')
        if tgt in by_title:
            b = by_title[tgt]
            S.append(f'<path class="f" d="M{dx+dw/2:.0f},{DATA_Y+52} V{b["y"]-8} '
                     f'H{SPINE_X if dx+dw/2 < cx else right} '
                     f'{"" if dx+dw/2 < cx else ""}L{SPINE_X if dx+dw/2<cx else right},{b["y"]+BOX_H/2:.0f}"/>')

    # spine boxes + arrows
    for i, b in enumerate(flat):
        yv = b["y"]
        S.append(f'<rect class="bx" x="{SPINE_X}" y="{yv}" width="{BOX_W}" height="{BOX_H}" rx="8"/>')
        S.append(f'<text class="t" x="{cx:.0f}" y="{yv+22}" text-anchor="middle">{_esc(b["title"])}</text>')
        S.append(f'<text class="s" x="{cx:.0f}" y="{yv+40}" text-anchor="middle">{_esc(b["sub"])}</text>')
        if i + 1 < len(flat):
            S.append(f'<path class="a" d="M{cx:.0f},{yv+BOX_H} V{flat[i+1]["y"]}"/>')

    # GA bus (right) over ga=True boxes
    ga_boxes = [b for b in flat if b["ga"]]
    if ga_boxes:
        bx = right + 50
        y0 = ga_boxes[0]["y"] + BOX_H / 2
        y1 = ga_boxes[-1]["y"] + BOX_H / 2
        S.append(f'<path class="bus" d="M{bx},{y0:.0f} V{y1:.0f}"/>')
        for b in ga_boxes:
            S.append(f'<path class="bus" d="M{right},{b["y"]+BOX_H/2:.0f} H{bx}"/>')
        gx, gy = bx + 60, (y0 + y1) / 2 - 60
        S.append(f'<rect class="bx" x="{gx}" y="{gy:.0f}" width="280" height="120" rx="10" stroke="{C["ga"]}" stroke-dasharray="6 4"/>')
        S.append(f'<text class="t" x="{gx+140}" y="{gy+34:.0f}" text-anchor="middle" fill="{C["ga"]}">{_esc(GA_LABEL[0])}</text>')
        for j, ln in enumerate(GA_LABEL[1:]):
            S.append(f'<text class="s" x="{gx+140}" y="{gy+56+j*18:.0f}" text-anchor="middle">{_esc(ln)}</text>')
        S.append(f'<path class="bus" d="M{gx},{gy+60:.0f} H{bx}"/>')

    # logsum rail (left) over ls=True boxes, sourced at the first ls box's producer (accessibility = first box)
    ls_boxes = [b for b in flat if b["ls"]]
    if ls_boxes:
        rx = SPINE_X - 60
        src = flat[0]                     # accessibility box produces the logsum
        y0 = src["y"] + BOX_H / 2
        y1 = ls_boxes[-1]["y"] + BOX_H / 2
        S.append(f'<path class="lsrail" d="M{cx:.0f},{y0:.0f} C {rx},{y0+20:.0f} {rx},{y0+60:.0f} {rx},{min(y0+90,y1):.0f} V{y1:.0f}"/>')
        for b in ls_boxes:
            S.append(f'<path class="ls" d="M{rx},{b["y"]+BOX_H/2:.0f} H{SPINE_X}"/>')
        S.append(f'<text class="blab" fill="{C["ls"]}" transform="translate({rx-14},{(y0+y1)/2:.0f}) rotate(-90)" text-anchor="middle">accessibility logsum</text>')

    # skim loop: last box → back to the skims data input
    last = flat[-1]
    S.append(f'<path class="f" d="M{SPINE_X},{last["y"]+BOX_H/2:.0f} H150 V{DATA_Y+52}"/>')

    # legend
    ly = total_h + 40
    S.append(f'<g font-size="12.5px" fill="{C["muted"]}">')
    S.append(f'<rect x="30" y="{ly}" width="14" height="14" rx="3" fill="{C["data"]}" opacity="0.5"/><text x="50" y="{ly+12}">data</text>')
    S.append(f'<line x1="110" y1="{ly+7}" x2="150" y2="{ly+7}" class="ls"/><text x="158" y="{ly+12}">accessibility logsum</text>')
    S.append(f'<line x1="310" y1="{ly+7}" x2="350" y2="{ly+7}" class="bus"/><text x="358" y="{ly+12}">GA bus (searched models)</text>')
    S.append(f'<line x1="560" y1="{ly+7}" x2="600" y2="{ly+7}" class="f"/><text x="608" y="{ly+12}">data / skim loop</text>')
    S.append('</g>')
    S.append('</svg>')
    return "\n".join(S)


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__) or '.', 'pipeline_model_workflow.svg')
    svg = build_svg()
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(svg)
    print(f"wrote {out}  ({len(svg)} bytes, "
          f"{sum(len(p['boxes']) for p in PHASES)} model boxes, "
          f"{len(PHASES)} phases)")
