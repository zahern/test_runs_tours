"""
deck_edit.py — edit slide text in the presentation decks without touching HTML.

The decks (model_review_deck.html, intergrade_model_treivewal.html) are plain
HTML slide decks: each slide is a <section class="slide"> with an
<h2 class="slide-kicker"> and an <h1 class="slide-title">. This CLI lists the
slides and edits their text with plain-Python regex (no bs4 dependency),
always writing a .bak backup first.

Usage (from abm_safety/):
    python deck_edit.py --list
        -> numbered slides with kicker + title (default deck: model_review_deck.html)

    python deck_edit.py --slide 5 --field title  --set "New slide title"
    python deck_edit.py --slide 5 --field kicker --set "New kicker text"

    python deck_edit.py --replace "old phrase" "new phrase"
        -> replace text ANYWHERE in the deck body (exact match, all hits)
    python deck_edit.py --slide 5 --replace "old phrase" "new phrase"
        -> same, limited to slide 5

    python deck_edit.py --deck intergrade_model_treivewal.html --list

Auto-generated slides (SA2 patterns, safety outcome, person day) are refreshed
by their generator scripts between sentinel comments - edits inside them will
be overwritten on the next pipeline run; the tool warns when a slide is one.
"""
import argparse
import html as html_mod
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SECTION_RE = re.compile(r'<section class="slide[^"]*"[^>]*>.*?</section>', re.DOTALL)
TITLE_RE = re.compile(r'(<h1 class="slide-title"[^>]*>)(.*?)(</h1>)', re.DOTALL)
KICKER_RE = re.compile(r'(<h2 class="slide-kicker"[^>]*>)(.*?)(</h2>)', re.DOTALL)
SENTINEL_RE = re.compile(r'SLIDE_START|auto-refreshed')


def _strip_tags(s):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', s)).strip()


def load(deck_path):
    with open(deck_path, encoding='utf-8') as f:
        return f.read()


def save(deck_path, html):
    shutil.copy2(deck_path, deck_path + '.bak')
    with open(deck_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'saved (backup at {os.path.basename(deck_path)}.bak)')


def list_slides(html):
    for i, m in enumerate(SECTION_RE.finditer(html), start=1):
        sec = m.group(0)
        t = TITLE_RE.search(sec)
        k = KICKER_RE.search(sec)
        auto = '  [auto-generated - edits will be overwritten]' \
            if SENTINEL_RE.search(sec) else ''
        print(f'{i:3d}. [{_strip_tags(k.group(2)) if k else "-"}]  '
              f'{_strip_tags(t.group(2)) if t else "(no title)"}{auto}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deck', default='model_review_deck.html',
                    help='deck filename in abm_safety/ (or absolute path)')
    ap.add_argument('--list', action='store_true', help='list slides')
    ap.add_argument('--slide', type=int, help='1-based slide number to target')
    ap.add_argument('--field', choices=['title', 'kicker'],
                    help='with --set: which heading to replace')
    ap.add_argument('--set', dest='set_text', help='new text for --field')
    ap.add_argument('--replace', nargs=2, metavar=('OLD', 'NEW'),
                    help='replace exact text (whole deck, or one slide with --slide)')
    args = ap.parse_args()

    deck_path = args.deck if os.path.isabs(args.deck) else os.path.join(HERE, args.deck)
    if not os.path.exists(deck_path):
        sys.exit(f'deck not found: {deck_path}')
    html = load(deck_path)
    sections = list(SECTION_RE.finditer(html))

    if args.list or not (args.set_text or args.replace):
        print(f'{os.path.basename(deck_path)} - {len(sections)} slides\n')
        list_slides(html)
        return

    def _slide_span():
        if args.slide is None or not (1 <= args.slide <= len(sections)):
            sys.exit(f'--slide must be 1..{len(sections)}')
        m = sections[args.slide - 1]
        if SENTINEL_RE.search(m.group(0)):
            print('WARNING: this slide is auto-generated; the next pipeline '
                  'run will overwrite your edit.')
        return m.span()

    if args.set_text:
        if not args.field:
            sys.exit('--set requires --field title|kicker')
        s, e = _slide_span()
        sec = html[s:e]
        rx = TITLE_RE if args.field == 'title' else KICKER_RE
        if not rx.search(sec):
            sys.exit(f'slide {args.slide} has no {args.field}')
        new_sec = rx.sub(lambda m: m.group(1) + html_mod.escape(args.set_text) + m.group(3),
                         sec, count=1)
        save(deck_path, html[:s] + new_sec + html[e:])
        print(f'slide {args.slide} {args.field} -> {args.set_text!r}')
        return

    if args.replace:
        old, new = args.replace
        if args.slide is not None:
            s, e = _slide_span()
            n = html[s:e].count(old)
            if not n:
                sys.exit(f'text not found in slide {args.slide}: {old!r}')
            save(deck_path, html[:s] + html[s:e].replace(old, new) + html[e:])
        else:
            n = html.count(old)
            if not n:
                sys.exit(f'text not found in deck: {old!r}')
            save(deck_path, html.replace(old, new))
        print(f'replaced {n} occurrence(s) of {old!r}')


if __name__ == '__main__':
    main()
