"""Structural validation of the preprint package (no LaTeX engine here)."""
import os
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


def strip_comments(tex):
    return re.sub(r'(?<!\\)%.*', '', tex)


def _read_opt(name):
    p = os.path.join(D, name)
    return open(p, encoding='utf-8').read() if os.path.exists(p) else ''


main = open(os.path.join(D, 'main.tex'), encoding='utf-8').read()
table = _read_opt('table.tex')
figs = _read_opt('figures.tex')
full = main + "\n" + table + "\n" + figs
code = strip_comments(full)

# 1. balanced braces (ignore escaped)
tmp = re.sub(r'\\[{}\\]', '', code)
check("braces balanced", tmp.count('{') == tmp.count('}'),
      f"open={tmp.count('{')} close={tmp.count('}')}")

# 2. environments balanced (excluding document, handled too)
begins = re.findall(r'\\begin\{([^}]+)\}', code)
ends = re.findall(r'\\end\{([^}]+)\}', code)
from collections import Counter
cb, ce = Counter(begins), Counter(ends)
mismatch = {k: (cb[k], ce.get(k, 0)) for k in set(cb) | set(ce)
            if cb[k] != ce.get(k, 0)}
check("environments balanced", not mismatch, str(mismatch) or "")

# 3. includes resolve
for inc in re.findall(r'\\include\{([^}]+)\}', code):
    p = os.path.join(D, inc if inc.endswith('.tex') else inc + '.tex')
    check(f"include {inc}", os.path.exists(p))

# 4. figures exist
for fig in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', code):
    p = os.path.join(D, fig)
    if not os.path.splitext(p)[1]:
        p += '.png'
    check(f"figure {fig}", os.path.exists(p))

# 5. no leftover template/red/test text
check("no lipsum", 'lipsum' not in code.lower())
check("no red author notes", 'textcolor{red}' not in code)
check("no FIXME/XXX", not re.search(r'FIXME|XXX', code))
check("TODOs are explicit status markers",
      not re.search(r'(?<!Status)TODO', code)
      or 'StatusTodo' in code)

# 6. document begin/end exactly once
check("single document env",
      code.count(r'\begin{document}') == 1 and code.count(r'\end{document}') == 1)

# 7. natbib + bibliography present
check("bibliography wired",
      (r'\bibliography{localbibliography.bib}' in code
       or r'\bibliography{localbibliography}' in code)
      and os.path.exists(os.path.join(D, 'localbibliography.bib')))

print()
print("RESULT:", "ALL PASS" if not fails else f"FAIL: {fails}")
sys.exit(0 if not fails else 1)
