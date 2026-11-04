#!/usr/bin/env python
"""
Apply the grouped-draws allocation fix to the INSTALLED metacountregressor on
the cluster, with a backup, idempotently, and prove it worked.

WHY PATCH IN PLACE RATHER THAN INSTALL OR VENDOR.
  - The cluster has no DNS, so `pip install --upgrade` cannot reach PyPI. The
    behavioural gate correctly FAILS there on 1.0.143/1.0.153.
  - The Windows source tree is 1.0.169 and 6067 lines; the installed copy is
    7171 lines. They are different builds, so shadowing the package with the
    source tree would change far more than this one allocation. A minimal,
    auditable edit to one line is the honest option.
  - The install lives in the user's own ~/.local, so it is user-writable.

The fix: build_eta indexes the group coefficient table as
    beta_g = beta_g_all[data["group_ids"]]
so only the first G rows of draws_g are addressable. Allocating _n_draw (the
panel count) rows produced a tensor that was silent dead weight rather than a
crash, because G <= N always holds. Allocate G rows when a group is present.

Install metadata here is self-contradictory -- _version.py says 1.0.143 while
importlib.metadata reports 1.0.153 for the same directory -- so this script
gates on the resulting SHAPE, not on any version string.
"""
import ast
import importlib.util
import os
import re
import shutil
import sys
import time

TARGET_RE = re.compile(
    r'^(?P<ind>\s*)draws_g = generate_halton_normal\(_n_draw, Xg\.shape\[2\], '
    r'R, seed=44\)\s*$')
MARKER = '_g_draws = int(G) if'
NEW_BODY = (
    '{ind}_g_draws = int(G) if (group_codes is not None and G > 0) '
    'else int(_n_draw)\n'
    '{ind}draws_g = generate_halton_normal(_g_draws, Xg.shape[2], R, seed=44)'
)


def find_module():
    """Locate the installed main_hpc.py without importing the whole package."""
    for base in (os.path.expanduser("~/.local/lib"),):
        for dirpath, _dirnames, filenames in os.walk(base):
            if "main_hpc.py" in filenames and dirpath.endswith(
                    "metacountregressor"):
                return os.path.join(dirpath, "main_hpc.py")
    return None


def main():
    rc = 0
    path = find_module()
    if path is None:
        print("FAIL: could not locate an installed metacountregressor/main_hpc.py")
        return 1
    print("target: %s" % path)

    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    if MARKER in text:
        print("already patched (marker present) -- nothing to do")
    else:
        lines = text.splitlines(keepends=True)
        hits = [i for i, ln in enumerate(lines) if TARGET_RE.match(ln.rstrip())]
        if len(hits) != 1:
            print("FAIL: expected exactly 1 vulnerable line, found %d" % len(hits))
            for i in hits:
                print("   line %d: %s" % (i + 1, lines[i].rstrip()))
            return 1
        i = hits[0]
        ind = TARGET_RE.match(lines[i].rstrip()).group("ind")
        # Refuse to patch if the names the fix depends on are not in scope.
        seg = "".join(lines[max(0, i - 400):i])
        tree = ast.parse("".join(lines))
        assigned = {n.id for node in ast.walk(tree)
                    for n in getattr(node, "targets", [])
                    if isinstance(n, ast.Name)}
        for need in ("G", "_n_draw", "Xg", "group_codes"):
            if need not in assigned and ("%s =" % need) not in seg:
                print("WARN: %r not obviously assigned before line %d; the "
                      "patch may not be valid here" % (need, i + 1))

        bak = path + ".bak_grouped_draws"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print("backup: %s" % bak)
        else:
            print("backup already exists: %s" % bak)

        # Indent any comment block directly above, keep it with the code.
        lines[i] = NEW_BODY.format(ind=ind) + "\n"
        new_text = "".join(lines)
        ast.parse(new_text)          # refuse to write unparsable output
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        print("patched line %d" % (i + 1))

    # ---- prove it, by shape, through a normal package import ------------
    # Not by exec'ing the file: main_hpc.py does
    #     from ._jax_config import configure_jax
    # with a bare `from _jax_config import ...` fallback, so loading it
    # standalone raises ImportError even when the package imports cleanly.
    # Import the package the way the models do.
    import numpy as np
    import pandas as pd
    from metacountregressor.main_hpc import build_jax_data as bjd

    n_panels, n_groups = 24, 3
    df = pd.DataFrame({
        "panel_id": np.arange(n_panels),
        "y": np.random.default_rng(0).poisson(3.0, size=n_panels).astype(float),
        "grp": [("g%d" % (i % n_groups)) for i in range(n_panels)],
        "gvar": np.random.default_rng(1).normal(size=n_panels),
    })
    data, _ = bjd(df=df, id_col="panel_id", y_col="y",
                  group_id_col="grp", grouped_cols=["gvar"], R=4)
    shape = tuple(np.shape(data.get("draws_g")))
    print("draws_g shape = %s (want %d rows, not %d)"
          % (shape, n_groups, n_panels))
    if not (len(shape) and shape[0] == n_groups):
        print("FAIL: patch did not take effect")
        rc = 1
    else:
        print("OK: group draws allocated per group")

    # Re-running must be a no-op.
    with open(path, encoding="utf-8") as fh:
        if TARGET_RE.search("".join(
                ln for ln in fh.read().splitlines(keepends=True))):
            print("FAIL: vulnerable line still present")
            rc = 1
        else:
            print("OK: no vulnerable line remains")
    print("done at %s" % time.strftime("%H:%M:%S"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
