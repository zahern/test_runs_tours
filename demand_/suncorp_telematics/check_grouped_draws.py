"""
Behavioural gate for the grouped-draws allocation fix (commit ca34749).

WHY BEHAVIOURAL, NOT A VERSION FLOOR. The T4 job gated on
`version >= 1.0.170`, but that number could not be verified as published from
the machine that writes the jobs, and the local interpreter still had
1.0.164 installed -- so the gate was asserting a fact about a package index
that nothing in the job actually tested. A version floor is also wrong in
principle: it cannot distinguish two builds with the same version string but
different code.

The fixed property is directly observable: build_jax_data is handed more
panels than groups (N=24 panels, G=3 groups), so a correct implementation
allocates draws_g with G rows. The unfixed one allocated N rows. Same call,
same arguments, different tensor shape -- that is the real contract.

Run with no arguments to gate the installed package; pass a path to gate a
source tree.
"""
import sys
import traceback

FAIL = []


def chk(c, label, d=""):
    print("  %-5s %-50s %s"
          % ("[ok]" if c else "[FAIL]", label, d), flush=True)
    if not c:
        FAIL.append(label)


def probe(build_jax_data, n_panels=24, n_groups=3, R=4):
    """Build a panel with N > G and return the shape of the group draws."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "panel_id": np.arange(n_panels),
        "y": rng.poisson(3.0, size=n_panels).astype(float),
        "grp": [("g%d" % (i % n_groups)) for i in range(n_panels)],
        "gvar": rng.normal(size=n_panels),
    })
    data, _spec = build_jax_data(
        df=df, id_col="panel_id", y_col="y",
        group_id_col="grp", grouped_cols=["gvar"], R=R,
    )
    g = data.get("draws_g")
    return None if g is None else tuple(np.shape(g)), n_panels, n_groups


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    print("=" * 74)
    print("grouped-draws behavioural gate%s" % (" (%s)" % src if src else ""))
    print("=" * 74)
    if src:
        sys.path.insert(0, src)
    from metacountregressor.main_hpc import build_jax_data  # noqa: E402
    from importlib.metadata import version  # noqa: E402
    try:
        print("  installed metacountregressor: %s" % version(
            "metacountregressor"))
    except Exception:  # noqa: BLE001
        pass
    import metacountregressor.main_hpc as _m
    print("  module: %s" % _m.__file__)

    try:
        shape, n_panels, n_groups = probe(build_jax_data)
    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        chk(False, "build_jax_data probe ran", "raised")
        return 1

    print("  draws_g shape = %s, expected (%d, ...) with N=%d panels, G=%d "
          "groups" % (shape, n_groups, n_panels, n_groups))
    chk(shape is not None and len(shape) > 0 and shape[0] == n_groups,
        "group draws are allocated per GROUP, not per panel",
        "got %s" % (shape,))
    chk(shape is None or shape[0] != n_panels,
        "not the oversized per-panel allocation (%d rows)" % n_panels,
        "got %s" % (shape,))

    print("=" * 74)
    print("gate: %s" % ("PASS" if not FAIL else "FAIL %s" % FAIL))
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
