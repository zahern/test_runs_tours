"""
Reproduce the 26095208 figure failure in the CLUSTER environment.

Matplotlib 3.11 (cluster) tightened set_yticklabels: passing a label count
that disagrees with the installed FixedLocator now raises ValueError where
3.10 only warned. Stage 59 called

    ax.set_yticklabels(_rowlab, fontsize=5.5, rotation=0)

with 20 labels; seaborn's heatmap had left 10 tick locations. On the cluster
that raised, the enclosing try/except caught it, and the surfaces figure, the
coefs figure and game_day_pred_vs_actual.png were ALL skipped while the job
still exited 0.

Run on the cluster to confirm the trigger and the fix in the environment that
actually failed.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FAIL = []


def chk(c, label, d=""):
    print("  %-4s %-52s %s" % ("ok" if c else "FAIL", label, d))
    if not c:
        FAIL.append(label)


print("matplotlib %s | seaborn %s"
      % (matplotlib.__version__, sns.__version__))

rng = np.random.default_rng(0)
osmids = list(range(1000, 1020))
rowlab = ["Road %d  %d" % (i, o) for i, o in enumerate(osmids)]
pivot = pd.DataFrame(
    rng.random((20, 6)) * 5,
    index=osmids, columns=range(6)).astype(float)

print()
print("=" * 74)
print("1. buggy pattern: set_yticklabels with no matching ticks")
print("=" * 74)
fig, ax = plt.subplots()
sns.heatmap(pivot, ax=ax, cmap="magma", cbar_kws={"label": "speeding"})
raised = None
try:
    ax.set_yticklabels(rowlab, fontsize=5.5, rotation=0)
except Exception as exc:  # noqa: BLE001
    raised = exc
print("   raised: %s" % (type(raised).__name__ if raised else "None"))
if raised:
    print("   msg   : %s" % str(raised)[:100])
chk(raised is not None,
    "this environment DOES raise on the unguarded call",
    "matplotlib %s" % matplotlib.__version__)
plt.close(fig)

print()
print("=" * 74)
print("2. fixed pattern: ticks and labels set together")
print("=" * 74)
fig, ax = plt.subplots()
sns.heatmap(pivot, ax=ax, cmap="magma", cbar_kws={"label": "speeding"})
err = None
try:
    if len(rowlab):
        ax.set_yticks(np.arange(len(rowlab)) + 0.5)
        ax.set_yticklabels(rowlab, fontsize=5.5, rotation=0)
except Exception as exc:  # noqa: BLE001
    err = exc
chk(err is None, "fixed call does not raise", str(err)[:70] if err else "")
got = [t.get_text() for t in ax.get_yticklabels()]
chk(len([g for g in got if g]) == len(rowlab),
    "all 20 row labels present", "%d non-empty" % len([g for g in got if g]))
worst = max(abs(t.get_position()[1] - (i + 0.5))
            for i, t in enumerate(ax.get_yticklabels())[:len(rowlab)])
chk(worst < 1e-6, "label i centred on row i", "max offset %.2g" % worst)
fig.savefig("/tmp/tickfix_check.png", dpi=90)
plt.close(fig)
chk(True, "figure saved without error", "/tmp/tickfix_check.png")

print()
print("=" * 74)
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL %s" % FAIL))
print("=" * 74)
sys.exit(1 if FAIL else 0)
