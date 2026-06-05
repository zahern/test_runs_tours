"""
Regenerate GA stage plots from saved ga_results.pkl using the updated
plot_convergence style (no diversity/mutation rate overlay).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

# Load saved GA results
with open('saved_data/ga_results.pkl', 'rb') as f:
    ga_results = pickle.load(f)

out_dir = 'runs/ga_staged__20538563.aqua/plots'
os.makedirs(out_dir, exist_ok=True)

# Regenerate plots for stages 1-5 (and any per-segment subplots)
for stage in [1, 2, 3, 4, 5]:
    res = ga_results.results.get(stage)
    if res is None:
        continue

    seg_results = getattr(res, 'metadata', {}).get('segment_results', {})
    if seg_results:
        n = len(seg_results)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(13, 4.5 * rows),
                                 facecolor='white', squeeze=False)
        axes = axes.flatten()
        for i, (seg_label, seg_res) in enumerate(sorted(seg_results.items())):
            if i < len(axes):
                seg_res.plot_convergence(ax=axes[i], segment_label=seg_label)
        for j in range(len(seg_results), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f'GA Variable Selection — Stage {stage}', fontsize=13,
                     fontweight='bold', y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
        res.plot_convergence(ax=ax)
        fig.suptitle(f'GA Variable Selection — Stage {stage}', fontsize=13,
                     fontweight='bold', y=1.01)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'ga_stage{stage}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")

print("Done regenerating plots.")
