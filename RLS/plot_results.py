import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_splits import RESULTS_DIR, FIGURES_DIR

STYLE = {'RLS': ('#e41a1c', 'o'), 'ALS': ('#377eb8', 's'), 'NuNo': ('#ff7f00', '^'), 'NuNoR': ('#4daf4a', 'D'), 'CSMC': ('#984ea3', 'v')}
METRICS = [('MSE', 'MSE', 'mse'), ('Error', 'Error Rate', 'error'), ('Time', 'Time (ms)', 'time')]
plt.rcParams.update({'font.size': 13, 'font.family': 'serif', 'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.edgecolor': '0.8'})

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'results.csv'))
    k_values = sorted(df.k.unique())
    for key, ylabel, short in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method, (color, marker) in STYLE.items():
            d = df[df.Method == method].sort_values('k')
            ax.plot(d.k, d[f'{key}_mean'], label=method, color=color, marker=marker, linewidth=2.2, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=3)
            ax.fill_between(d.k, d[f'{key}_mean'] - d[f'{key}_std'], d[f'{key}_mean'] + d[f'{key}_std'], alpha=0.15, color=color, zorder=1)
        ax.set_xticks(k_values)
        ax.set_xlabel('Rank (k)', fontweight='bold')
        ax.set_ylabel(ylabel + (' (log scale)' if key == 'Time' else ''), fontweight='bold')
        if key == 'Time':
            ax.set_yscale('log')
        ax.legend(fontsize=10, framealpha=0.95, ncol=5 if key == 'Time' else 1, loc='upper center' if key == 'Time' else 'best',
                  bbox_to_anchor=(0.5, 1.12) if key == 'Time' else None)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIGURES_DIR, f'{short}.{ext}'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved figures/{short}.png / .pdf")
