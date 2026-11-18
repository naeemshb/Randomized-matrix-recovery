import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from data_splits import RESULTS_DIR, FIGURES_DIR

PANELS = [('a', 'mean'), ('b', 'median'), ('c', 'random')]
LABEL = {'mean': 'Mean', 'median': 'Median', 'random': 'Random Gaussian'}
COLOR = {'mean': 'tab:orange', 'median': 'tab:green', 'random': 'tab:red'}
LEGEND_FS, LABEL_FS, TICK_FS = 16, 19, 16        # sized for a sub-figure shown at one third of the text width
plt.rcParams.update({'font.family': 'serif', 'axes.linewidth': 1.4, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS})

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rls = pd.read_csv(os.path.join(RESULTS_DIR, 'imputation_vs_rls.csv'))
    base = pd.read_csv(os.path.join(RESULTS_DIR, 'imputation_baselines.csv')).set_index('imputation').MSE
    for tag, imp in PANELS:
        mean = rls[rls.imputation == imp].groupby('k').MSE.mean()
        k, y = mean.index.values.astype(float), mean.values
        fig, ax = plt.subplots(figsize=(8, 5.5))
        if len(k) > 1:
            ks = np.linspace(k[0], k[-1], 300)
            ax.plot(ks, PchipInterpolator(k, y)(ks), color='tab:blue', linewidth=2.6, zorder=3)
        ax.plot(k, y, marker='o', color='tab:blue', linestyle='none', markersize=8.5, markeredgecolor='white',
                markeredgewidth=0.9, zorder=4)
        ax.axhline(base[imp], color=COLOR[imp], linestyle='--', linewidth=2.6, zorder=2)
        ax.set_xlabel('Rank (k)', fontsize=LABEL_FS, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=LABEL_FS, fontweight='bold')
        ax.set_xticks(k.astype(int))
        handles = [plt.Line2D([0], [0], color='tab:blue', linewidth=2.6, marker='o', markersize=8.5, markeredgecolor='white',
                              markeredgewidth=0.9, label='RLS'),
                   plt.Line2D([0], [0], color=COLOR[imp], linewidth=2.6, linestyle='--', label=f'{LABEL[imp]} Imputation')]
        ax.legend(handles=handles, loc='center right', fontsize=LEGEND_FS, framealpha=0.95, edgecolor='0.8',
                  handlelength=2.4, borderpad=0.5)
        fig.tight_layout()
        name = f'fig2{tag}_{imp}_mse'
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIGURES_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved figures/{name}.png / .pdf")
