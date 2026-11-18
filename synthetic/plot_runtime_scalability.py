import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import output_dirs

RES, FIG = output_dirs('runtime_scalability', '--quick' in sys.argv)
STYLE = {'RLS': dict(color='tab:blue', marker='o'), 'CSMC': dict(color='tab:orange', marker='s'), 'ALS': dict(color='tab:green', marker='^')}
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.labelsize': 10, 'ytick.labelsize': 11, 'legend.edgecolor': '0.8'})

if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    S = pd.read_csv(os.path.join(RES, 'summary.csv'))
    slope = pd.read_csv(os.path.join(RES, 'fit.csv')).set_index('method').slope
    sizes = S.drop_duplicates(['m', 'n']).sort_values('entries')
    iters = S.groupby('method').iterations.first().round().astype(int)
    label = {'RLS': 'RLS (one-shot)', 'CSMC': f"CSMC ({iters['CSMC']} internal it.)", 'ALS': f"ALS ({iters['ALS']} sweeps)"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in ['RLS', 'CSMC', 'ALS']:
        d = S[S.method == method].sort_values('entries')
        ax.errorbar(d.entries, d.time_mean_ms, yerr=d.time_ci95_ms.fillna(0), linewidth=2.0, markersize=7, capsize=4, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=3, label=f"{label[method]}, slope {slope[method]:.2f}", **STYLE[method])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(sizes.entries.values)
    ax.set_xticklabels([f"{m}$\\times${n}" for m, n in zip(sizes.m, sizes.n)], rotation=25)
    ax.minorticks_off()
    ax.set_xlabel('Matrix size (m × n, log scale)', fontweight='bold')
    ax.set_ylabel('Wall-clock runtime (ms, log scale)', fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'runtime_scalability.{ext}'), dpi=300, bbox_inches='tight')
    print("saved runtime_scalability.png / .pdf")
