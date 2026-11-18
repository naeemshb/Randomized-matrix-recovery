import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import output_dirs

RES, FIG = output_dirs('scalability_noiseless', '--quick' in sys.argv)
STYLE = {'RLS': dict(color='tab:blue', marker='o'), 'ALS': dict(color='tab:green', marker='^')}
LINE = dict(linewidth=2.0, markersize=7, markeredgecolor='white', markeredgewidth=0.8, zorder=3)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.edgecolor': '0.8'})

if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    S = pd.read_csv(os.path.join(RES, 'summary.csv'))
    SP = pd.read_csv(os.path.join(RES, 'speedup.csv')).sort_values('entries')
    slope = pd.read_csv(os.path.join(RES, 'fit.csv')).set_index('method').slope
    sizes = S.drop_duplicates(['m', 'n']).sort_values('entries')

    def new_axis():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_xscale('log')
        ax.set_xticks(sizes.entries.values)
        ax.set_xticklabels([f"{m}$\\times${n}" for m, n in zip(sizes.m, sizes.n)], rotation=20)
        ax.minorticks_off()
        ax.set_xlabel('Matrix size (m × n, log scale)', fontweight='bold')
        return fig, ax

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {name}.png / .pdf")

    fig, ax = new_axis()
    for method in ['RLS', 'ALS']:
        d = S[S.method == method].sort_values('entries')
        ax.errorbar(d.entries, d.time_mean_ms, yerr=d.time_ci95_ms.fillna(0), capsize=4, label=f"{method} (slope {slope[method]:.2f})", **LINE, **STYLE[method])
    ax.set_yscale('log')
    ax.set_ylabel('Wall-clock runtime (ms, log scale)', fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    save(fig, 'runtime')

    fig, ax = new_axis()
    ax.plot(SP.entries, SP.speedup_ALS_over_RLS, color='tab:red', marker='s', **LINE)
    for x, y in zip(SP.entries, SP.speedup_ALS_over_RLS):
        ax.annotate(f'{y:.1f}×', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10)
    ax.set_ylim(0, SP.speedup_ALS_over_RLS.max() * 1.2)
    ax.set_ylabel('Speedup: ALS runtime / RLS runtime', fontweight='bold')
    save(fig, 'speedup')

    fig, ax = new_axis()
    for method in ['RLS', 'ALS']:
        d = S[S.method == method].sort_values('entries')
        ax.errorbar(d.entries, d.MSE_mean, yerr=d.MSE_ci95.fillna(0), capsize=4, label=method, **LINE, **STYLE[method])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Test MSE', fontweight='bold')
    ax.legend(fontsize=10)
    save(fig, 'mse')
