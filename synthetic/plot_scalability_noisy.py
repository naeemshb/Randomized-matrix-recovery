import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import output_dirs

RES, FIG = output_dirs('scalability_noisy', '--quick' in sys.argv)
STYLE = {'RLS': dict(color='tab:blue', marker='o'), 'ALS': dict(color='tab:green', marker='^')}
LINE = dict(linewidth=2.0, markersize=7, markeredgecolor='white', markeredgewidth=0.8, zorder=3)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.labelsize': 10, 'ytick.labelsize': 11, 'legend.edgecolor': '0.8'})

if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    S = pd.read_csv(os.path.join(RES, 'summary.csv'))
    C = pd.read_csv(os.path.join(RES, 'comparison.csv'))
    noise = sorted(S.noise_sigma.unique())
    sizes = S.drop_duplicates(['m', 'n']).sort_values('entries')

    def size_axis(ax):
        ax.set_xscale('log')
        ax.set_xticks(sizes.entries.values)
        ax.set_xticklabels([f"{m}$\\times${n}" for m, n in zip(sizes.m, sizes.n)], rotation=25)
        ax.minorticks_off()
        ax.set_xlabel('Matrix size (m × n, log scale)', fontweight='bold')

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {name}.png / .pdf")

    def panels(mean_col, ci_col, ylabel, name, logy):
        fig, axes = plt.subplots(1, len(noise), figsize=(5.6 * len(noise), 4.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, sigma in zip(axes, noise):
            for method in ['RLS', 'ALS']:
                d = S[np.isclose(S.noise_sigma, sigma) & (S.method == method)].sort_values('entries')
                ax.errorbar(d.entries, d[mean_col], yerr=d[ci_col].fillna(0), capsize=4, label=method, **LINE, **STYLE[method])
            size_axis(ax)
            ax.set_title(f'σ = {sigma:.2f}', fontsize=12)
            if logy:
                ax.set_yscale('log')
        axes[0].set_ylabel(ylabel + (' (log scale)' if logy else ''), fontweight='bold')
        axes[0].legend(fontsize=10)
        save(fig, name)

    panels('time_mean_ms', 'time_ci95_ms', 'Wall-clock runtime (ms)', 'runtime', logy=True)
    panels('MSE_mean', 'MSE_ci95', 'Test MSE', 'mse', logy=S.MSE_mean.max() / max(S.MSE_mean.min(), 1e-12) > 100)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sigma, color in zip(noise, ['tab:purple', 'tab:red', 'tab:brown']):
        d = C[np.isclose(C.noise_sigma, sigma)].sort_values('entries')
        ax.plot(d.entries, d.speedup_ALS_over_RLS, color=color, marker='s', label=f'σ = {sigma:.2f}', **LINE)
    size_axis(ax)
    ax.set_ylim(0, C.speedup_ALS_over_RLS.max() * 1.15)
    ax.set_ylabel('Speedup: ALS runtime / RLS runtime', fontweight='bold')
    ax.legend(fontsize=10)
    save(fig, 'speedup')

    big = C[C.entries == C.entries.max()].sort_values('noise_sigma')
    x, w = np.arange(len(big)), 0.28
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - w, big.RLS_MSE, w, color='tab:blue', label='RLS MSE')
    ax.bar(x, big.ALS_MSE, w, color='tab:green', label='ALS MSE')
    ax.bar(x + w, big.MSE_difference, w, color='tab:gray', label='RLS − ALS')
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ = {s:.2f}' for s in big.noise_sigma])
    ax.set_ylabel(f'Test MSE at {int(big.m.iloc[0])}×{int(big.n.iloc[0])}', fontweight='bold')
    ax.legend(fontsize=10)
    save(fig, 'gap')
