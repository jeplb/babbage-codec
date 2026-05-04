#!/usr/bin/env python3
"""
section 4.03 - effective throughput vs read depth.

uses pre-computed bootstrap convergence data from section 2.11 to compute
bits recovered per sequencing read as a function of total reads sampled.
k is total reads sampled from the barcode (including chimeras, off-target,
and failed reads) - an end-to-end efficiency metric where the denominator
reflects the true sequencing cost.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ANALYSIS_DIR)

from shared import remove_spines, save_fig, PHI, WONG, plt, np

FIGURE_DIR = SCRIPT_DIR

N_CONSTANTS = 17
BITS_PER_CONSTANT = 32

# consensus convergence TSV from serial_amplification analysis
TSV_PATH = os.path.join(
    ANALYSIS_DIR, 'serial_amplification', '08_consensus_convergence.tsv',
)


def main():
    print("section 4.03: effective throughput vs read depth")

    # load bootstrap results
    data = np.genfromtxt(TSV_PATH, delimiter='\t', skip_header=1,
                         dtype=[('gen', int), ('depth', int), ('rep', int),
                                ('n_correct', int), ('frac', float)])

    gens = sorted(set(data['gen']))
    all_depths = sorted(set(data['depth']))

    # for each (gen, depth), compute median P(k) = fraction of 17 correct
    # effective throughput = 32 * 17 * P(k) / k  bits per read
    throughput = {}  # (gen, depth) -> median bits/read
    pk_median = {}

    for g in gens:
        for d in all_depths:
            mask = (data['gen'] == g) & (data['depth'] == d)
            fracs = data['frac'][mask]
            if len(fracs) == 0:
                continue
            med_pk = np.median(fracs)
            pk_median[(g, d)] = med_pk
            throughput[(g, d)] = BITS_PER_CONSTANT * N_CONSTANTS * med_pk / d

    # plot: throughput vs depth, colored by generation
    fig_w = 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / PHI))

    # colormap for 12 generations
    cmap = plt.cm.viridis
    gen_colors = {g: cmap(i / (len(gens) - 1)) for i, g in enumerate(gens)}

    for g in gens:
        depths_g = sorted([d for (gg, d) in throughput if gg == g])
        if not depths_g:
            continue
        tp = [throughput[(g, d)] for d in depths_g]
        ax.plot(depths_g, tp, '-o', color=gen_colors[g], markersize=3,
                label=f'Gen {g}', lw=1.2)

        # mark optimal depth
        best_idx = np.argmax(tp)
        ax.plot(depths_g[best_idx], tp[best_idx], 'v', color=gen_colors[g],
                markersize=6, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Total reads sampled (k)')
    ax.set_ylabel('Effective throughput [bits/read]')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    remove_spines(ax)
    fig.tight_layout()

    save_fig(fig, os.path.join(FIGURE_DIR, 'throughput.pdf'))

    # save TSV
    tsv_path = os.path.join(FIGURE_DIR, '03_throughput.tsv')
    with open(tsv_path, 'w') as f:
        f.write("generation\tdepth\tmedian_pk\tthroughput_bits_per_read\n")
        for g in gens:
            depths_g = sorted([d for (gg, d) in throughput if gg == g])
            for d in depths_g:
                f.write(f"{g}\t{d}\t{pk_median[(g, d)]:.6f}\t{throughput[(g, d)]:.6f}\n")
    print(f"    saved 03_throughput.tsv")

    print("  done.")


if __name__ == '__main__':
    main()
