#!/usr/bin/env python3
"""
section 2.11 - analysis 08 - consensus convergence through chimeric noise.

for each generation (1-12), bootstrap-subsample from ALL reads (including
chimeras and off-target) at increasing total read counts. the decoder sifts
qualifying reads from the sample, groups by predicted constant, and does
confidence-weighted majority vote. y-axis is fraction of all 17 constants
whose consensus matches ground truth. x-axis is total reads sampled.

this shows that the decoder recovers all 17 constants even when most reads
are chimeric noise, with later generations needing more total reads due to
lower on-target fraction.

the bootstrap TSV is pre-computed on AWS (see babbage_extract_barcode.py).
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import remove_spines, save_fig, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR
N_GENERATIONS = 12
N_CONSTANTS = 17
TSV_PATH = os.path.join(SCRIPT_DIR, '08_consensus_convergence.tsv')


def run_analysis():
    # load bootstrap results
    data = np.genfromtxt(TSV_PATH, delimiter='\t', skip_header=1,
                         dtype=[('gen', int), ('depth', int), ('rep', int),
                                ('n_correct', int), ('frac', float)])

    gens = list(range(1, N_GENERATIONS + 1))

    # get unique depths from data
    all_depths = sorted(set(data['depth']))

    # build lookup: (gen, depth) -> array of fractions across replicates
    # map depths to evenly-spaced indices for geometric spacing
    depth_to_idx = {d: i for i, d in enumerate(all_depths)}
    curves = {}
    for g in gens:
        for d in all_depths:
            mask = (data['gen'] == g) & (data['depth'] == d)
            vals = data['frac'][mask]
            if len(vals) > 0:
                curves[(g, d)] = vals

    # plot - one labeled line per generation
    # cycle WONG colours with solid (gen 1-6) / dashed (gen 7-12)
    fig_w = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / PHI))

    styles = [('-', 'o'), ('--', 's')]  # solid+circle, dashed+square
    for g in gens:
        color = WONG[(g - 1) % len(WONG)]
        ls, marker = styles[0 if g <= 6 else 1]
        medians = []
        lo = []
        hi = []
        x_pos = []

        for d in all_depths:
            if (g, d) not in curves:
                continue
            vals = curves[(g, d)]
            x_pos.append(depth_to_idx[d])
            medians.append(np.median(vals))
            lo.append(np.percentile(vals, 5))
            hi.append(np.percentile(vals, 95))

        if x_pos:
            ax.plot(x_pos, medians, ls, color=color, marker=marker,
                    markersize=3, lw=1.2, label=f'Gen {g}')
            ax.fill_between(x_pos, lo, hi, color=color, alpha=0.10)

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Total reads sampled')
    ax.set_ylabel('Fraction of 17 constants recovered')
    ax.set_ylim(-0.02, 1.08)
    # label a subset of depths to avoid crowding
    show = {100, 500, 2000, 10000, 50000, 200000}
    ax.set_xticks(range(len(all_depths)))
    ax.set_xticklabels([f'{d:,}' if d in show else ''
                        for d in all_depths], fontsize=7)
    remove_spines(ax)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc='lower right')

    pdf_path = os.path.join(FIGURE_DIR, 'consensus_convergence.pdf')
    save_fig(fig, pdf_path)

    return pdf_path


def main():
    print("section 2.11 - analysis 08 - consensus convergence (plot)", flush=True)
    pdf_path = run_analysis()
    print(f"done. {pdf_path}", flush=True)


if __name__ == '__main__':
    main()
