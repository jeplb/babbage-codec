#!/usr/bin/env python3
"""
section 2.11 - error rates figures from pre-computed TSV.

figure 1: stacked bars showing match/substitution/deletion fractions vs generation.
figure 2: combined log-scale line plot - sub/del on left axis, insertion on right axis.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import remove_spines, save_fig, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR


def main():
    tsv_path = os.path.join(SCRIPT_DIR, '04_error_modes.tsv')
    gens, sub_rates, ins_rates, del_rates = [], [], [], []
    with open(tsv_path) as f:
        f.readline()  # header
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.strip().split('\t')
            gens.append(int(cols[0]))
            sub_rates.append(float(cols[1]))
            ins_rates.append(float(cols[2]))
            del_rates.append(float(cols[3]))

    x = np.array(gens)
    width = 0.7
    match_rates = [1.0 - s - d for s, d in zip(sub_rates, del_rates)]

    # two-panel stacked: insertion rate (top), match/sub/del fractions (bottom)
    fig, (ax_ins, ax_frac) = plt.subplots(
        2, 1, figsize=(7, 7 / PHI * 1.4),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True,
    )

    # top: insertion rate line plot
    ax_ins.plot(x, ins_rates, 's-', color=WONG[1], lw=2, markersize=5)
    ax_ins.set_ylabel('Insertion rate\n[runs per ref position]')
    ax_ins.set_ylim(0, 3.2)
    remove_spines(ax_ins)

    # bottom: stacked bar (match + sub + del = 1.0)
    ax_frac.bar(x, match_rates, width, color=WONG[5], alpha=0.7, label='Match')
    ax_frac.bar(x, sub_rates, width, bottom=match_rates, color=WONG[0],
                alpha=0.7, label='Substitution')
    bottom_del = [m + s for m, s in zip(match_rates, sub_rates)]
    ax_frac.bar(x, del_rates, width, bottom=bottom_del, color=WONG[2],
                alpha=0.7, label='Deletion')
    ax_frac.set_xlabel('Generation')
    ax_frac.set_ylabel('Fraction of ref positions')
    ax_frac.set_xticks(x)
    ax_frac.set_ylim(0, 1.05)
    ax_frac.legend(frameon=False, fontsize=8)
    remove_spines(ax_frac)

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'error_rates_stacked.pdf'))
    print("saved error_rates_stacked.pdf")


if __name__ == '__main__':
    main()
