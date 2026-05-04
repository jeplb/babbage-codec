#!/usr/bin/env python3
"""
section 2.11 - analysis 01 - read inventory.

per-generation read count with Qubit concentration overlay.
shows divergence between total DNA mass and sequenceable target.
"""

import os
import sys

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import load_all_generations, QUBIT_NG_UL, N_GENERATIONS
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=False, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))
    read_counts = [len(data[g].raw_reads) for g in gens]
    qubit_concs = [QUBIT_NG_UL[g] for g in gens]

    fig, ax1 = plt.subplots(figsize=(7, 7 / PHI))

    ax1.bar(gens, read_counts, color=WONG[0], alpha=0.8, zorder=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Total read count')
    ax1.set_xticks(gens)
    remove_spines(ax1)

    ax2 = ax1.twinx()
    ax2.plot(gens, qubit_concs, 'o-', color=WONG[1], lw=2, markersize=6,
             zorder=3)
    ax2.set_ylabel('Qubit concentration [ng/$\\mu$L]')
    ax2.spines['top'].set_visible(False)

    # combined legend
    handles = [
        Patch(facecolor=WONG[0], alpha=0.8, label='Read count'),
        Line2D([0], [0], color=WONG[1], marker='o', lw=2, label='Qubit concentration'),
    ]
    ax1.legend(handles=handles, frameon=False, loc='upper left')

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'read_inventory.pdf'))

    # summary tsv
    tsv_path = os.path.join(FIGURE_DIR, '01_read_inventory.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\tbarcode\tread_count\tqubit_ng_ul\n')
        for g in gens:
            f.write(f"{g}\t{data[g].barcode}\t{len(data[g].raw_reads)}"
                    f"\t{QUBIT_NG_UL[g]}\n")
    print("    saved 01_read_inventory.tsv", flush=True)

    return {'read_counts': read_counts, 'qubit': qubit_concs}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='analysis 01 - read inventory')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 01 - read inventory", flush=True)
    data = load_all_generations(decode=False, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
