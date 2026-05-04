#!/usr/bin/env python3
"""
per-bit decision breakdown -> the hmm decoder against the edit-distance baseline,
across the 12 serial-passaging generations.

for every generation we already counted each per-read bit call as correct,
flipped or erased, so this just reads that table and plots three rates. nothing
clever here, it is a diagnostic showing the two decoders fail in different ways,
the baseline trades flips for erasures and falls off a cliff once the chimeras
pile up while the hmm barely moves.

input is per_bit_accuracy.tsv with one row per (barcode, decoder) and columns
correct_rate, flip_rate, erasure_rate (fractions, multiplied by 100 here). the
serial generations are barcode07 -> barcode18. writes per_bit_breakdown.png next
to this file. the baseline producer lives elsewhere, so the tsv has to be on
disk already, this script will not regenerate it.
"""

import os
import sys
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import plt, PHI, WONG, remove_spines, save_fig

TSV = os.path.join(SCRIPT_DIR, 'per_bit_accuracy.tsv')
PNG = os.path.join(SCRIPT_DIR, 'per_bit_breakdown.png')
GENS = list(range(1, 13))

# blue is the hmm, vermillion is the baseline, same as the other panels.
HMM, BASE = WONG[0], WONG[1]


def load():
    # barcode07 is gen 1 and barcode18 is gen 12. skip anything else in the tsv.
    gen_for_bc = {f'barcode{6 + g:02d}': g for g in GENS}
    data = {}
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            g = gen_for_bc.get(r['barcode'])
            if g is None:
                continue
            data[(r['decoder'], g)] = {
                'erasure': float(r['erasure_rate']) * 100,
                'flip': float(r['flip_rate']) * 100,
                'correct': float(r['correct_rate']) * 100,
            }
    return data


def main():
    if not os.path.exists(TSV):
        # the baseline pipeline writes this, not me. bail loudly.
        print(f"missing {TSV}, run the baseline per-bit accuracy step first")
        sys.exit(1)
    data = load()

    fig_w = 7.5
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w / PHI / 3 * 1.05),
                             gridspec_kw={'wspace': 0.32})

    # left to right, erasure then flip then correct. correct gets a tighter
    # y-window because nobody is interested in the bottom half.
    panels = [
        ('erasure', 'Erasure rate [%]', None),
        ('flip', 'Bit-flip rate [%]', None),
        ('correct', 'Correct rate [%]', (45, 100)),
    ]
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        ax.plot(GENS, [data[('hmm', g)][key] for g in GENS], '-o',
                color=HMM, markersize=4, label='HMM')
        ax.plot(GENS, [data[('baseline', g)][key] for g in GENS], '-s',
                color=BASE, markersize=4, label='Baseline')
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_xticks(GENS)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, frameon=False)
        remove_spines(ax)

    save_fig(fig, PNG)


if __name__ == '__main__':
    main()
