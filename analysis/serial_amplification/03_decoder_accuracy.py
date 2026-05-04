#!/usr/bin/env python3
"""
section 2.11 - analysis 03 - single-read decoder accuracy.

per-generation single-read accuracy with 95% Wilson CI error bars.
denominator: all reads that pass filter and have a predicted constant.
"""

import os
import sys
import math

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import (
    load_all_generations, N_GENERATIONS, GROUND_TRUTH,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR

MIN_READS_FLAG = 50


def _wilson_ci(k, n, z=1.96):
    """wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(center - spread, 0), min(center + spread, 1)


def _is_correct(result, ground_truth):
    """check if a decoded float matches ground truth."""
    if result.decoded_float is None or result.predicted_constant is None:
        return False
    gt = ground_truth.get(result.predicted_constant)
    if gt is None:
        return False
    if math.isnan(gt) and math.isnan(result.decoded_float):
        return True
    return result.decoded_float == gt


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=True, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))

    rows = []
    for g in gens:
        n_total = len(data[g].decoded_reads)

        # all reads that passed filter with a predicted constant
        passed = [
            d for d in data[g].decoded_reads
            if d.result.passed_filter
            and d.result.predicted_constant is not None
        ]
        n_passed = len(passed)

        n_decoded = sum(1 for d in passed
                        if d.result.decoded_float is not None)

        correct = [d for d in passed if _is_correct(d.result, GROUND_TRUTH)]
        n_correct = len(correct)

        acc, ci_lo, ci_hi = _wilson_ci(n_correct, n_passed)

        rows.append({
            'gen': g,
            'n_total': n_total,
            'n_passed': n_passed,
            'n_decoded': n_decoded,
            'n_correct': n_correct,
            'accuracy': acc,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'low_n': n_passed < MIN_READS_FLAG,
        })

    # accuracy plot
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))

    # separate low-n and normal points
    normal = [r for r in rows if not r['low_n']]
    flagged = [r for r in rows if r['low_n']]

    if normal:
        ax.errorbar([r['gen'] for r in normal],
                    [r['accuracy'] for r in normal],
                    yerr=[[r['accuracy'] - r['ci_lo'] for r in normal],
                          [r['ci_hi'] - r['accuracy'] for r in normal]],
                    fmt='o-', color=WONG[0], lw=2, markersize=6, capsize=3,
                    label='Accuracy (95% CI)')
    if flagged:
        ax.errorbar([r['gen'] for r in flagged],
                    [r['accuracy'] for r in flagged],
                    yerr=[[r['accuracy'] - r['ci_lo'] for r in flagged],
                          [r['ci_hi'] - r['accuracy'] for r in flagged]],
                    fmt='o', color=WONG[1], markersize=6, capsize=3,
                    markerfacecolor='none', markeredgewidth=1.5,
                    label=f'< {MIN_READS_FLAG} target reads')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Single-read decoding accuracy')
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'decoder_accuracy.pdf'))

    # read breakdown plot (raw counts)
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    n_passed_arr = [r['n_passed'] for r in rows]
    n_decoded_arr = [r['n_decoded'] for r in rows]
    n_correct_arr = [r['n_correct'] for r in rows]
    width = 0.25
    x = np.arange(len(gens))

    ax.bar(x - width, n_passed_arr, width, color=WONG[0], alpha=0.8,
           label='Passed filter')
    ax.bar(x, n_decoded_arr, width, color=WONG[2], alpha=0.8,
           label='Decoded')
    ax.bar(x + width, n_correct_arr, width, color=WONG[5], alpha=0.8,
           label='Correct')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Read count')
    ax.set_xticks(x)
    ax.set_xticklabels(gens)
    ax.legend(frameon=False, fontsize=7)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'read_breakdown.pdf'))

    # read breakdown plot (fractions of total reads per generation)
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    n_total_arr = [r['n_total'] for r in rows]
    frac_passed = [p / n if n > 0 else 0
                   for p, n in zip(n_passed_arr, n_total_arr)]
    frac_decoded = [d / n if n > 0 else 0
                    for d, n in zip(n_decoded_arr, n_total_arr)]
    frac_correct = [c / n if n > 0 else 0
                    for c, n in zip(n_correct_arr, n_total_arr)]

    ax.bar(x - width, frac_passed, width, color=WONG[0], alpha=0.8,
           label='Passed filter')
    ax.bar(x, frac_decoded, width, color=WONG[2], alpha=0.8,
           label='Decoded')
    ax.bar(x + width, frac_correct, width, color=WONG[5], alpha=0.8,
           label='Correct')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fraction of total reads')
    ax.set_xticks(x)
    ax.set_xticklabels(gens)
    ax.legend(frameon=False, fontsize=7)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'read_breakdown_fraction.pdf'))

    # summary tsv
    tsv_path = os.path.join(FIGURE_DIR, '03_decoder_accuracy.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\tn_total\tn_passed\tn_decoded\tn_correct'
                '\taccuracy\tci_lo\tci_hi\tlow_n_flag\n')
        for r in rows:
            f.write(f"{r['gen']}\t{r['n_total']}\t{r['n_passed']}"
                    f"\t{r['n_decoded']}\t{r['n_correct']}"
                    f"\t{r['accuracy']:.4f}\t{r['ci_lo']:.4f}"
                    f"\t{r['ci_hi']:.4f}\t{r['low_n']}\n")
    print("    saved 03_decoder_accuracy.tsv", flush=True)

    return {'rows': rows}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='analysis 03 - decoder accuracy')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 03 - decoder accuracy", flush=True)
    data = load_all_generations(decode=True, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
