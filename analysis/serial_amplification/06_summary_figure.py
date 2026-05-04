#!/usr/bin/env python3
"""
section 2.11 - analysis 06 - summary figure.

three-panel publication figure:
  panel A: Qubit concentration and target-length read fraction vs generation
  panel B: decoder accuracy vs generation (target-length, 95% CI)
  panel C: read length distributions for gen 1, 4, 8, 12 overlaid
"""

import os
import sys
import math

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import (
    load_all_generations, N_GENERATIONS, TARGET_MIN, TARGET_MAX,
    EXPECTED_LENGTH, QUBIT_NG_UL, GROUND_TRUTH,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR

SHOWCASE_GENS = [1, 4, 8, 12]


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(center - spread, 0), min(center + spread, 1)


def _is_correct(result):
    if result.decoded_float is None or result.predicted_constant is None:
        return False
    gt = GROUND_TRUTH.get(result.predicted_constant)
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

    # compute metrics
    target_fracs = []
    qubit = [QUBIT_NG_UL[g] for g in gens]
    accuracies = []
    ci_los = []
    ci_his = []

    for g in gens:
        # target fraction
        lengths = [r.length for r in data[g].raw_reads]
        n_total = len(lengths)
        n_target = sum(1 for l in lengths if TARGET_MIN <= l <= TARGET_MAX)
        target_fracs.append(n_target / n_total if n_total > 0 else 0.0)

        # accuracy on target reads (need decoded data)
        if data[g].decoded_reads:
            target_decoded = [
                d for d in data[g].decoded_reads
                if TARGET_MIN <= d.raw.length <= TARGET_MAX
                and d.result.passed_filter
                and d.result.predicted_constant is not None
                and d.result.decoded_float is not None
            ]
            n_decoded = len(target_decoded)
            n_correct = sum(1 for d in target_decoded if _is_correct(d.result))
            acc, lo, hi = _wilson_ci(n_correct, n_decoded)
        else:
            acc, lo, hi = 0.0, 0.0, 0.0
        accuracies.append(acc)
        ci_los.append(lo)
        ci_his.append(hi)

    # create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 12 / PHI / 2))

    # panel A: Qubit + target fraction
    ax = axes[0]
    ax.set_title('A', fontsize=11, fontweight='bold', loc='left')
    color_qubit = WONG[1]
    color_frac = WONG[0]

    ax.bar(gens, target_fracs, color=color_frac, alpha=0.7, zorder=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Target read fraction (200-300 bp)', color=color_frac)
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelcolor=color_frac)
    remove_spines(ax)

    ax2 = ax.twinx()
    ax2.plot(gens, qubit, 'o-', color=color_qubit, lw=2, markersize=4,
             zorder=3)
    ax2.set_ylabel('Qubit [ng/$\\mu$L]', color=color_qubit)
    ax2.tick_params(axis='y', labelcolor=color_qubit)
    ax2.spines['top'].set_visible(False)

    # panel B: accuracy
    ax = axes[1]
    ax.set_title('B', fontsize=11, fontweight='bold', loc='left')
    err_lo = [a - l for a, l in zip(accuracies, ci_los)]
    err_hi = [h - a for a, h in zip(accuracies, ci_his)]
    ax.errorbar(gens, accuracies, yerr=[err_lo, err_hi],
                fmt='o-', color=WONG[0], lw=2, markersize=5, capsize=3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Single-read decoding accuracy')
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    remove_spines(ax)

    # panel C: length distributions for showcase generations
    ax = axes[2]
    ax.set_title('C', fontsize=11, fontweight='bold', loc='left')
    bins = np.arange(0, 1501, 25)

    for idx, g in enumerate(SHOWCASE_GENS):
        lengths = [r.length for r in data[g].raw_reads]
        ax.hist(lengths, bins=bins, alpha=0.5, color=WONG[idx],
                label=f'Gen {g}', density=True, edgecolor='none')

    ax.axvline(EXPECTED_LENGTH, color='gray', ls='--', lw=1, alpha=0.6)
    ax.axvspan(TARGET_MIN, TARGET_MAX, alpha=0.05, color='gray')
    ax.set_xlabel('Read length [bp]')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1200)
    ax.legend(frameon=False, fontsize=7)
    remove_spines(ax)

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'summary_figure.pdf'))

    return {
        'target_fracs': target_fracs,
        'accuracies': accuracies,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='analysis 06 - summary figure')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 06 - summary figure", flush=True)
    data = load_all_generations(decode=True, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
