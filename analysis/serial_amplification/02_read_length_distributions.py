#!/usr/bin/env python3
"""
section 2.11 - analysis 02 - read length distributions per generation.

histogram grid, target fraction vs generation, exponential decay fit.
"""

import os
import sys

import numpy as np
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import (
    load_all_generations, N_GENERATIONS, TARGET_MIN, TARGET_MAX,
    EXPECTED_LENGTH,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR


def _exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=False, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))

    # collect lengths and target fractions
    all_lengths = {}
    target_fracs = []
    for g in gens:
        lengths = [r.length for r in data[g].raw_reads]
        all_lengths[g] = lengths
        n_total = len(lengths)
        n_target = sum(1 for ln in lengths if TARGET_MIN <= ln <= TARGET_MAX)
        target_fracs.append(n_target / n_total if n_total > 0 else 0.0)

    # histogram grid (3 rows x 4 cols)
    fig, axes = plt.subplots(3, 4, figsize=(12, 12 / PHI))
    axes_flat = axes.flatten()
    bins = np.arange(0, 2001, 25)

    for idx, g in enumerate(gens):
        ax = axes_flat[idx]
        lengths = all_lengths[g]
        ax.hist(lengths, bins=bins, color=WONG[0], alpha=0.8, edgecolor='none')
        ax.axvspan(TARGET_MIN, TARGET_MAX, alpha=0.12, color=WONG[1])
        ax.set_title(f'Gen {g}', fontsize=12)
        ax.set_xlim(0, 1500)
        if idx >= 8:
            ax.set_xlabel('Read length [bp]', fontsize=12)
        if idx % 4 == 0:
            ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(labelsize=12)
        remove_spines(ax)

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'length_histograms.pdf'))

    # target fraction vs generation
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    ax.plot(gens, target_fracs, 'o-', color=WONG[0], lw=2, markersize=6,
            label='Observed')

    # fit exponential decay: f(x) = a * exp(-b*x) + c
    gens_arr = np.array(gens, dtype=float)
    fracs_arr = np.array(target_fracs)
    fit_params = None
    r_squared = None
    try:
        p0 = [fracs_arr[0], 0.1, 0.01]
        popt, pcov = curve_fit(_exp_decay, gens_arr, fracs_arr, p0=p0,
                               bounds=([0, 0, 0], [1, 10, 1]), maxfev=5000)
        fit_params = popt
        # goodness of fit
        y_pred = _exp_decay(gens_arr, *popt)
        ss_res = np.sum((fracs_arr - y_pred) ** 2)
        ss_tot = np.sum((fracs_arr - np.mean(fracs_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        x_fit = np.linspace(1, N_GENERATIONS, 100)
        y_fit = _exp_decay(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color=WONG[1], lw=1.5,
                label=f'Exp. decay ($\\lambda$={popt[1]:.3f}, $R^2$={r_squared:.3f})')
    except (RuntimeError, ValueError) as e:
        print(f"    warning: exponential decay fit failed: {e}", flush=True)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Target fraction (200-300 bp)')
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'target_fraction.pdf'))

    # per-generation target loss rate from decay constant.
    # note: 1 - exp(-lambda) is the per-generation probability of a target read
    # leaving the target window. this assumes pure first-order decay and does not
    # account for the asymptotic floor c (i.e., a fraction c/(a+c) of reads may
    # be resistant to chimera formation).
    target_loss_rate = None
    if fit_params is not None:
        lam = fit_params[1]
        target_loss_rate = 1 - np.exp(-lam)

    # summary tsv
    tsv_path = os.path.join(FIGURE_DIR, '02_length_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\ttotal_reads\ttarget_reads\ttarget_fraction'
                '\tmedian_length\tmean_length\n')
        for g in gens:
            lengths = all_lengths[g]
            n_total = len(lengths)
            n_target = sum(1 for ln in lengths if TARGET_MIN <= ln <= TARGET_MAX)
            frac = n_target / n_total if n_total > 0 else 0.0
            med = float(np.median(lengths)) if lengths else 0.0
            mean = float(np.mean(lengths)) if lengths else 0.0
            f.write(f"{g}\t{n_total}\t{n_target}\t{frac:.4f}"
                    f"\t{med:.0f}\t{mean:.1f}\n")
        if fit_params is not None:
            f.write(f"\n# exponential decay fit: a={fit_params[0]:.4f}"
                    f" lambda={fit_params[1]:.4f} c={fit_params[2]:.4f}\n")
            if r_squared is not None:
                f.write(f"# R-squared: {r_squared:.4f}\n")
            f.write(f"# per-generation target loss rate:"
                    f" {target_loss_rate:.4f}\n")
    print("    saved 02_length_summary.tsv", flush=True)

    return {
        'target_fracs': target_fracs,
        'fit_params': fit_params,
        'r_squared': r_squared,
        'target_loss_rate': target_loss_rate,
        'all_lengths': all_lengths,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='analysis 02 - read length distributions')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 02 - read length distributions", flush=True)
    data = load_all_generations(decode=False, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
