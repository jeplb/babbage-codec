#!/usr/bin/env python3
"""
section 2.11 - analysis 07 - consensus read depth requirements.

for each generation, bootstrap-subsample reads at various depths and
measure how often confidence-weighted majority vote produces the correct
consensus float. quantifies how many reads are needed for reliable
decoding as PCR errors accumulate.
"""

import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import (
    load_all_generations, N_GENERATIONS, TARGET_MIN, TARGET_MAX,
    GROUND_TRUTH_BITS,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR

DEPTHS = [3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_TRIALS = 200
SEED = 42
THRESHOLD = 0.95


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=True, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))
    rng = np.random.default_rng(SEED)

    # collect reads per (generation, constant) as raw lists
    pool_raw = defaultdict(list)
    for g in gens:
        for d in data[g].decoded_reads:
            r = d.result
            if (TARGET_MIN <= d.raw.length <= TARGET_MAX
                    and r.passed_filter
                    and r.predicted_constant is not None
                    and r.bits is not None
                    and len(r.bits) == 32
                    and all(b in (0, 1) for b in r.bits)):
                pool_raw[(g, r.predicted_constant)].append(
                    (r.bits, r.confidence))

    # convert to numpy arrays for fast vectorized consensus
    pool = {}
    for key, entries in pool_raw.items():
        bits_arr = np.array([b for b, c in entries], dtype=np.float64)
        conf_arr = np.array([c for b, c in entries], dtype=np.float64)
        pool[key] = (bits_arr, conf_arr)

    # ground truth bit arrays for direct comparison (avoids bits_to_float)
    gt_bits = {name: np.array(bl, dtype=int)
               for name, bl in GROUND_TRUTH_BITS.items()}

    # bootstrap consensus at each depth
    results = {}
    tsv_rows = []

    for g in gens:
        constants_g = sorted(set(
            name for (gg, name) in pool if gg == g))
        for name in constants_g:
            if name not in gt_bits:
                continue
            bits_arr, conf_arr = pool[(g, name)]
            n_reads = len(bits_arr)
            gt = gt_bits[name]
            for depth in DEPTHS:
                if n_reads < depth:
                    continue
                n_correct = 0
                for _ in range(N_TRIALS):
                    idx = rng.choice(n_reads, size=depth, replace=False)
                    sub_bits = bits_arr[idx]
                    sub_conf = conf_arr[idx]
                    # confidence-weighted majority vote
                    weighted = (sub_bits * sub_conf[:, None]).sum(axis=0)
                    total_w = sub_conf.sum()
                    consensus = (weighted / total_w > 0.5).astype(int)
                    if np.array_equal(consensus, gt):
                        n_correct += 1
                acc = n_correct / N_TRIALS
                results[(g, name, depth)] = (N_TRIALS, n_correct, acc)
                tsv_rows.append({
                    'generation': g,
                    'constant': name,
                    'read_depth': depth,
                    'n_trials': N_TRIALS,
                    'n_correct': n_correct,
                    'accuracy': acc,
                })

    # save tsv
    tsv_path = os.path.join(FIGURE_DIR, '07_consensus_depth.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\tconstant\tread_depth\t'
                'n_trials\tn_correct\taccuracy\n')
        for row in tsv_rows:
            f.write(f"{row['generation']}\t{row['constant']}"
                    f"\t{row['read_depth']}\t{row['n_trials']}"
                    f"\t{row['n_correct']}\t{row['accuracy']:.4f}\n")
    print(f"    saved 07_consensus_depth.tsv", flush=True)

    # figure 1: consensus accuracy vs read depth (line per generation)
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    cmap = plt.cm.viridis
    gen_colors = {g: cmap(i / max(N_GENERATIONS - 1, 1))
                  for i, g in enumerate(gens)}

    for g in gens:
        depths_plot = []
        accs_plot = []
        for depth in DEPTHS:
            acc_vals = [
                results[(g, name, depth)][2]
                for name in sorted(set(n for (gg, n) in pool if gg == g))
                if (g, name, depth) in results
            ]
            if acc_vals:
                depths_plot.append(depth)
                accs_plot.append(np.mean(acc_vals))
        if depths_plot:
            ax.plot(depths_plot, accs_plot, 'o-', color=gen_colors[g],
                    markersize=3, lw=1.2, label=f'Gen {g}')

    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Read depth')
    ax.set_ylabel('Consensus accuracy')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc='lower right')
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'consensus_depth.pdf'))

    # figure 2: minimum reads for 95% consensus vs generation
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))

    # for each generation, find minimum depth where all testable constants
    # (those with >= depth reads) achieve >= 95%. constants with too few
    # reads are skipped - they reflect limited coverage, not decoding failure.
    MIN_TESTABLE = 8  # require at least this many constants testable
    min_depth = {}
    for g in gens:
        constants_g = sorted(set(
            name for (gg, name) in pool if gg == g))
        if not constants_g:
            min_depth[g] = None
            continue
        found = None
        for depth in DEPTHS:
            testable = [
                name for name in constants_g
                if (g, name, depth) in results
            ]
            if len(testable) < MIN_TESTABLE:
                continue
            all_pass = all(
                results[(g, name, depth)][2] >= THRESHOLD
                for name in testable
            )
            if all_pass:
                found = depth
                break
        min_depth[g] = found

    reached = [(g, d) for g, d in min_depth.items() if d is not None]
    not_reached = [g for g, d in min_depth.items() if d is None]

    if reached:
        ax.plot([g for g, d in reached], [d for g, d in reached],
                'o-', color=WONG[0], markersize=7, lw=2)
    if not_reached:
        ax.plot(not_reached, [DEPTHS[-1]] * len(not_reached),
                'o', color=WONG[1], markersize=7, markerfacecolor='none',
                markeredgewidth=1.5, label='95% not reached')
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Minimum reads for 95% consensus')
    ax.set_xticks(gens)
    ax.set_yscale('log')
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'consensus_min_reads.pdf'))

    # summary
    for g in gens:
        d = min_depth.get(g)
        tag = str(d) if d else ">500"
        constants_g = sorted(set(n for (gg, n) in pool if gg == g))
        n_reads = sum(len(pool[(g, n)][0]) for n in constants_g)
        print(f"    gen {g:2d}: {len(constants_g):2d} constants, "
              f"{n_reads:5d} reads, min depth for 95% = {tag}", flush=True)

    return {'results': results, 'min_depth_95': min_depth}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='analysis 07 - consensus depth requirements')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 07 - consensus depth", flush=True)
    data = load_all_generations(decode=True, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data, quick=args.quick, quick_n=args.quick_n)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
