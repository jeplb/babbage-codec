#!/usr/bin/env python3
"""
panel b aggregator -> turns panel_b_raw.tsv into the figure plus a tidy
aggregated TSV with mean and 5th / 95th percentile bands.

y-axis is fraction of 17 constants recovered, matching the original Fig 1
convention. x-axis is consensus depth (log).
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import plt, PHI, WONG, remove_spines, save_fig

RAW = os.path.join(SCRIPT_DIR, 'panel_b_raw.tsv')
AGG = os.path.join(SCRIPT_DIR, 'panel_b_aggregated.tsv')
PDF = os.path.join(SCRIPT_DIR, 'panel_b.pdf')


def load_raw(path):
    rows = []
    with open(path) as f:
        h = f.readline().rstrip().split('\t')
        for line in f:
            fs = line.rstrip().split('\t')
            d = dict(zip(h, fs))
            rows.append({
                'depth': int(d['depth']),
                'rep': int(d['replicate']),
                'sm': int(d['smolecule_recovered'])
                       if d['smolecule_recovered'] != 'None' else None,
                'hmm': int(d['hmm_recovered']),
            })
    return rows


def aggregate(rows):
    depths = sorted(set(r['depth'] for r in rows))
    out = []
    for d in depths:
        sub = [r for r in rows if r['depth'] == d]
        sm = [r['sm'] for r in sub if r['sm'] is not None]
        hmm = [r['hmm'] for r in sub]
        if not sm or not hmm:
            continue
        out.append({
            'depth': d,
            'n_reps': len(sm),
            'sm_mean': float(np.mean(sm)),
            'sm_p5': float(np.percentile(sm, 5)),
            'sm_p95': float(np.percentile(sm, 95)),
            'sm_min': int(min(sm)),
            'sm_max': int(max(sm)),
            'hmm_mean': float(np.mean(hmm)),
            'hmm_p5': float(np.percentile(hmm, 5)),
            'hmm_p95': float(np.percentile(hmm, 95)),
            'hmm_min': int(min(hmm)),
            'hmm_max': int(max(hmm)),
        })
    return out


def write_agg(agg, path):
    with open(path, 'w') as f:
        f.write('depth\tn_reps\tdecoder_mean\tdecoder_p5\tdecoder_p95\t'
                'smolecule_mean\tsmolecule_p5\tsmolecule_p95\n')
        for r in agg:
            f.write(f"{r['depth']}\t{r['n_reps']}\t"
                    f"{r['hmm_mean']:.3f}\t{r['hmm_p5']:.1f}\t{r['hmm_p95']:.1f}\t"
                    f"{r['sm_mean']:.3f}\t{r['sm_p5']:.1f}\t{r['sm_p95']:.1f}\n")


def plot(agg):
    fig_w = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / PHI))
    depths = np.array([r['depth'] for r in agg])
    # convert counts out of 17 to fractions, matches original Fig 1 axis.
    h_mean = np.array([r['hmm_mean'] / 17 for r in agg])
    h_lo = np.array([r['hmm_p5'] / 17 for r in agg])
    h_hi = np.array([r['hmm_p95'] / 17 for r in agg])
    s_mean = np.array([r['sm_mean'] / 17 for r in agg])
    s_lo = np.array([r['sm_p5'] / 17 for r in agg])
    s_hi = np.array([r['sm_p95'] / 17 for r in agg])

    ax.fill_between(depths, h_lo, h_hi, color=WONG[0], alpha=0.18)
    ax.plot(depths, h_mean, '-o', color=WONG[0], linewidth=1.6,
            markersize=4, label='Profile HMM decoder')
    ax.fill_between(depths, s_lo, s_hi, color=WONG[1], alpha=0.18)
    ax.plot(depths, s_mean, '-s', color=WONG[1], linewidth=1.6,
            markersize=4, label='Medaka smolecule')

    ax.axhline(1.0, color='grey', ls=':', lw=0.6)
    ax.text(depths[-1], 1.01, '17 / 17', fontsize=7, color='grey',
            ha='right', va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('Consensus depth [reads]')
    ax.set_ylabel('Fraction of 17 constants recovered')
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(min(depths) * 0.85, max(depths) * 1.15)
    remove_spines(ax)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    save_fig(fig, PDF)


def main():
    if not os.path.exists(RAW):
        sys.stderr.write(f"missing {RAW} - run depth_sweep_panel_b.py first\n")
        sys.exit(1)
    rows = load_raw(RAW)
    agg = aggregate(rows)
    write_agg(agg, AGG)
    plot(agg)
    print(f"wrote {AGG}")
    print(f"wrote {PDF}")
    print(f"\n{'depth':>5} {'decoder':>9} {'smolecule':>10}")
    for r in agg:
        print(f"{r['depth']:>5} {r['hmm_mean']:>9.2f} {r['sm_mean']:>10.2f}")
    # smolecule plateau check (last three depths)
    if len(agg) >= 3:
        sm_tail = float(np.mean([r['sm_mean'] for r in agg[-3:]]))
        sm_max  = max(r['sm_max'] for r in agg)
        print(f"\nsmolecule tail mean (last 3 depths): {sm_tail:.2f} of 17")
        print(f"smolecule max single observation: {sm_max} of 17")


if __name__ == '__main__':
    main()
