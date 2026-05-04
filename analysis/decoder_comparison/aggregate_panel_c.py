#!/usr/bin/env python3
"""
panel c aggregator -> turns panel_c_raw.tsv into the figure plus a tidy
aggregated TSV with mean and 5th / 95th percentile bands.

y-axis is fraction of 17 constants recovered. x-axis is generation 1 to 12.
the depth is fixed at 25 (see gen_trajectory_panel_c.py).
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import plt, PHI, WONG, remove_spines, save_fig

RAW = os.path.join(SCRIPT_DIR, 'panel_c_raw.tsv')
AGG = os.path.join(SCRIPT_DIR, 'panel_c_aggregated.tsv')
PDF = os.path.join(SCRIPT_DIR, 'panel_c.pdf')


def load_raw(path):
    rows = []
    with open(path) as f:
        h = f.readline().rstrip().split('\t')
        for line in f:
            fs = line.rstrip().split('\t')
            d = dict(zip(h, fs))
            rows.append({
                'gen': int(d['generation']),
                'rep': int(d['replicate']),
                'sm': int(d['smolecule_recovered'])
                       if d['smolecule_recovered'] != 'None' else None,
                'hmm': int(d['hmm_recovered']),
            })
    return rows


def aggregate(rows):
    gens = sorted(set(r['gen'] for r in rows))
    out = []
    for g in gens:
        sub = [r for r in rows if r['gen'] == g]
        sm = [r['sm'] for r in sub if r['sm'] is not None]
        hmm = [r['hmm'] for r in sub]
        if not sm or not hmm:
            continue
        out.append({
            'gen': g,
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
        f.write('generation\tn_reps\tdecoder_mean\tdecoder_p5\tdecoder_p95\t'
                'smolecule_mean\tsmolecule_p5\tsmolecule_p95\n')
        for r in agg:
            f.write(f"{r['gen']}\t{r['n_reps']}\t"
                    f"{r['hmm_mean']:.3f}\t{r['hmm_p5']:.1f}\t{r['hmm_p95']:.1f}\t"
                    f"{r['sm_mean']:.3f}\t{r['sm_p5']:.1f}\t{r['sm_p95']:.1f}\n")


def plot(agg):
    fig_w = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / PHI))
    x = np.array([r['gen'] for r in agg])
    h_mean = np.array([r['hmm_mean'] / 17 for r in agg])
    h_lo = np.array([r['hmm_p5'] / 17 for r in agg])
    h_hi = np.array([r['hmm_p95'] / 17 for r in agg])
    s_mean = np.array([r['sm_mean'] / 17 for r in agg])
    s_lo = np.array([r['sm_p5'] / 17 for r in agg])
    s_hi = np.array([r['sm_p95'] / 17 for r in agg])

    ax.fill_between(x, h_lo, h_hi, color=WONG[0], alpha=0.18)
    ax.plot(x, h_mean, '-o', color=WONG[0], linewidth=1.6,
            markersize=4, label='Profile HMM decoder')
    ax.fill_between(x, s_lo, s_hi, color=WONG[1], alpha=0.18)
    ax.plot(x, s_mean, '-s', color=WONG[1], linewidth=1.6,
            markersize=4, label='Medaka smolecule')

    ax.axhline(1.0, color='grey', ls=':', lw=0.6)
    ax.text(x[-1], 1.01, '17 / 17', fontsize=7, color='grey',
            ha='right', va='bottom')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fraction of 17 constants recovered')
    ax.set_xticks(x)
    ax.set_ylim(-0.02, 1.08)
    remove_spines(ax)
    ax.legend(fontsize=8, frameon=False, loc='center right')
    save_fig(fig, PDF)


def main():
    if not os.path.exists(RAW):
        sys.stderr.write(f"missing {RAW} - run gen_trajectory_panel_c.py first\n")
        sys.exit(1)
    rows = load_raw(RAW)
    agg = aggregate(rows)
    write_agg(agg, AGG)
    plot(agg)
    print(f"wrote {AGG}")
    print(f"wrote {PDF}")
    print(f"\n{'gen':>3} {'decoder mean':>13} {'smolecule mean':>15}")
    for r in agg:
        print(f"{r['gen']:>3} {r['hmm_mean']:>13.2f} {r['sm_mean']:>15.2f}")
    # widening summary
    if len(agg) >= 2:
        d_first = agg[0]['hmm_mean'] - agg[0]['sm_mean']
        d_last = agg[-1]['hmm_mean'] - agg[-1]['sm_mean']
        print(f"\ndisparity (decoder - smolecule):")
        print(f"  first gen: {d_first:.2f} of 17")
        print(f"  last gen : {d_last:.2f} of 17")


if __name__ == '__main__':
    main()
