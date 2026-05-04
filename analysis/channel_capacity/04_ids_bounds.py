#!/usr/bin/env python3
"""
section 4.04 - IDS channel capacity bounds.

computes binary insertion-deletion-substitution (IDS) channel capacity bounds
using empirical error rates from barcode02 (PrimeStar GXL, 25c). the Babbage
transition encoding maps 1 bit per nucleotide, so binary capacity bounds
apply directly.

error rates come from analysis/error_rates/summary.tsv, which aligns reads
against the actual babbage.fasta oligos (not the compressed transition patterns).

bounds:
  - deletion upper: Cheraghchi 2019, C ≤ 1 - d·log2(4/φ)
  - deletion lower: Kanoria-Montanari 2013, C ≥ 1 - H(d)
  - IDS upper: C_del_upper (subs and insertions can only reduce capacity)
  - IDS lower: 1 - H(d) - H(s)  (ignores insertions - noted explicitly)
"""

import os
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ANALYSIS_DIR)

from shared import remove_spines, save_fig, PHI, WONG, plt, np

FIGURE_DIR = SCRIPT_DIR

# golden ratio (used in Cheraghchi bound)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

# error rates summary (aligned against actual babbage.fasta oligos)
ERROR_RATES_TSV = os.path.join(ANALYSIS_DIR, 'error_rates', 'summary.tsv')

# consensus convergence data
CONVERGENCE_TSV = os.path.join(
    ANALYSIS_DIR, 'serial_amplification', '08_consensus_convergence.tsv',
)


def binary_entropy(p):
    """binary entropy H(p) in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def cheraghchi_del_upper(d):
    """deletion channel upper bound (Cheraghchi 2019).
    C ≤ 1 - d·log2(4/φ) for binary deletion channel, d < 0.5.
    """
    if d <= 0:
        return 1.0
    return 1 - d * math.log2(4 / GOLDEN_RATIO)


def kanoria_montanari_del_lower(d):
    """deletion channel lower bound (Kanoria-Montanari 2013).
    C ≥ 1 - H(d) for small d.
    """
    return 1 - binary_entropy(d)


def ids_lower_bound(s, d):
    """IDS lower bound (Gallager-style): 1 - H(d) - H(s).
    caveat: ignores insertion channel - lower bound is optimistic.
    """
    return 1 - binary_entropy(d) - binary_entropy(s)


def load_error_rates(tsv_path, barcode='barcode02'):
    """load sub and del rates from section2 summary TSV."""
    with open(tsv_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            row = dict(zip(header, fields))
            if row['barcode'] == barcode:
                return {
                    'sub': float(row['sub_rate']),
                    'del': float(row['del_rate']),
                }
    raise ValueError(f"barcode {barcode} not found in {tsv_path}")


def main():
    print("section 4.04: IDS channel capacity bounds")

    # load error rates (aligned against actual babbage.fasta oligos)
    print("  loading error rates from error_rates/summary.tsv...")
    rates = load_error_rates(ERROR_RATES_TSV)
    s = rates['sub']
    d = rates['del']
    print(f"    substitution rate: {s:.4f}")
    print(f"    deletion rate:     {d:.4f}")

    # compute bounds
    C_del_upper = cheraghchi_del_upper(d)
    C_del_lower = kanoria_montanari_del_lower(d)
    C_ids_upper = C_del_upper  # subs/ins only reduce capacity
    C_ids_lower = ids_lower_bound(s, d)

    # Babbage rates
    babbage_noiseless = 1.0  # 32 bits / 32 nt

    # single-read effective rate: approximate as (1 - per-bit error rate)
    # per-bit error ≈ s + d (each corrupts ~1 bit)
    per_bit_err = s + d
    babbage_single = babbage_noiseless * (1 - per_bit_err)

    # consensus rate from throughput data
    conv_data = np.genfromtxt(CONVERGENCE_TSV, delimiter='\t', skip_header=1,
                              dtype=[('gen', int), ('depth', int), ('rep', int),
                                     ('n_correct', int), ('frac', float)])
    gen1_depths = sorted(set(conv_data['depth'][conv_data['gen'] == 1]))
    best_throughput = 0.0
    best_depth = 0
    for depth in gen1_depths:
        mask = (conv_data['gen'] == 1) & (conv_data['depth'] == depth)
        med_frac = np.median(conv_data['frac'][mask])
        tp = 32 * 17 * med_frac / depth
        if tp > best_throughput:
            best_throughput = tp
            best_depth = depth

    babbage_consensus = best_throughput
    print(f"  gen 1 optimal: {best_throughput:.4f} bits/read at depth {best_depth}")

    print(f"\n  capacity bounds (bits/nt):")
    print(f"    IDS upper (Cheraghchi del): {C_ids_upper:.4f}")
    print(f"    deletion lower (K-M):       {C_del_lower:.4f}")
    print(f"    IDS lower (Gallager):        {C_ids_lower:.4f}")
    print(f"    Babbage noiseless:           {babbage_noiseless:.4f}")
    print(f"    Babbage single-read:         {babbage_single:.4f}")

    # plot: horizontal bar chart
    entries = [
        ('IDS upper bound\n(Cheraghchi 2019)',          C_ids_upper,      WONG[5]),
        ('Deletion lower bound\n(Kanoria-Montanari)',   C_del_lower,      WONG[5]),
        ('IDS lower bound*\n(Gallager-style)',          C_ids_lower,      WONG[5]),
        ('Babbage noiseless\n(1.0 bit/nt)',             babbage_noiseless, WONG[2]),
        ('Babbage single-read',                         babbage_single,    WONG[0]),
    ]

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig_w = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    y_pos = np.arange(len(labels))[::-1]

    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Rate [bits/nt]')
    ax.set_xlim(0, 1.15)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    # footnote about IDS lower bound
    ax.text(0.02, -0.08,
            '* lower bound excludes insertion degradation and overstates achievable rate',
            transform=ax.transAxes, fontsize=7, fontstyle='italic', color='gray')

    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'ids_bounds.pdf'))

    # save TSV
    tsv_path = os.path.join(FIGURE_DIR, '04_ids_bounds.tsv')
    with open(tsv_path, 'w') as f:
        f.write("metric\tvalue_bits_per_nt\n")
        f.write(f"ids_upper_cheraghchi\t{C_ids_upper:.6f}\n")
        f.write(f"del_lower_kanoria_montanari\t{C_del_lower:.6f}\n")
        f.write(f"ids_lower_gallager\t{C_ids_lower:.6f}\n")
        f.write(f"babbage_noiseless\t{babbage_noiseless:.6f}\n")
        f.write(f"babbage_single_read\t{babbage_single:.6f}\n")
        f.write(f"babbage_consensus_bits_per_read\t{babbage_consensus:.6f}\n")
        f.write(f"\n")
        f.write(f"# empirical error rates (barcode02, vs actual oligos)\n")
        f.write(f"substitution_rate\t{s:.6f}\n")
        f.write(f"deletion_rate\t{d:.6f}\n")
        f.write(f"consensus_optimal_depth\t{best_depth}\n")
    print(f"    saved 04_ids_bounds.tsv")

    print("  done.")


if __name__ == '__main__':
    main()
