"""
section 1 - codeword design and properties

generates all 17 codewords, computes pairwise hamming distances, identifies
closest pair, and produces publication-quality figures characterizing the
codebook.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# publication defaults: arial, editable text in illustrator, golden ratio
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

PHI = (1 + 5 ** 0.5) / 2

from babbage_sequence_list import CONSTANTS
from binary_dna_codec import (
    Float32Parts, SCRAMBLE_SEQUENCE_32, encode_bits, scramble_bits,
    decode_dna, descramble_bits, START_BASE,
)

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

# wong palette (colorblind-safe)
WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']


# utilities

def hamming_distance(a, b):
    """pairwise hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def compute_distance_matrix(patterns):
    """compute symmetric hamming distance matrix for a list of patterns."""
    n = len(patterns)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(patterns[i], patterns[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def upper_triangle_distances(dist_matrix):
    """extract upper-triangle distances from a symmetric matrix."""
    n = dist_matrix.shape[0]
    return [int(dist_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]


def closest_pair(dist_matrix, names, patterns):
    """find closest pair and return (name1, name2, distance, diff_positions)."""
    n = len(names)
    best_d = dist_matrix.max() + 1
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < best_d:
                best_d = dist_matrix[i, j]
                best_i, best_j = i, j
    diff_pos = [k for k in range(len(patterns[best_i]))
                if patterns[best_i][k] != patterns[best_j][k]]
    return names[best_i], names[best_j], int(best_d), diff_pos


# codebook generation

def build_codebook():
    """decode all 17 codewords back to IEEE 754 metadata.

    returns list of dicts with keys:
        name, value, ieee_hex, ieee_bits, dna
    """
    rows = []
    for name, dna in CONSTANTS:
        scrambled_bits = decode_dna(dna, START_BASE)
        ieee_bits = descramble_bits(scrambled_bits, SCRAMBLE_SEQUENCE_32)
        parts = Float32Parts.from_bits(ieee_bits)
        value = parts.to_float()

        raw_int = (parts.sign << 31) | (parts.exponent << 23) | parts.mantissa
        ieee_hex = f"0x{raw_int:08X}"

        rows.append({
            'name': name,
            'value': value,
            'ieee_hex': ieee_hex,
            'ieee_bits': ieee_bits,
            'dna': dna,
        })
    return rows


# display names for heatmap axes (mathtext for symbols missing from arial)

DISPLAY_NAMES = {
    'electron_charge': r'$e^-$ charge',
    'electron_mass': r'$e^-$ mass',
    'proton_mass': r'$p^+$ mass',
    'golden_ratio': r'$\varphi$',
    'epsilon_0': r'$\epsilon_0$',
    'sqrt_2': r'$\sqrt{2}$',
    'fine_structure': r'$\alpha$',
    'hydrogen_frequency': 'H freq',
    'calculator_easter_egg': '5318008',
    'millionth_prime': r'$p(10^6)$',
    'Avogadro': r'$N_A$',
    'boltzmann': r'$k_B$',
    'hbar': r'$\hbar$',
}


# text output

def format_float(value):
    """format float handling signed zeros."""
    if value == 0.0:
        return '-0.0' if math.copysign(1.0, value) < 0 else '0.0'
    return f'{value:.8g}'


def print_table(rows):
    """table 1: all 17 codewords with name, ieee hex, dna sequence."""
    print("\ntable 1: codewords")
    print(f"{'name':<24} {'ieee 754 hex':>14}  {'32-base DNA sequence'}")
    print(f"{'─' * 24} {'─' * 14}  {'─' * 32}")
    for r in rows:
        print(f"{r['name']:<24} {r['ieee_hex']:>14}  {r['dna']}")


def print_distance_stats(dist_matrix, names, patterns):
    """print distance stats and closest-pair analysis."""
    dists = upper_triangle_distances(dist_matrix)
    print(f"\npairwise hamming distances (136 pairs)")
    print(f"  min:  {min(dists)}")
    print(f"  mean: {np.mean(dists):.1f}")
    print(f"  max:  {max(dists)}")

    n1, n2, d, diff_pos = closest_pair(dist_matrix, names, patterns)
    i1 = names.index(n1)
    i2 = names.index(n2)
    print(f"\n  closest pair: {n1} vs {n2} (distance = {d})")
    print(f"  differing positions: {diff_pos}")
    print(f"  {n1:>24}: {patterns[i1]}")
    marker = ''.join('^' if k in diff_pos else ' ' for k in range(len(patterns[i1])))
    print(f"  {'':>24}  {marker}")
    print(f"  {n2:>24}: {patterns[i2]}")


# figure 1a: histogram

def plot_histogram(dists, outpath):
    """histogram of pairwise hamming distances."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5 / PHI))

    lo = min(dists) - 0.5
    hi = max(dists) + 1.5
    bins = np.arange(lo, hi, 1)

    ax.hist(dists, bins=bins, color=WONG[0], edgecolor='white', linewidth=0.5)

    d_min = min(dists)
    ymax = ax.get_ylim()[1]
    ax.axvline(d_min, color=WONG[1], linestyle='--', linewidth=1, alpha=0.8)
    ax.text(d_min + 0.3, ymax * 0.92, f'min = {d_min}',
            color=WONG[1], fontsize=8, va='top')

    ax.set_xlabel('Hamming distance [bases]')
    ax.set_ylabel('Frequency [count]')
    ax.set_title('Pairwise codeword distances (n = 136 pairs)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nsaved {outpath}")


# figure 1b: heatmap

def plot_heatmap(dist_matrix, names, outpath):
    """17x17 heatmap of pairwise hamming distances."""
    display = [DISPLAY_NAMES.get(nm, nm) for nm in names]
    n = len(names)
    vmin = 0
    vmax = int(dist_matrix.max())

    fig, ax = plt.subplots(figsize=(6.5, 6.5 / PHI))

    im = ax.imshow(dist_matrix, cmap='cividis', interpolation='nearest',
                   vmin=vmin, vmax=vmax)

    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            frac = val / vmax if vmax > 0 else 0
            color = 'white' if frac < 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(display, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Hamming distance [bases]')

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {outpath}")


# figure 1c: closest-pair difference positions

def plot_closest_pair_positions(names, patterns, dist_matrix, outpath):
    """32-position dot plot showing where the closest pair differs."""
    n1, n2, d, diff_pos = closest_pair(dist_matrix, names, patterns)

    fig, ax = plt.subplots(figsize=(5.5, 5.5 / PHI))

    positions = np.arange(32)
    mask = np.array([1 if p in diff_pos else 0 for p in positions])
    ax.bar(positions[mask == 1], 1, width=0.8, color=WONG[0], alpha=0.85)
    ax.bar(positions[mask == 0], 1, width=0.8, color='#DDDDDD', alpha=0.4)

    # IEEE field boundaries
    ax.axvline(0.5, color='grey', linewidth=0.5, linestyle=':')
    ax.axvline(8.5, color='grey', linewidth=0.5, linestyle=':')

    # field labels
    ax.text(0, -0.15, 'sign', fontsize=7, color='grey', ha='center',
            transform=ax.get_xaxis_transform())
    ax.text(4.5, -0.15, 'exponent', fontsize=7, color='grey', ha='center',
            transform=ax.get_xaxis_transform())
    ax.text(20, -0.15, 'mantissa', fontsize=7, color='grey', ha='center',
            transform=ax.get_xaxis_transform())

    ax.set_xlabel('Bit position')
    ax.set_ylabel('Differs')
    ax.set_yticks([])
    ax.set_title(f'Closest pair: {n1} vs {n2} (d = {d})', fontsize=9)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {outpath}")


if __name__ == '__main__':
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # build codebook
    rows = build_codebook()
    names = [r['name'] for r in rows]
    patterns = [r['dna'] for r in rows]

    # table 1
    print_table(rows)

    # distance matrix
    dist_matrix = compute_distance_matrix(patterns)

    # stats and closest-pair analysis
    print_distance_stats(dist_matrix, names, patterns)

    # distances for histogram
    dists = upper_triangle_distances(dist_matrix)

    # figure 1a: histogram
    plot_histogram(dists, os.path.join(FIGURE_DIR, 'hamming_histogram.pdf'))

    # figure 1b: heatmap
    plot_heatmap(dist_matrix, names, os.path.join(FIGURE_DIR, 'hamming_heatmap.pdf'))

    # figure 1c: closest-pair positions
    plot_closest_pair_positions(
        names, patterns, dist_matrix,
        os.path.join(FIGURE_DIR, 'closest_pair_positions.pdf'))

    print(f"\nfigures saved to {FIGURE_DIR}")
