#!/usr/bin/env python3
"""
section 4.01 - empirical homopolymer channel matrix and mutual information.

loads barcode02 (PrimeStar GXL, 25c), aligns observed HP runs to reference
patterns, builds P(obs_len | ref_len, base) channel matrices, then computes
channel capacity via Blahut-Arimoto.

reference HP lengths come from babbage.fasta (the actual oligos ordered from
Twist, which include designed HP runs to simulate TdT synthesis). each constant
has ~118 variants with different HP lengths; we use the per-position median
designed length as the reference for each aligned run.
"""

import os
import sys
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import (
    remove_spines, save_fig, PHI, WONG, plt,
    DataLoader, PATTERNS, add_quick_arg, np,
    extract_homopolymer_runs, align_homopolymer_runs,
)
from babbage_sequence_list import DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER

FIGURE_DIR = SCRIPT_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FASTA_PATH = os.path.join(PROJECT_ROOT, 'data', 'babbage.fasta')

MAX_REF_LEN = 12
MAX_OBS_LEN = 20
BASES = ['A', 'C', 'T']


def parse_fasta_references(fasta_path):
    """parse babbage.fasta and extract per-variant HP run-length vectors.

    returns dict: constant_name -> list of numpy arrays, each shape (32,)
    containing the designed HP lengths for one variant.
    """
    seqs_by_constant = defaultdict(list)
    header = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if header and seq_lines:
                    seq = ''.join(seq_lines)
                    constant = header.split('constant=')[1]
                    seqs_by_constant[constant].append(seq)
                header = line.strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header and seq_lines:
            seq = ''.join(seq_lines)
            constant = header.split('constant=')[1]
            seqs_by_constant[constant].append(seq)

    fwd_len = len(DEFAULT_FORWARD_PRIMER)
    rev = DEFAULT_REVERSE_PRIMER

    # for each constant, store all variant HP vectors as numpy arrays
    ref_variants = {}
    for const_name, seqs in seqs_by_constant.items():
        variants = []
        for seq in seqs:
            core = seq[fwd_len:]
            if core.endswith(rev):
                core = core[:-len(rev)]
            core = core.rstrip('G')
            runs = extract_homopolymer_runs(core)
            lengths = np.array([length for _, length, _ in runs], dtype=np.int32)
            variants.append(lengths)
        ref_variants[const_name] = variants

    return ref_variants


def match_variant(obs_lengths, variants):
    """find the variant with minimum L1 distance to observed HP lengths.

    obs_lengths: numpy array of observed HP run lengths (from alignment).
    variants: list of numpy arrays of designed HP run lengths.
    returns the best-matching variant's HP length vector.
    """
    n_obs = len(obs_lengths)
    best_dist = np.inf
    best_variant = variants[0]
    for var in variants:
        n = min(n_obs, len(var))
        dist = np.sum(np.abs(obs_lengths[:n] - var[:n]))
        if dist < best_dist:
            best_dist = dist
            best_variant = var
    return best_variant


def build_channel_matrices(data, ref_variants):
    """build per-base and aggregate channel matrices from barcode data.

    for each read, aligns observed HP runs to the transition pattern, then
    matches to the closest babbage.fasta variant by L1 distance on run
    lengths. uses the matched variant's designed HP lengths as reference.

    returns (matrices, counts, pairs_by_base) where matrices are column-normalized
    conditional probabilities and counts are raw integer (obs, ref) tallies.
    """
    pairs_by_base = defaultdict(list)

    for r in data.reads:
        if not r.result.passed_filter or not r.result.predicted_constant:
            continue
        if r.trimmed is None:
            continue

        const_name = r.result.predicted_constant
        ref_pattern = PATTERNS.get(const_name)
        if ref_pattern is None:
            continue
        variants = ref_variants.get(const_name)
        if not variants:
            continue

        obs_stripped = r.trimmed.rstrip('G')
        if not obs_stripped:
            continue

        obs_runs = extract_homopolymer_runs(obs_stripped)
        ref_runs = extract_homopolymer_runs(ref_pattern)
        alignments, _ = align_homopolymer_runs(obs_runs, ref_runs)

        # extract observed lengths at matched reference positions
        obs_at_ref = []
        for aln in alignments:
            atype = aln[0]
            if atype == 'match':
                obs_at_ref.append(aln[3])
            elif atype == 'deletion':
                obs_at_ref.append(0)
            # insertions don't consume a reference position

        if not obs_at_ref:
            continue

        obs_arr = np.array(obs_at_ref, dtype=np.int32)
        best_var = match_variant(obs_arr, variants)

        # collect (ref_len, obs_len) pairs per base
        ref_pos = 0
        for aln in alignments:
            atype = aln[0]
            if atype in ('match', 'deletion'):
                base = aln[1]
                obs_len = aln[3] if atype == 'match' else 0
                if ref_pos < len(best_var):
                    pairs_by_base[base].append((int(best_var[ref_pos]), obs_len))
                ref_pos += 1

    # build raw count matrices first, then column-normalize for probabilities
    counts = {}
    for base in BASES:
        cnt = np.zeros((MAX_OBS_LEN + 1, MAX_REF_LEN + 1), dtype=np.int64)
        for ref_len, obs_len in pairs_by_base[base]:
            if 1 <= ref_len <= MAX_REF_LEN and 0 <= obs_len <= MAX_OBS_LEN:
                cnt[obs_len, ref_len] += 1
        counts[base] = cnt

    agg_counts = np.zeros((MAX_OBS_LEN + 1, MAX_REF_LEN + 1), dtype=np.int64)
    for base in BASES:
        agg_counts += counts[base]
    counts['aggregate'] = agg_counts

    matrices = {}
    for key, cnt in counts.items():
        mat = cnt.astype(np.float64)
        col_sums = mat.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        mat /= col_sums
        matrices[key] = mat

    return matrices, counts, pairs_by_base


def blahut_arimoto(W, max_iter=1000, tol=1e-10):
    """compute channel capacity via Blahut-Arimoto.

    W: channel matrix, W[y, x] = P(y|x), shape (n_y, n_x).
    only uses columns (input symbols) that have observations.
    returns (capacity_bits, input_distribution).
    """
    n_y, n_x = W.shape

    active = W.sum(axis=0) > 0
    if not active.any():
        return 0.0, np.zeros(n_x)
    W_active = W[:, active]
    n_active = W_active.shape[1]

    if n_active == 1:
        p_full = np.zeros(n_x)
        p_full[active] = 1.0
        return 0.0, p_full

    p = np.ones(n_active) / n_active

    for _ in range(max_iter):
        r = W_active @ p
        r[r == 0] = 1e-300

        log_ratio = np.zeros_like(W_active)
        mask = W_active > 0
        log_ratio[mask] = np.log(W_active[mask] / np.broadcast_to(r[:, None], W_active.shape)[mask])
        c = np.exp((W_active * log_ratio).sum(axis=0))

        p_new = p * c
        p_new /= p_new.sum()

        C_lower = np.log2(np.sum(p_new * c))
        C_upper = np.log2(np.max(c))
        if C_upper - C_lower < tol:
            p = p_new
            break
        p = p_new

    capacity = np.log2(np.sum(p * c))

    p_full = np.zeros(n_x)
    p_full[active] = p
    return capacity, p_full


def noise_entropy(col):
    """H(Y|X=x) for a single column of the channel matrix."""
    col = col[col > 0]
    if len(col) == 0:
        return 0.0
    return -np.sum(col * np.log2(col))


def compute_mean_ref_run_length(pairs_by_base):
    """mean reference run length across all observations."""
    all_ref = [ref_len for base in BASES for ref_len, _ in pairs_by_base[base]]
    if not all_ref:
        return 1.0
    return np.mean(all_ref)


def _fmt_count(n):
    if n >= 1e6:
        return f'{n/1e6:.1f}M'
    if n >= 1e3:
        return f'{n/1e3:.0f}K'
    return str(n)


def plot_channel_heatmap(matrices, counts_by_base):
    """heatmap of channel matrices: one panel per base + aggregate."""
    panels = BASES + ['aggregate']
    panel_labels = BASES + ['Aggregate']

    fig, axes = plt.subplots(1, 4, figsize=(12, 12 / PHI / 1.5),
                             gridspec_kw={'right': 0.88})

    # find min/max ref_len with data across all panels
    min_ref = MAX_REF_LEN
    max_ref = 1
    for key in panels:
        mat = matrices[key]
        for c in range(1, mat.shape[1]):
            if mat[:, c].sum() > 0:
                min_ref = min(min_ref, c)
                max_ref = max(max_ref, c)
    show_obs = 16

    for idx, (key, lbl) in enumerate(zip(panels, panel_labels)):
        ax = axes[idx]
        mat = matrices[key]
        # drop obs=0 row (deletion bin)
        mat_show = mat[1:show_obs, min_ref:max_ref + 1]
        n_ref = max_ref - min_ref + 1

        im = ax.imshow(mat_show, aspect='auto', origin='lower',
                       cmap='cividis', vmin=0, vmax=0.7)
        # obs = designed reference diagonal (display col = ref - min_ref, display row = obs - 1)
        diag_x_lo = 0
        diag_x_hi = max_ref - min_ref
        diag_y_lo = min_ref - 1
        diag_y_hi = max_ref - 1
        # clip to displayed obs range (rows 0..show_obs-2 after the [1:show_obs] slice)
        if diag_y_hi > show_obs - 2:
            diag_x_hi = diag_x_lo + (show_obs - 2 - diag_y_lo)
            diag_y_hi = show_obs - 2
        ax.plot([diag_x_lo, diag_x_hi], [diag_y_lo, diag_y_hi],
                color='white', linestyle='--', linewidth=0.8, alpha=0.7, zorder=3)
        ax.set_xlabel('Designed HP length')
        if idx == 0:
            ax.set_ylabel('Observed HP length')
        count_str = _fmt_count(counts_by_base.get(key, 0))
        ax.set_title(f'{lbl} (n={count_str})')
        ax.set_xticks(range(n_ref))
        ax.set_xticklabels(range(min_ref, max_ref + 1))
        # display row p corresponds to obs_len = p + 1 after the slice above
        ax.set_yticks(range(1, show_obs - 1, 2))
        ax.set_yticklabels(range(2, show_obs, 2))

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='P(observed | designed)')
    return fig


def main():
    parser = argparse.ArgumentParser(description='section 4.01: channel matrix')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 4.01: empirical channel matrix")

    # parse designed HP run-length vectors from babbage.fasta
    print("  parsing babbage.fasta...")
    ref_variants = parse_fasta_references(FASTA_PATH)
    for name in list(ref_variants.keys())[:3]:
        n_var = len(ref_variants[name])
        median_lens = np.median(np.array([v for v in ref_variants[name]]), axis=0).astype(int)
        print(f"    {name}: {n_var} variants, median HP = {list(median_lens[:8])}...")

    loader = DataLoader(quick=args.quick, quick_n=args.quick_n)
    print("  loading barcode02...")
    data = loader.load('merged', 'barcode02')
    print(f"  {len(data.reads)} reads loaded")

    print("  building channel matrices (variant matching)...")
    matrices, counts, pairs_by_base = build_channel_matrices(data, ref_variants)

    for base in BASES:
        n = len(pairs_by_base[base])
        print(f"    {base}: {n:,} aligned HP pairs")

    # Blahut-Arimoto capacity
    print("  computing channel capacity (Blahut-Arimoto)...")
    capacities = {}
    input_dists = {}
    for key in BASES + ['aggregate']:
        mat = matrices[key]
        W = mat[:, 1:]  # ref_len 1..MAX_REF_LEN
        cap, p = blahut_arimoto(W)
        capacities[key] = cap
        input_dists[key] = p
        print(f"    {key}: {cap:.4f} bits/HP run")

    mean_ref_len = compute_mean_ref_run_length(pairs_by_base)
    print(f"  mean reference run length: {mean_ref_len:.2f} nt")
    for key in BASES + ['aggregate']:
        bits_per_nt = capacities[key] / mean_ref_len if mean_ref_len > 0 else 0
        print(f"    {key}: {bits_per_nt:.4f} bits/nt")

    # noise entropy per column
    print("  noise entropy H(obs | ref_len) per base:")
    for key in BASES:
        mat = matrices[key]
        for rl in range(1, MAX_REF_LEN + 1):
            col = mat[:, rl]
            if col.sum() > 0:
                h = noise_entropy(col)
                print(f"    {key} ref_len={rl}: H = {h:.4f} bits")

    # plot
    print("  plotting...")
    counts_by_base = {base: len(pairs_by_base[base]) for base in BASES}
    total = sum(counts_by_base.values())
    counts_by_base['aggregate'] = total
    fig = plot_channel_heatmap(matrices, counts_by_base)
    save_fig(fig, os.path.join(FIGURE_DIR, 'channel_matrix.pdf'))

    # save capacity results
    cap_path = os.path.join(FIGURE_DIR, 'channel_capacity.txt')
    with open(cap_path, 'w') as f:
        f.write("channel capacity (Blahut-Arimoto)\n\n")
        f.write(f"mean reference run length: {mean_ref_len:.4f} nt\n\n")
        for key in BASES + ['aggregate']:
            cap = capacities[key]
            bits_nt = cap / mean_ref_len if mean_ref_len > 0 else 0
            f.write(f"{key}:\n")
            f.write(f"  capacity: {cap:.6f} bits/HP run\n")
            f.write(f"  capacity: {bits_nt:.6f} bits/nt\n")
            p = input_dists[key]
            nonzero = [(i + 1, p[i]) for i in range(len(p)) if p[i] > 1e-6]
            if nonzero:
                f.write(f"  capacity-achieving distribution:\n")
                for ref_len, prob in nonzero:
                    f.write(f"    P(ref_len={ref_len}) = {prob:.6f}\n")
            f.write("\n")
    print(f"    saved channel_capacity.txt")

    # save raw TSV - counts and probabilities both, so downstream can re-bin
    tsv_path = os.path.join(FIGURE_DIR, '01_channel_matrix.tsv')
    with open(tsv_path, 'w') as f:
        f.write("base\tref_len\tobs_len\tcount\tprobability\n")
        for key in BASES + ['aggregate']:
            mat = matrices[key]
            cnt = counts[key]
            for ref_len in range(1, MAX_REF_LEN + 1):
                for obs_len in range(MAX_OBS_LEN + 1):
                    n = int(cnt[obs_len, ref_len])
                    if n > 0:
                        p = mat[obs_len, ref_len]
                        f.write(f"{key}\t{ref_len}\t{obs_len}\t{n}\t{p:.8f}\n")
    print(f"    saved 01_channel_matrix.tsv")

    print("  done.")


if __name__ == '__main__':
    main()
