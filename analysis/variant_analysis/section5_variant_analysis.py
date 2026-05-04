#!/usr/bin/env python3
"""
section 5 - variant-level channel analysis.

reads all_variant_alignments.tsv (produced by variant_decoder + collect),
computes per-base channel matrices, noise entropy, single-read accuracy
vs variant mean run length, and summary statistics.

figures saved alongside the script.
"""

import os
import sys
import math
from collections import defaultdict

import numpy as np

import logging
logging.getLogger('fontTools').setLevel(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from babbage_sequence_list import CONSTANTS
from error_analysis import extract_homopolymer_runs

# publication defaults
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

PHI = (1 + 5 ** 0.5) / 2
WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = SCRIPT_DIR

TSV_PATH = os.path.join(PROJECT_ROOT, 'all_variant_alignments.tsv')
FASTA_PATH = os.path.join(PROJECT_ROOT, 'data', 'babbage.fasta')

PATTERNS = {name: pattern for name, pattern in CONSTANTS}
BASES = ['A', 'C', 'T']
MAX_REF_LEN = 15
MAX_OBS_LEN = 20

# primers for FASTA trimming
from babbage_sequence_list import DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER


def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


def _fmt_count(n):
    if n >= 1e6:
        return f'{n/1e6:.1f}M'
    if n >= 1e3:
        return f'{n/1e3:.0f}K'
    return str(n)


def load_tsv(path):
    """load variant alignment TSV.

    returns list of dicts with keys: constant_name, read_id, variant_idx,
    score, designed (list of 32 ints), observed (list of 32 ints).
    """
    rows = []
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            const = parts[0]
            read_id = parts[1]
            vidx = int(parts[2])
            score = float(parts[3])

            designed = []
            observed = []
            for i in range(4, len(parts) - 1, 2):
                designed.append(int(parts[i]))
                observed.append(int(parts[i + 1]))

            rows.append({
                'constant_name': const,
                'read_id': read_id,
                'variant_idx': vidx,
                'score': score,
                'designed': designed,
                'observed': observed,
            })
    return rows


def get_position_bases():
    """return dict: constant_name -> list of 32 base characters."""
    return {name: list(pat) for name, pat in CONSTANTS}


def parse_variant_hp_lengths(fasta_path):
    """parse babbage.fasta -> dict of constant_name -> np.ndarray(n_variants, 32).

    each row is one variant's 32 designed HP lengths.
    """
    fwd_len = len(DEFAULT_FORWARD_PRIMER)
    rev = DEFAULT_REVERSE_PRIMER

    seqs_by_const = defaultdict(list)
    header = None
    seq_lines = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if header and seq_lines:
                    seq = ''.join(seq_lines)
                    const = header.split('constant=')[1]
                    seqs_by_const[const].append(seq)
                header = line.strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header and seq_lines:
            seq = ''.join(seq_lines)
            const = header.split('constant=')[1]
            seqs_by_const[const].append(seq)

    hp_by_const = {}
    for const_name, seqs in seqs_by_const.items():
        rows = []
        for seq in seqs:
            core = seq[fwd_len:]
            if core.endswith(rev):
                core = core[:-len(rev)]
            core = core.rstrip('G')
            runs = extract_homopolymer_runs(core)
            lengths = [length for _, length, _ in runs]
            rows.append(lengths)
        hp_by_const[const_name] = np.array(rows, dtype=np.int32)
    return hp_by_const


def reassign_variants(rows, hp_by_const):
    """reassign each read to the variant with the closest HP match.

    the viterbi self-loop override provides almost no path differentiation -
    all 118 variants produce the same observed HP counts. so the original
    variant assignment is unreliable.

    instead, we compare each read's observed HP lengths against all variants
    of its assigned constant using sum of absolute differences, and pick
    the closest match. this gives a much better channel matrix.
    """
    n_reassigned = 0
    for row in rows:
        const = row['constant_name']
        hp_matrix = hp_by_const.get(const)
        if hp_matrix is None:
            continue

        obs = np.array(row['observed'], dtype=np.int32)
        n_pos = min(len(obs), hp_matrix.shape[1])
        obs_trim = obs[:n_pos]

        # sum of absolute differences per variant
        diffs = np.abs(hp_matrix[:, :n_pos] - obs_trim[np.newaxis, :]).sum(axis=1)
        best_vi = int(np.argmin(diffs))

        if best_vi != row['variant_idx']:
            n_reassigned += 1
        row['variant_idx'] = best_vi
        row['designed'] = hp_matrix[best_vi].tolist()

    return n_reassigned


def build_channel_matrices(rows):
    """build P(observed | designed) per base and aggregate.

    includes deletions (observed=0) in the channel matrix - they represent
    positions where the viterbi path skipped a match state entirely.
    returns matrices dict, counts dict (raw integer tallies), pairs_by_base
    dict, deletion counts, and per-base totals.
    """
    pos_bases = get_position_bases()
    pairs_by_base = defaultdict(list)
    deletions_by_base = defaultdict(int)
    total_by_base = defaultdict(int)

    for row in rows:
        const = row['constant_name']
        bases = pos_bases.get(const)
        if bases is None:
            continue
        for i in range(min(32, len(row['designed']), len(row['observed']))):
            d = row['designed'][i]
            o = row['observed'][i]
            if d <= 0:
                continue
            base = bases[i]
            if base == 'G':
                continue
            total_by_base[base] += 1
            pairs_by_base[base].append((d, o))
            if o == 0:
                deletions_by_base[base] += 1

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

    return matrices, counts, pairs_by_base, deletions_by_base, total_by_base


def noise_entropy(col):
    """H(Y|X=x) for a single column of the channel matrix."""
    col = col[col > 0]
    if len(col) == 0:
        return 0.0
    return -np.sum(col * np.log2(col))


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
        log_ratio[mask] = np.log(
            W_active[mask] / np.broadcast_to(r[:, None], W_active.shape)[mask])
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


def plot_channel_heatmap(matrices, counts_by_base):
    """4-panel heatmap: A, C, T, aggregate."""
    panels = BASES + ['aggregate']
    panel_labels = BASES + ['Aggregate']

    fig, axes = plt.subplots(1, 4, figsize=(12, 12 / PHI / 1.5),
                             gridspec_kw={'right': 0.88})

    # find min/max ref_len with data
    min_ref = MAX_REF_LEN
    max_ref = 1
    for key in panels:
        mat = matrices[key]
        for c in range(1, mat.shape[1]):
            if mat[:, c].sum() > 0:
                min_ref = min(min_ref, c)
                max_ref = max(max_ref, c)
    show_obs = max_ref + 4  # show a few rows above max_ref

    vmax = 0.6  # covers the data max (~0.60) without saturation, tight enough for contrast

    for idx, (key, lbl) in enumerate(zip(panels, panel_labels)):
        ax = axes[idx]
        mat = matrices[key]
        # drop obs=0 row (deletion bin) - display row p corresponds to obs_len = p + 1
        mat_show = mat[1:show_obs, min_ref:max_ref + 1]

        im = ax.imshow(mat_show, aspect='auto', origin='lower',
                       cmap='cividis', vmin=0, vmax=vmax)

        # obs = designed reference diagonal in display coords (col, row=obs-1)
        n_ref = max_ref - min_ref + 1
        diag_x_lo = 0
        diag_x_hi = n_ref - 1
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
        ax.set_xticks(range(0, n_ref, 2))
        ax.set_xticklabels(range(min_ref, max_ref + 1, 2))
        # display row p corresponds to obs_len = p + 1 after the slice above
        ax.set_yticks(range(1, show_obs - 1, 2))
        ax.set_yticklabels(range(2, show_obs, 2))

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='P(observed | designed)')
    return fig


def plot_noise_entropy(matrices, pairs_by_base):
    """noise entropy H(obs | designed=L) vs designed run length, per base + aggregate."""
    fig, ax = plt.subplots(figsize=(5, 5 / PHI))

    ref_lens = list(range(3, MAX_REF_LEN + 1))
    colors = {
        'A': WONG[0], 'C': WONG[1], 'T': WONG[2], 'aggregate': WONG[3],
    }
    markers = {'A': 'o', 'C': 's', 'T': '^', 'aggregate': 'D'}

    for key in BASES + ['aggregate']:
        mat = matrices[key]
        entropies = []
        valid_lens = []
        for rl in ref_lens:
            col = mat[:, rl]
            if col.sum() > 0:
                entropies.append(noise_entropy(col))
                valid_lens.append(rl)
        if valid_lens:
            ax.plot(valid_lens, entropies, marker=markers[key], color=colors[key],
                    label=key if key != 'aggregate' else 'Aggregate',
                    markersize=5, linewidth=1.5)

    ax.set_xlabel('Designed HP length [bp]')
    ax.set_ylabel('Noise entropy [bits]')
    ax.legend(frameon=False, fontsize=8)
    remove_spines(ax)
    return fig


def compute_variant_accuracy(rows):
    """per-variant accuracy: fraction of reads whose argmax constant is correct.

    since we don't have an external classification_confidence, we use the fact
    that the argmax constant assignment IS the classification. we instead measure
    how well the 17-way argmax agrees across reads assigned to the same variant.

    actually, we compute: for each (constant, variant_idx), the mean designed
    HP run length and the number of reads. the "accuracy" concept requires
    ground truth from the original 17-way HMM - we approximate by computing
    the fraction of reads where the viterbi score is above a threshold that
    indicates a good alignment (not just the least-bad constant).

    returns list of (mean_designed_hp, n_reads, accuracy) tuples.
    """
    # group by (constant, variant_idx)
    variant_groups = defaultdict(list)
    for row in rows:
        key = (row['constant_name'], row['variant_idx'])
        variant_groups[key].append(row)

    results = []
    for (const, vidx), group_rows in variant_groups.items():
        if len(group_rows) < 5:
            continue
        # mean designed HP across 32 positions for this variant
        designed = group_rows[0]['designed']
        mean_hp = np.mean([d for d in designed if d > 0])

        # "accuracy" proxy: fraction of reads with score above median - 1 IQR
        # (reads that actually match this constant well vs noise reads assigned here)
        scores = np.array([r['score'] for r in group_rows])
        finite = scores[np.isfinite(scores)]
        if len(finite) < 5:
            continue
        # use a score threshold: reads with score > (median - 2*IQR) are "good"
        q25 = np.percentile(finite, 25)
        q75 = np.percentile(finite, 75)
        threshold = q25 - 1.5 * (q75 - q25)
        accuracy = np.mean(finite > threshold)

        results.append((mean_hp, len(group_rows), accuracy))

    return results


def plot_accuracy_vs_run_length(variant_data):
    """single-read accuracy vs mean variant run length with Wilson CI."""
    if not variant_data:
        return None

    # bin by mean run length (bin width 0.5)
    bins = defaultdict(lambda: {'n_total': 0, 'n_correct': 0})
    for mean_hp, n_reads, accuracy in variant_data:
        b = round(mean_hp * 2) / 2  # round to nearest 0.5
        bins[b]['n_total'] += n_reads
        bins[b]['n_correct'] += int(round(accuracy * n_reads))

    bin_centers = sorted(bins.keys())
    accuracies = []
    ci_lo = []
    ci_hi = []
    sizes = []

    for b in bin_centers:
        n = bins[b]['n_total']
        k = bins[b]['n_correct']
        p = k / n if n > 0 else 0

        # wilson confidence interval
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        lo = max(0, centre - margin)
        hi = min(1, centre + margin)

        accuracies.append(p)
        ci_lo.append(p - lo)
        ci_hi.append(hi - p)
        sizes.append(n)

    fig, ax = plt.subplots(figsize=(5, 5 / PHI))
    ax.errorbar(bin_centers, accuracies, yerr=[ci_lo, ci_hi],
                fmt='o', color=WONG[0], capsize=3, markersize=4, linewidth=1)
    ax.set_xlabel('Mean designed HP length [bp]')
    ax.set_ylabel('Single-read alignment quality')
    ax.set_ylim(0.5, 1.02)
    remove_spines(ax)
    return fig


def fit_laplace_mle(errors):
    """MLE fit of Laplace distribution: location=median, scale=MAD."""
    if len(errors) == 0:
        return 0.0, 1.0
    errors = np.array(errors)
    mu = np.median(errors)
    scale = np.mean(np.abs(errors - mu))
    return mu, max(scale, 1e-10)


def compute_summary(rows, pairs_by_base, deletions_by_base, total_by_base):
    """compute and print summary statistics."""
    n_total = len(rows)

    # per-base stats
    errors_by_base = defaultdict(list)
    for base in BASES:
        for d, o in pairs_by_base[base]:
            errors_by_base[base].append(o - d)

    print(f"\n{'='*60}")
    print(f"variant-level channel analysis summary")
    print(f"{'='*60}")
    print(f"total reads (after filter): {n_total:,}")
    print()

    # deletion rates
    print("per-base deletion rate:")
    total_del = 0
    total_pos = 0
    for base in BASES:
        d = deletions_by_base[base]
        t = total_by_base[base]
        rate = d / t if t > 0 else 0
        print(f"  {base}: {d:,} / {t:,} = {rate:.4f}")
        total_del += d
        total_pos += t
    print(f"  total: {total_del:,} / {total_pos:,} = {total_del/max(1,total_pos):.4f}")
    print()

    # mean absolute run-length error
    print("per-base mean absolute run-length error [bp]:")
    all_errors = []
    for base in BASES:
        errs = errors_by_base[base]
        mae = np.mean(np.abs(errs)) if errs else 0
        print(f"  {base}: {mae:.3f}  (n={len(errs):,})")
        all_errors.extend(errs)
    overall_mae = np.mean(np.abs(all_errors)) if all_errors else 0
    print(f"  total: {overall_mae:.3f}  (n={len(all_errors):,})")
    print()

    # laplace MLE per base
    print("Laplace scale parameter (MLE) per base:")
    summary_rows = []
    for base in BASES:
        errs = errors_by_base[base]
        mu, scale = fit_laplace_mle(errs)
        print(f"  {base}: location={mu:.3f}, scale={scale:.3f}")
        mae = np.mean(np.abs(errs)) if errs else 0
        del_rate = deletions_by_base[base] / max(1, total_by_base[base])
        summary_rows.append((base, len(errs), mae, del_rate, mu, scale))

    all_mu, all_scale = fit_laplace_mle(all_errors)
    print(f"  total: location={all_mu:.3f}, scale={all_scale:.3f}")

    overall_del_rate = total_del / max(1, total_pos)
    summary_rows.append(('total', len(all_errors), overall_mae, overall_del_rate,
                         all_mu, all_scale))

    return summary_rows


def write_summary_tsv(summary_rows, path):
    """write summary statistics TSV."""
    with open(path, 'w') as f:
        f.write('base\tn_pairs\tmean_abs_error\tdeletion_rate\t'
                'laplace_location\tlaplace_scale\n')
        for base, n, mae, dr, mu, scale in summary_rows:
            f.write(f'{base}\t{n}\t{mae:.4f}\t{dr:.4f}\t{mu:.4f}\t{scale:.4f}\n')
    print(f"  saved {os.path.basename(path)}")


def main():
    print("section 5: variant-level channel analysis")

    # load data
    print(f"  loading {TSV_PATH}...")
    rows = load_tsv(TSV_PATH)
    print(f"  {len(rows):,} reads loaded")

    # filter: exclude reads with -inf score (unaligned)
    filtered = [r for r in rows if np.isfinite(r['score'])]
    print(f"  {len(filtered):,} reads after removing -inf scores")

    # reassign variants by run-level matching.
    # the viterbi self-loop override provides negligible path differentiation -
    # all ~118 variants of a constant produce the same observed HP counts,
    # differing by only ~1 log-prob in total score. the original variant_idx
    # is therefore unreliable. instead, pick the variant whose designed HP
    # lengths are closest to the observed lengths (min sum of absolute diffs).
    print("  parsing variant HP lengths from FASTA...")
    hp_by_const = parse_variant_hp_lengths(FASTA_PATH)
    for name in sorted(hp_by_const):
        print(f"    {name}: {hp_by_const[name].shape[0]} variants")
    print("  reassigning variants by run-level matching...")
    n_reassigned = reassign_variants(filtered, hp_by_const)
    print(f"  {n_reassigned:,} / {len(filtered):,} reads reassigned "
          f"({n_reassigned/max(1,len(filtered))*100:.1f}%)")

    # build channel matrices
    print("  building channel matrices...")
    matrices, counts, pairs_by_base, del_by_base, total_by_base = build_channel_matrices(filtered)

    for base in BASES:
        n = len(pairs_by_base[base])
        print(f"    {base}: {n:,} HP pairs")

    # figure 1: channel matrix heatmap
    print("  plotting channel matrix heatmap...")
    n_by_base = {base: len(pairs_by_base[base]) for base in BASES}
    n_by_base['aggregate'] = sum(n_by_base.values())
    fig = plot_channel_heatmap(matrices, n_by_base)
    save_fig(fig, os.path.join(FIGURE_DIR, 'channel_matrix.pdf'))

    # figure 2: noise entropy
    print("  plotting noise entropy...")
    fig = plot_noise_entropy(matrices, pairs_by_base)
    save_fig(fig, os.path.join(FIGURE_DIR, 'noise_entropy.pdf'))

    # channel capacity via Blahut-Arimoto
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

    # mean reference run length
    all_ref = [ref_len for base in BASES for ref_len, _ in pairs_by_base[base]]
    mean_ref_len = np.mean(all_ref) if all_ref else 1.0
    print(f"  mean reference run length: {mean_ref_len:.2f} nt")
    for key in BASES + ['aggregate']:
        bits_per_nt = capacities[key] / mean_ref_len if mean_ref_len > 0 else 0
        print(f"    {key}: {bits_per_nt:.4f} bits/nt")

    # save channel capacity results
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
    print(f"  saved channel_capacity.txt")

    # save raw channel matrix TSV - counts and probabilities both
    cm_tsv_path = os.path.join(FIGURE_DIR, 'channel_matrix.tsv')
    with open(cm_tsv_path, 'w') as f:
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
    print(f"  saved channel_matrix.tsv")

    # figure 3: accuracy vs mean run length
    print("  computing per-variant accuracy...")
    variant_data = compute_variant_accuracy(filtered)
    fig = plot_accuracy_vs_run_length(variant_data)
    if fig:
        save_fig(fig, os.path.join(FIGURE_DIR, 'accuracy_vs_run_length.pdf'))

    # summary statistics
    summary_rows = compute_summary(
        filtered, pairs_by_base, del_by_base, total_by_base)
    write_summary_tsv(
        summary_rows,
        os.path.join(FIGURE_DIR, 'variant_analysis_summary.tsv'))

    # script already lives alongside figures

    print("\ndone.")


if __name__ == '__main__':
    main()
