#!/usr/bin/env python3
"""
section 2.04 - error rates.

per-position error rates (substitution, insertion, deletion) across the
32-position codeword. separate panels for polymerase comparison vs
cycle titration.

compares reads against the actual synthesized oligo pool (babbage.fasta)
rather than the canonical patterns, so that designed TdT HP variation
is not counted as error.
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import (
    DataLoader, RUNS, BARCODE_CONDITIONS, POLYMERASE_BARCODES, CYCLE_BARCODES,
    PATTERNS, GROUND_TRUTH, PROJECT_ROOT,
    label, remove_spines, save_fig, copy_script, add_quick_arg,
    PHI, WONG, plt,
)
from error_analysis import (
    extract_homopolymer_runs, align_homopolymer_runs, AlignmentErrorStats,
)
from binary_dna_codec import decode_dna

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUPS = [
    ('polymerase', 'Polymerase comparison', POLYMERASE_BARCODES),
    ('cycle', 'PCR cycle titration', CYCLE_BARCODES),
]

# oligo pool fasta
OLIGO_FASTA = os.path.join(PROJECT_ROOT, 'data', 'babbage.fasta')
FWD_PRIMER = 'AAAATGAACTCATGGGACCTTCCACCCCAT'
REV_PRIMER = 'ATGAGACCACGTCGAAGTTTGCGAGAGGAT'


def _load_oligo_pool():
    """load babbage.fasta, return dict: constant_name -> list of run representations.

    each oligo is stored as [(base, length), ...] after stripping primers
    and G-tail.
    """
    pool = defaultdict(list)
    with open(OLIGO_FASTA) as f:
        const = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                const = None
                for part in line.split():
                    if part.startswith('constant='):
                        const = part.split('=', 1)[1]
            elif const is not None:
                seq = line
                if seq.startswith(FWD_PRIMER):
                    seq = seq[len(FWD_PRIMER):]
                if seq.endswith(REV_PRIMER):
                    seq = seq[:-len(REV_PRIMER)]
                seq = seq.rstrip('G')
                if seq:
                    runs = extract_homopolymer_runs(seq)
                    pool[const].append(
                        [(b, l) for b, l, _p in runs])
                const = None
    return dict(pool)


def _build_oligo_index(oligo_pool):
    """precompute numpy arrays for fast variant selection.

    for each constant: canonical base order + (n_variants, max_runs) length matrix.
    all variants share the same base order (TdT only varies HP lengths),
    so we align the read once against the base order, then vectorize the
    run-length comparison across all variants.
    """
    index = {}
    for const, variants in oligo_pool.items():
        if not variants:
            continue
        # base order from the longest variant (most complete prefix)
        longest = max(variants, key=len)
        base_order = [b for b, _l in longest]
        max_runs = len(longest)
        # length matrix: nan where variant is shorter
        length_mat = np.full((len(variants), max_runs), np.nan)
        for i, v in enumerate(variants):
            for j, (_b, l) in enumerate(v):
                length_mat[i, j] = l
        index[const] = (base_order, length_mat, variants)
    return index


def _greedy_base_match(obs_runs, base_order):
    """match obs run bases to reference base order using greedy lookahead.

    returns parallel lists (obs_indices, ref_indices) of matched positions.
    """
    obs_matched = []
    ref_matched = []
    oi, ri = 0, 0
    n_obs, n_ref = len(obs_runs), len(base_order)
    while oi < n_obs and ri < n_ref:
        if obs_runs[oi][0] == base_order[ri]:
            obs_matched.append(oi)
            ref_matched.append(ri)
            oi += 1
            ri += 1
        else:
            skip_obs = (oi + 1 < n_obs
                        and obs_runs[oi + 1][0] == base_order[ri])
            skip_ref = (ri + 1 < n_ref
                        and base_order[ri + 1] == obs_runs[oi][0])
            if skip_obs and not skip_ref:
                oi += 1
            elif skip_ref and not skip_obs:
                ri += 1
            elif skip_obs and skip_ref:
                oi += 1
            else:
                oi += 1
                ri += 1
    return obs_matched, ref_matched


def _find_best_oligo(obs_runs, const, oligo_index):
    """find best-matching oligo using vectorized scoring + one full alignment.

    1. greedy base-match obs against canonical base order (one pass)
    2. vectorized run-length distance across all variants (numpy)
    3. full alignment against best variant for proper error stats

    returns (stats, n_ref_runs, score_margin) or (None, 0, 0).
    """
    if const not in oligo_index:
        return None, 0, 0.0
    base_order, length_mat, variants = oligo_index[const]

    # step 1: greedy base matching
    obs_matched, ref_matched = _greedy_base_match(obs_runs, base_order)
    if not obs_matched:
        return None, 0, 0.0

    # step 2: vectorized distance
    obs_lengths = np.array([obs_runs[i][1] for i in obs_matched])
    ref_idx = np.array(ref_matched)
    variant_lengths = length_mat[:, ref_idx]  # (n_variants, n_matched)
    diffs = np.abs(obs_lengths[None, :] - variant_lengths)
    # penalize positions where variant is too short (nan)
    diffs = np.where(np.isnan(diffs), 1000.0, diffs)
    scores = diffs.sum(axis=1)
    sorted_scores = np.sort(scores)
    best_idx = int(np.argmin(scores))
    # score margin: gap between best and second-best
    margin = (sorted_scores[1] - sorted_scores[0]) if len(sorted_scores) > 1 else 0.0

    # step 3: full alignment against best variant
    # trim observed runs to the region covered by the variant - otherwise
    # the ~20 excess runs beyond the variant get consumed as insertions
    best_variant = variants[best_idx]
    max_obs = max(obs_matched) + 2  # small slack for boundary mismatches
    trimmed_obs = obs_runs[:min(max_obs, len(obs_runs))]
    ref_runs = []
    pos = 0
    for b, l in best_variant:
        ref_runs.append((b, l, pos))
        pos += l
    _, stats = align_homopolymer_runs(trimmed_obs, ref_runs)
    return stats, len(best_variant), float(margin)


def _compute_error_stats(reads, oligo_index):
    """compute error stats using best-matching oligo as reference.

    also collects per-position error data (for truncation bias check)
    and score margins (for misassignment check).
    """
    global_stats = AlignmentErrorStats()
    n_aligned = 0
    margins = []
    per_read_stats = []  # (margin, subs, dels, hp_dev, ref_bases) per aligned read
    # per-position: track (n_errors, n_total) at each ref run position
    pos_hp_errors = defaultdict(list)  # ref_pos -> list of abs(hp_length_diff)
    pos_coverage = defaultdict(int)    # ref_pos -> n_reads covering this position

    for r in reads:
        if not r.result.passed_filter or not r.result.predicted_constant:
            continue
        if r.trimmed is None:
            continue
        obs_stripped = r.trimmed.rstrip('G')
        if not obs_stripped:
            continue
        obs_runs = extract_homopolymer_runs(obs_stripped)

        read_stats, n_ref_runs, margin = _find_best_oligo(
            obs_runs, r.result.predicted_constant, oligo_index)
        if read_stats is None:
            continue

        n_aligned += 1
        margins.append(margin)
        hp_dev_read = (np.mean(np.abs(read_stats.hp_length_diffs))
                       if read_stats.hp_length_diffs else 0.0)
        per_read_stats.append((
            margin,
            read_stats.substitutions,
            read_stats.deletions,
            hp_dev_read,
            read_stats.total_reference_bases,
        ))

        # track per-position coverage and errors from hp_length_diffs
        # the alignment produces one hp_length_diff per matched position,
        # in order of reference position
        for pos_i, ld in enumerate(read_stats.hp_length_diffs):
            pos_coverage[pos_i] += 1
            pos_hp_errors[pos_i].append(abs(ld))

        global_stats.insertions += read_stats.insertions
        global_stats.deletions += read_stats.deletions
        global_stats.substitutions += read_stats.substitutions
        global_stats.homopolymer_length_errors += read_stats.homopolymer_length_errors
        global_stats.total_aligned_bases += read_stats.total_aligned_bases
        global_stats.total_reference_bases += read_stats.total_reference_bases
        for base, count in read_stats.insertions_by_base.items():
            global_stats.insertions_by_base[base] += count
        for base, count in read_stats.deletions_by_base.items():
            global_stats.deletions_by_base[base] += count
        for pair, count in read_stats.substitutions_by_pair.items():
            global_stats.substitutions_by_pair[pair] += count
        global_stats.hp_length_diffs.extend(read_stats.hp_length_diffs)
        for base, pairs in read_stats.hp_length_by_base.items():
            global_stats.hp_length_by_base[base].extend(pairs)
        for ref_len, diffs in read_stats.hp_errors_by_ref_length.items():
            global_stats.hp_errors_by_ref_length[ref_len].extend(diffs)

    diagnostics = {
        'margins': margins,
        'pos_hp_errors': dict(pos_hp_errors),
        'pos_coverage': dict(pos_coverage),
        'per_read': per_read_stats,
    }
    return global_stats, n_aligned, diagnostics


def _error_rates(stats):
    """compute sub/del rates and mean HP deviation from alignment stats."""
    total = stats.total_reference_bases
    hp_dev = (np.mean(np.abs(stats.hp_length_diffs))
              if stats.hp_length_diffs else 0.0)
    if total == 0:
        return {'sub': 0.0, 'ins': 0.0, 'del': 0.0, 'hp_dev': hp_dev}
    return {
        'sub': stats.substitutions / total,
        'ins': stats.insertions / total,
        'del': stats.deletions / total,
        'hp_dev': hp_dev,
    }


def run_analysis(loader):
    out_dir = FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("    loading oligo pool...", flush=True)
    oligo_pool = _load_oligo_pool()
    oligo_index = _build_oligo_index(oligo_pool)
    n_oligos = sum(len(v) for v in oligo_pool.values())
    print(f"    {n_oligos} oligos across {len(oligo_pool)} constants", flush=True)

    rows = []
    positional_errors = {}
    all_diagnostics = {}

    for barcode in BARCODE_CONDITIONS:
        all_reads = []
        for run in RUNS:
            data = loader.load(run, barcode)
            all_reads.extend(data.reads)

        print(f"    {barcode}: aligning {len(all_reads)} reads against "
              f"oligo pool...", flush=True)
        stats, n_aligned, diag = _compute_error_stats(all_reads, oligo_index)
        rates = _error_rates(stats)
        all_diagnostics[barcode] = diag
        rows.append({
            'barcode': barcode,
            'label': label(barcode),
            'sub_rate': rates['sub'],
            'ins_rate': rates['ins'],
            'del_rate': rates['del'],
            'hp_dev': rates['hp_dev'],
            'total_ref_bases': stats.total_reference_bases,
            'n_aligned': n_aligned,
            'substitutions_by_pair': dict(stats.substitutions_by_pair),
        })
        print(f"      {n_aligned} aligned, sub={rates['sub']:.4f} "
              f"ins={rates['ins']:.4f} del={rates['del']:.4f} "
              f"hp_dev={rates['hp_dev']:.2f} bp",
              flush=True)

        # positional error profile (bit-level, still vs canonical)
        pos_errors = np.zeros(32)
        pos_total = np.zeros(32)
        for r in all_reads:
            if (r.result.passed_filter and r.result.bits
                    and len(r.result.bits) == 32
                    and r.result.predicted_constant
                    and r.result.predicted_constant in PATTERNS):
                gt_bits = decode_dna(PATTERNS[r.result.predicted_constant])
                for i in range(32):
                    if r.result.bits[i] >= 0:
                        pos_total[i] += 1
                        if r.result.bits[i] != gt_bits[i]:
                            pos_errors[i] += 1
        positional_errors[barcode] = (pos_errors, pos_total)

    # stacked bar (sub + del) and HP deviation: separate figure per group
    for gkey, gtitle, barcodes in GROUPS:
        group_rows = [r for r in rows if r['barcode'] in barcodes]
        labels_list = [r['label'] for r in group_rows]
        x = np.arange(len(labels_list))
        subs = [r['sub_rate'] for r in group_rows]
        dels = [r['del_rate'] for r in group_rows]

        # sub + del stacked bar
        fig, ax = plt.subplots(figsize=(4.2, 4.2 / PHI))
        ax.bar(x, subs, 0.6, label='Substitution', color=WONG[0])
        ax.bar(x, dels, 0.6, bottom=subs, label='Deletion', color=WONG[2])
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Error rate [per reference base]')
        ax.legend(frameon=False, fontsize=7)
        remove_spines(ax)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'error_rate_bar_{gkey}.pdf'))

        # mean HP run-length deviation
        hp_devs = [r['hp_dev'] for r in group_rows]
        fig, ax = plt.subplots(figsize=(4.2, 4.2 / PHI))
        ax.bar(x, hp_devs, 0.6, color=WONG[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean HP run-length deviation [bp]')
        remove_spines(ax)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'hp_deviation_{gkey}.pdf'))

    # heatmap: separate for each group
    for gkey, gtitle, barcodes in GROUPS:
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        barcode_labels = []
        matrix = []
        for barcode in barcodes:
            pos_err, pos_tot = positional_errors[barcode]
            rate = np.divide(pos_err, pos_tot, out=np.zeros_like(pos_err),
                             where=pos_tot > 0)
            matrix.append(rate)
            barcode_labels.append(label(barcode))
        if matrix:
            matrix = np.array(matrix)
            im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                           interpolation='nearest')
            ax.set_yticks(range(len(barcode_labels)))
            ax.set_yticklabels(barcode_labels, fontsize=7)
            ax.set_xlabel('Bit position')
            ax.set_ylabel('Condition')
            ax.set_title(gtitle, fontsize=9)
            plt.colorbar(im, ax=ax, label='Error rate')
        save_fig(fig, os.path.join(out_dir, f'error_by_position_{gkey}.pdf'))

    # substitution matrix: one per barcode, grouped by experiment
    bases = ['A', 'C', 'T']
    for gkey, gtitle, barcodes in GROUPS:
        n = len(barcodes)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3 / PHI))
        if n == 1:
            axes = [axes]
        for ai, barcode in enumerate(barcodes):
            ax = axes[ai]
            r = next((r for r in rows if r['barcode'] == barcode), None)
            mat = np.zeros((3, 3))
            if r:
                for pair, count in r['substitutions_by_pair'].items():
                    if '>' in pair:
                        ref_b, obs_b = pair.split('>')
                        ri = bases.index(ref_b) if ref_b in bases else -1
                        ci = bases.index(obs_b) if obs_b in bases else -1
                        if ri >= 0 and ci >= 0:
                            mat[ri, ci] = count
            if mat.sum() > 0:
                mat = mat / mat.sum()
            im = ax.imshow(mat, cmap='cividis')
            ax.set_xticks(range(3))
            ax.set_xticklabels(bases)
            ax.set_yticks(range(3))
            ax.set_yticklabels(bases)
            ax.set_xlabel('Observed base')
            if ai == 0:
                ax.set_ylabel('Reference base')
            ax.set_title(label(barcode), fontsize=7)
            plt.colorbar(im, ax=ax, label='Fraction')
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'substitution_matrix_{gkey}.pdf'))

    # diagnostic: per-position HP error rate + coverage (truncation bias)
    for gkey, gtitle, barcodes in GROUPS:
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5 / PHI),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)
        ax_err, ax_cov = axes
        for bi, barcode in enumerate(barcodes):
            diag = all_diagnostics[barcode]
            positions = sorted(diag['pos_coverage'].keys())
            if not positions:
                continue
            max_pos = max(positions)
            cov = np.zeros(max_pos + 1)
            mean_err = np.zeros(max_pos + 1)
            for p in range(max_pos + 1):
                cov[p] = diag['pos_coverage'].get(p, 0)
                errs = diag['pos_hp_errors'].get(p, [])
                if errs:
                    mean_err[p] = np.mean(errs)
            xs = np.arange(max_pos + 1)
            ax_err.plot(xs, mean_err, 'o-', color=WONG[bi], markersize=3,
                        lw=1, label=label(barcode), alpha=0.8)
            ax_cov.plot(xs, cov, '-', color=WONG[bi], lw=1, alpha=0.8)

        ax_err.set_ylabel('Mean |HP length error| [bp]')
        ax_err.legend(frameon=False, fontsize=6, ncol=2)
        remove_spines(ax_err)
        ax_cov.set_xlabel('Run position in variant')
        ax_cov.set_ylabel('Coverage [reads]')
        remove_spines(ax_cov)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'diag_position_error_{gkey}.pdf'))

    # diagnostic: score margin distribution + margin-filtered error comparison
    for gkey, gtitle, barcodes in GROUPS:
        fig, axes = plt.subplots(1, 2, figsize=(7, 7 / PHI))
        ax_hist, ax_comp = axes

        for bi, barcode in enumerate(barcodes):
            diag = all_diagnostics[barcode]
            margins = np.array(diag['margins'])
            if len(margins) == 0:
                continue
            # margin distribution (log scale for x, since margins can be large)
            bins = np.linspace(0, min(margins.max(), 100), 50)
            ax_hist.hist(margins, bins=bins, alpha=0.5, color=WONG[bi],
                         label=label(barcode), density=True)

        ax_hist.set_xlabel('Score margin (best - 2nd best)')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(frameon=False, fontsize=6)
        remove_spines(ax_hist)

        # margin-filtered HP deviation using cached per-read stats
        # (margin, subs, dels, hp_dev, ref_bases)
        margin_thresholds = [0, 5, 10, 20]
        x = np.arange(len(margin_thresholds))
        width = 0.8 / len(barcodes)
        for bi, barcode in enumerate(barcodes):
            diag = all_diagnostics[barcode]
            prs = diag['per_read']
            if not prs:
                continue
            arr = np.array(prs)  # shape (n_reads, 5)
            filtered_devs = []
            for thresh in margin_thresholds:
                mask = arr[:, 0] >= thresh
                if mask.sum() == 0:
                    filtered_devs.append(0.0)
                else:
                    filtered_devs.append(arr[mask, 3].mean())

            ax_comp.bar(x + bi * width, filtered_devs, width,
                        color=WONG[bi], label=label(barcode), alpha=0.8)

        ax_comp.set_xticks(x + width * (len(barcodes) - 1) / 2)
        ax_comp.set_xticklabels([f'>={t}' if t > 0 else 'All'
                                  for t in margin_thresholds], fontsize=7)
        ax_comp.set_xlabel('Score margin filter')
        ax_comp.set_ylabel('Mean HP deviation [bp]')
        ax_comp.legend(frameon=False, fontsize=6)
        remove_spines(ax_comp)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'diag_margin_filter_{gkey}.pdf'))

    # summary tsv
    tsv_path = os.path.join(out_dir, 'summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('barcode\tlabel\tsub_rate\tdel_rate\thp_dev_bp'
                '\ttotal_ref_bases\tn_aligned\n')
        for r in rows:
            f.write(f"{r['barcode']}\t{r['label']}\t"
                    f"{r['sub_rate']:.6f}\t{r['del_rate']:.6f}\t"
                    f"{r['hp_dev']:.4f}\t{r['total_ref_bases']}"
                    f"\t{r['n_aligned']}\n")
    print(f"    saved summary.tsv")

    return {'rows': rows, 'diagnostics': all_diagnostics}


def main():
    parser = argparse.ArgumentParser(description='section 2.04 - error rates')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.04 - error rates")
    loader = DataLoader(quick=args.quick, quick_n=args.quick_n)
    loader.load_all()
    run_analysis(loader)
    copy_script(__file__, FIGURE_DIR)
    print("done.")


if __name__ == '__main__':
    main()
