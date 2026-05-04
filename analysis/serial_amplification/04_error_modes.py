#!/usr/bin/env python3
"""
section 2.11 - analysis 04 - error mode characterization across generations.

per-generation substitution, insertion, and deletion rates on target-length
reads. per-position error comparison between early and late generations.
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
    PATTERNS, extract_homopolymer_runs,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR

# early vs late generation groups for position analysis
EARLY_GENS = [1, 2, 3]
LATE_GENS = [10, 11, 12]


def _align_runs(obs_runs, ref_runs):
    """greedy run-level alignment with 1-step lookahead.

    returns list of (event_type, ref_position, detail):
      - ('match', ri, length_diff)   - correct base, hp length deviation
      - ('substitution', ri, (ref_base, obs_base))
      - ('insertion', ri, obs_base)   - extra observed run near ref pos ri
      - ('deletion', ri, ref_base)    - missing reference run at pos ri
    """
    events = []
    oi, ri = 0, 0
    while oi < len(obs_runs) and ri < len(ref_runs):
        ob, ol, _ = obs_runs[oi]
        rb, rl, _ = ref_runs[ri]
        if ob == rb:
            # match - record HP length difference
            events.append(('match', ri, ol - rl))
            oi += 1
            ri += 1
        else:
            # lookahead: would skipping obs or ref realign?
            skip_obs = (oi + 1 < len(obs_runs)
                        and obs_runs[oi + 1][0] == rb)
            skip_ref = (ri + 1 < len(ref_runs)
                        and ref_runs[ri + 1][0] == ob)
            if skip_obs and not skip_ref:
                # extra observed run = insertion
                events.append(('insertion', ri, ob))
                oi += 1
            elif skip_ref and not skip_obs:
                # missing reference run = deletion
                events.append(('deletion', ri, rb))
                ri += 1
            elif skip_obs and skip_ref:
                # ambiguous - prefer insertion (skip obs), preserves ref alignment
                events.append(('insertion', ri, ob))
                oi += 1
            else:
                # neither lookahead matches - substitution
                events.append(('substitution', ri, (rb, ob)))
                oi += 1
                ri += 1
    # remaining observed = insertions
    tail_pos = max(ri - 1, 0)
    while oi < len(obs_runs):
        events.append(('insertion', tail_pos, obs_runs[oi][0]))
        oi += 1
    # remaining reference = deletions
    while ri < len(ref_runs):
        events.append(('deletion', ri, ref_runs[ri][0]))
        ri += 1
    return events


def _compute_gen_errors(decoded_reads, patterns):
    """compute run-level error stats for decoded target-length reads.

    counts run-level events (1 per event, not per base) and normalizes
    by total reference run positions (32 × n_aligned_reads).
    """
    n_sub = 0
    n_ins = 0
    n_del = 0
    n_match = 0
    n_hp_err = 0  # matched positions with nonzero length deviation
    n_ref_positions = 0
    n_aligned_reads = 0

    # per-position: run_position -> list of abs(length_diff) for matches
    position_errors = defaultdict(list)

    for d in decoded_reads:
        if not d.result.passed_filter or not d.result.predicted_constant:
            continue
        if d.trimmed is None:
            continue
        ref_pattern = patterns.get(d.result.predicted_constant)
        if ref_pattern is None:
            continue
        obs_stripped = d.trimmed.rstrip('G')
        if not obs_stripped:
            continue
        obs_runs = extract_homopolymer_runs(obs_stripped)
        ref_runs = extract_homopolymer_runs(ref_pattern)
        events = _align_runs(obs_runs, ref_runs)

        n_aligned_reads += 1
        n_ref_positions += len(ref_runs)

        for ev_type, ref_pos, detail in events:
            if ev_type == 'match':
                n_match += 1
                length_diff = detail
                if length_diff != 0:
                    n_hp_err += 1
                position_errors[ref_pos].append(abs(length_diff))
            elif ev_type == 'substitution':
                n_sub += 1
            elif ev_type == 'insertion':
                n_ins += 1
            elif ev_type == 'deletion':
                n_del += 1
                # record deletion as a large error at that position
                position_errors[ref_pos].append(float('nan'))

    total = max(n_ref_positions, 1)
    n_match_safe = max(n_match, 1)
    return {
        'sub_rate': n_sub / total,
        'ins_rate': n_ins / total,
        'del_rate': n_del / total,
        'hp_err_rate': n_hp_err / n_match_safe,
        'n_ref_positions': n_ref_positions,
        'n_aligned_reads': n_aligned_reads,
        'n_match': n_match,
        'position_errors': dict(position_errors),
    }


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=True, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))

    # compute per-generation error rates
    gen_errors = {}
    for g in gens:
        target = [
            d for d in data[g].decoded_reads
            if TARGET_MIN <= d.raw.length <= TARGET_MAX
        ]
        gen_errors[g] = _compute_gen_errors(target, PATTERNS)

    # error rates vs generation
    sub_rates = [gen_errors[g]['sub_rate'] for g in gens]
    ins_rates = [gen_errors[g]['ins_rate'] for g in gens]
    del_rates = [gen_errors[g]['del_rate'] for g in gens]

    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    ax.plot(gens, sub_rates, 'o-', color=WONG[0], lw=2, markersize=5,
            label='Substitution')
    ax.plot(gens, ins_rates, 's-', color=WONG[1], lw=2, markersize=5,
            label='Insertion')
    ax.plot(gens, del_rates, '^-', color=WONG[2], lw=2, markersize=5,
            label='Deletion')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Run-level error rate')
    ax.set_xticks(gens)
    ax.legend(frameon=False)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'error_rates_vs_generation.pdf'))

    # two-panel stacked figure: insertion rate (top), match/sub/del fractions (bottom)
    x = np.arange(1, N_GENERATIONS + 1)
    width = 0.7
    match_rates = [1.0 - s - d for s, d in zip(sub_rates, del_rates)]

    fig, (ax_ins, ax_frac) = plt.subplots(
        2, 1, figsize=(7, 7 / PHI * 1.4),
        gridspec_kw={'height_ratios': [1, 1]},
        sharex=True,
    )

    # top: insertion rate line plot
    ax_ins.plot(x, ins_rates, 's-', color=WONG[1], lw=2, markersize=5)
    ax_ins.set_ylabel('Insertion rate\n[runs per ref position]')
    ax_ins.set_ylim(0, 3.2)
    remove_spines(ax_ins)

    # bottom: stacked bar (match + sub + del = 1.0)
    ax_frac.bar(x, match_rates, width, color=WONG[5], alpha=0.7, label='Match')
    ax_frac.bar(x, sub_rates, width, bottom=match_rates, color=WONG[0],
                alpha=0.7, label='Substitution')
    bottom_del = [m + s for m, s in zip(match_rates, sub_rates)]
    ax_frac.bar(x, del_rates, width, bottom=bottom_del, color=WONG[2],
                alpha=0.7, label='Deletion')
    ax_frac.set_xlabel('Generation')
    ax_frac.set_ylabel('Fraction of ref positions')
    ax_frac.set_xticks(x)
    ax_frac.set_ylim(0, 1.05)
    ax_frac.legend(frameon=False, fontsize=8)
    remove_spines(ax_frac)

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'error_rates_stacked.pdf'))

    # per-position error comparison: early vs late
    # aggregate position errors for early and late groups
    early_pos_errors = defaultdict(list)
    late_pos_errors = defaultdict(list)

    for g in EARLY_GENS:
        if g in gen_errors:
            for pos, errs in gen_errors[g]['position_errors'].items():
                early_pos_errors[pos].extend(errs)
    for g in LATE_GENS:
        if g in gen_errors:
            for pos, errs in gen_errors[g]['position_errors'].items():
                late_pos_errors[pos].extend(errs)

    # compute mean error per position
    all_positions = sorted(set(early_pos_errors.keys()) |
                           set(late_pos_errors.keys()))
    hotspots = []
    if all_positions:
        max_pos = min(max(all_positions) + 1, 40)
        positions = list(range(max_pos))
        early_means = []
        late_means = []
        for p in positions:
            e_errs = early_pos_errors.get(p, [0])
            l_errs = late_pos_errors.get(p, [0])
            # filter NaN (deletions) - only average HP length errors from matches
            e_valid = [v for v in e_errs if not np.isnan(v)]
            l_valid = [v for v in l_errs if not np.isnan(v)]
            early_means.append(np.mean(e_valid) if e_valid else 0)
            late_means.append(np.mean(l_valid) if l_valid else 0)

        fig, ax = plt.subplots(figsize=(8, 8 / PHI))
        width = 0.35
        x = np.arange(len(positions))
        ax.bar(x - width / 2, early_means, width, color=WONG[0], alpha=0.8,
               label=f'Early (gen {EARLY_GENS[0]}-{EARLY_GENS[-1]})')
        ax.bar(x + width / 2, late_means, width, color=WONG[1], alpha=0.8,
               label=f'Late (gen {LATE_GENS[0]}-{LATE_GENS[-1]})')
        ax.set_xlabel('Run position in reference')
        ax.set_ylabel('Mean HP length error [bp]')
        ax.set_xticks(x[::2])
        ax.set_xticklabels(positions[::2], fontsize=7)
        ax.legend(frameon=False)
        remove_spines(ax)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIGURE_DIR, 'positional_errors.pdf'))

        # identify hotspot positions (late > early by > 50% relative)
        for i, p in enumerate(positions):
            e_val = early_means[i]
            l_val = late_means[i]
            if e_val > 0 and l_val > e_val * 1.5:
                hotspots.append((p, e_val, l_val, (l_val - e_val) / e_val))

    # summary tsv
    tsv_path = os.path.join(FIGURE_DIR, '04_error_modes.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\tsub_rate\tins_rate\tdel_rate\thp_err_rate'
                '\tn_ref_positions\tn_match\tn_aligned_reads\n')
        for g in gens:
            e = gen_errors[g]
            f.write(f"{g}\t{e['sub_rate']:.6f}\t{e['ins_rate']:.6f}"
                    f"\t{e['del_rate']:.6f}\t{e['hp_err_rate']:.6f}"
                    f"\t{e['n_ref_positions']}\t{e['n_match']}"
                    f"\t{e['n_aligned_reads']}\n")
        if hotspots:
            f.write('\n# hotspot positions (late error > 1.5x early)\n')
            f.write('# position\tearly_mean\tlate_mean\trelative_increase\n')
            for pos, e, l, rel in hotspots:
                f.write(f"# {pos}\t{e:.4f}\t{l:.4f}\t{rel:.2f}\n")
    print("    saved 04_error_modes.tsv", flush=True)

    return {'gen_errors': gen_errors}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='analysis 04 - error modes')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 04 - error modes", flush=True)
    data = load_all_generations(decode=True, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
