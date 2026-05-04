#!/usr/bin/env python3
"""
section 2.11 - analysis 05 - chimera characterization.

for reads outside the 200-300 bp target window, characterize chimera
structure by searching for internal primer sequences and aligning
fragments to known codewords.
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
    EXPECTED_LENGTH, PATTERNS, FWD_PRIMER, REV_PRIMER, RC_FWD, RC_REV,
)
from shared import remove_spines, save_fig, add_quick_arg, PHI, WONG, plt

FIGURE_DIR = SCRIPT_DIR

# primer core subsequences for fast pre-screening (12bp from center)
# then verify candidates with full fuzzy match
_CORE_LEN = 12
PRIMER_QUERIES = {}
PRIMER_CORES = {}
for _pname, _pseq in [('FWD', FWD_PRIMER), ('REV', REV_PRIMER),
                       ('RC_FWD', RC_FWD), ('RC_REV', RC_REV)]:
    PRIMER_QUERIES[_pname] = _pseq
    _mid = len(_pseq) // 2
    PRIMER_CORES[_pname] = _pseq[_mid - _CORE_LEN // 2:_mid + _CORE_LEN // 2]


def _find_internal_primers(seq, margin=35, max_mm=5):
    """find primer occurrences in the interior of a read.
    uses fast str.find for core screening, then fuzzy verification."""
    hits = []
    seq = seq.upper()
    for name, primer in PRIMER_QUERIES.items():
        plen = len(primer)
        if len(seq) < plen + 2 * margin:
            continue
        core = PRIMER_CORES[name]
        core_offset = len(primer) // 2 - _CORE_LEN // 2
        # fast scan for core matches
        start = margin
        end = len(seq) - plen - margin + 1
        pos = start
        while pos < end:
            idx = seq.find(core, pos + core_offset, end + core_offset)
            if idx == -1:
                break
            # candidate position for full primer
            cand = idx - core_offset
            if margin <= cand < end:
                mm = sum(1 for a, b in zip(seq[cand:cand + plen], primer)
                         if a != b)
                if mm <= max_mm:
                    hits.append((name, cand, mm))
            pos = idx - core_offset + 1
    return hits


def _classify_chimera(read_length, internal_primers):
    """classify chimera type based on length and internal primer hits."""
    if read_length < 100:
        return 'primer_dimer'
    elif read_length < TARGET_MIN:
        return 'truncated_fragment'
    elif read_length > TARGET_MAX:
        if internal_primers:
            return 'chimera_with_junction'
        elif read_length < 400:
            return 'extended_fragment'
        elif read_length < 600:
            return 'possible_dimer'
        else:
            return 'long_chimera'
    return 'other'


def run_analysis(data=None, quick=False, quick_n=500):
    if data is None:
        data = load_all_generations(decode=False, quick=quick, quick_n=quick_n)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    gens = list(range(1, N_GENERATIONS + 1))

    # collect non-target reads across all generations
    chimera_classes = defaultdict(lambda: defaultdict(int))  # gen -> class -> count
    all_chimera_lengths = defaultdict(list)  # gen -> [lengths]
    junction_examples = []  # (gen, length, n_junctions, classification)

    for g in gens:
        for r in data[g].raw_reads:
            if TARGET_MIN <= r.length <= TARGET_MAX:
                continue
            # non-target read
            all_chimera_lengths[g].append(r.length)
            internal = _find_internal_primers(r.sequence)
            classification = _classify_chimera(r.length, internal)
            chimera_classes[g][classification] += 1

            if internal and len(junction_examples) < 50:
                # classify junction orientation (e.g. FWD-FWD, FWD-REV)
                orientations = [h[0] for h in internal]
                junction_examples.append({
                    'gen': g,
                    'read_id': r.read_id,
                    'length': r.length,
                    'n_junctions': len(internal),
                    'junction_positions': [(h[0], h[1]) for h in internal],
                    'orientations': orientations,
                    'classification': classification,
                })

    # chimera length distribution (pooled)
    all_lengths_pooled = []
    for g in gens:
        all_lengths_pooled.extend(all_chimera_lengths[g])

    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    if all_lengths_pooled:
        bins = np.arange(0, max(all_lengths_pooled) + 50, 25)
        ax.hist(all_lengths_pooled, bins=bins, color=WONG[0], alpha=0.8,
                edgecolor='none')
        # mark multiples of expected length
        for mult in range(1, 5):
            x_line = mult * EXPECTED_LENGTH
            if x_line < max(all_lengths_pooled):
                ax.axvline(x_line, color=WONG[1], ls='--', lw=1, alpha=0.6)
                ax.text(x_line + 5, ax.get_ylim()[1] * 0.9,
                        f'{mult}x{EXPECTED_LENGTH}', fontsize=7, color=WONG[1],
                        rotation=90, va='top')
    ax.set_xlabel('Read length [bp]')
    ax.set_ylabel('Count')
    ax.set_title('Non-target read length distribution')
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'chimera_length_distribution.pdf'))

    # chimera classification breakdown per generation
    all_classes = sorted(set(
        cls for g in gens for cls in chimera_classes[g]
    ))
    class_colors = {cls: WONG[i % len(WONG)] for i, cls in enumerate(all_classes)}

    # stacked bar - raw counts
    fig, ax = plt.subplots(figsize=(8, 8 / PHI))
    bottom = np.zeros(len(gens))
    for cls in all_classes:
        vals = [chimera_classes[g].get(cls, 0) for g in gens]
        label_text = cls.replace('_', ' ')
        ax.bar(gens, vals, bottom=bottom, color=class_colors[cls], alpha=0.8,
               label=label_text)
        bottom += np.array(vals)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Non-target read count')
    ax.set_xticks(gens)
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'chimera_classification.pdf'))

    # stacked bar - fractions of total reads per generation
    fig, ax = plt.subplots(figsize=(8, 8 / PHI))
    bottom = np.zeros(len(gens))
    for cls in all_classes:
        vals = []
        for g in gens:
            n_total = len(data[g].raw_reads)
            cnt = chimera_classes[g].get(cls, 0)
            vals.append(cnt / n_total if n_total > 0 else 0)
        label_text = cls.replace('_', ' ')
        ax.bar(gens, vals, bottom=bottom, color=class_colors[cls], alpha=0.8,
               label=label_text)
        bottom += np.array(vals)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fraction of total reads')
    ax.set_xticks(gens)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'chimera_classification_fraction.pdf'))

    # chimera fraction vs generation with wilson 95% CI
    fig, ax = plt.subplots(figsize=(7, 7 / PHI))
    chimera_fracs = []
    ci_lo = []
    ci_hi = []
    z = 1.96  # 95% CI
    for g in gens:
        n_total = len(data[g].raw_reads)
        n_chimera = len(all_chimera_lengths[g])
        if n_total > 0:
            p_hat = n_chimera / n_total
            # wilson score interval
            denom = 1 + z ** 2 / n_total
            centre = (p_hat + z ** 2 / (2 * n_total)) / denom
            spread = z * np.sqrt(
                (p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total
            ) / denom
            ci_lo.append(max(0, centre - spread))
            ci_hi.append(min(1, centre + spread))
        else:
            p_hat = 0
            ci_lo.append(0)
            ci_hi.append(0)
        chimera_fracs.append(p_hat)
    fracs_arr = np.array(chimera_fracs)
    yerr = np.array([fracs_arr - ci_lo, np.array(ci_hi) - fracs_arr])
    ax.errorbar(gens, chimera_fracs, yerr=yerr, fmt='o-', color=WONG[0],
                lw=2, markersize=6, capsize=3, capthick=1)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Non-target read fraction')
    ax.set_xticks(gens)
    ax.set_ylim(0, 1.05)
    remove_spines(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGURE_DIR, 'chimera_fraction.pdf'))

    # example chimeras text output
    examples_path = os.path.join(FIGURE_DIR, 'chimera_examples.txt')
    with open(examples_path, 'w') as f:
        f.write('# example chimeric reads with internal primer junctions\n')
        f.write(f'# total examples: {len(junction_examples)}\n\n')
        for ex in junction_examples[:20]:
            f.write(f"read: {ex['read_id']}\n")
            f.write(f"  generation: {ex['gen']}\n")
            f.write(f"  length: {ex['length']} bp\n")
            f.write(f"  classification: {ex['classification']}\n")
            f.write(f"  internal junctions: {ex['n_junctions']}\n")
            f.write(f"  orientations: {', '.join(ex['orientations'])}\n")
            for pname, pos in ex['junction_positions']:
                f.write(f"    {pname} at position {pos}\n")
            f.write('\n')
    print("    saved chimera_examples.txt", flush=True)

    # summary tsv
    tsv_path = os.path.join(FIGURE_DIR, '05_chimera_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('generation\ttotal_reads\tnon_target_reads\tnon_target_fraction')
        for cls in all_classes:
            f.write(f'\t{cls}')
        f.write('\n')
        for g in gens:
            n_total = len(data[g].raw_reads)
            n_nontarget = len(all_chimera_lengths[g])
            frac = n_nontarget / n_total if n_total > 0 else 0
            f.write(f"{g}\t{n_total}\t{n_nontarget}\t{frac:.4f}")
            for cls in all_classes:
                f.write(f"\t{chimera_classes[g].get(cls, 0)}")
            f.write('\n')
    print("    saved 05_chimera_summary.tsv", flush=True)

    return {
        'chimera_classes': dict(chimera_classes),
        'chimera_fracs': chimera_fracs,
        'junction_examples': junction_examples,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='analysis 05 - chimera characterization')
    add_quick_arg(parser)
    args = parser.parse_args()

    print("section 2.11 - analysis 05 - chimera characterization", flush=True)
    data = load_all_generations(decode=False, quick=args.quick, quick_n=args.quick_n)
    run_analysis(data)
    # script already in output dir
    print("done.", flush=True)


if __name__ == '__main__':
    main()
