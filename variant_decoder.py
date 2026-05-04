#!/usr/bin/env python3
"""
variant-level HMM alignment.

scores each read against all ~118 synthesis variants of one constant and
writes a tsv with (designed_hp_length, observed_hp_length) at each of the
32 coding positions per read.

one process handles one constant. the collector script picks the best
constant per read by argmax across the 17 per-constant tsvs.

usage:
    python3 variant_decoder.py <constant_name> <reads.fastq[.gz] or dir/> \
        --fasta babbage.fasta -o output.tsv
"""

import argparse
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from babbage_sequence_list import (
    CONSTANTS, DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER,
)
from error_analysis import extract_homopolymer_runs
from profile_hmm import (
    ProfileHMM, ProfileHMMParameters, StateType, BASE_TO_IDX,
    VariantLibrary,
)


def parse_fasta_references(fasta_path):
    """parse babbage.fasta -> dict of constant_name -> list of (sequence, hp_lengths).

    strips primers and the g-tail, then runs the rle to get the 32-element
    hp-length vector per variant. caller uses both the trimmed core seq
    and the lengths to build per-variant HMMs.
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
            variants.append((core, lengths))
        ref_variants[const_name] = variants

    return ref_variants


def build_variant_library(constant_name, pattern, variants, params=None):
    """one VariantLibrary with the per-variant self-loops baked in.

    take the template HMM's transition matrix, override each match-state
    self-loop to match the variant's designed hp length at that position,
    rescale the non-self-loop outgoing mass proportionally so the row still
    sums to 1, and add the result to the library.
    """
    params = params or ProfileHMMParameters()
    template = ProfileHMM(pattern, params)
    n_states = template.n_states
    pattern_length = len(pattern)

    # position -> match-state index
    match_state_indices = {}
    for state, idx in template.state_to_idx.items():
        if state[0] == StateType.MATCH:
            match_state_indices[state[1]] = idx

    library = VariantLibrary(pattern_length)

    for _core, hp_lengths in variants:
        log_trans = template.log_trans.copy()

        for pos_i in range(pattern_length):
            m_idx = match_state_indices[pos_i]
            designed_len = float(hp_lengths[pos_i]) if pos_i < len(hp_lengths) else 1.0

            # geometric self-loop matched to designed length: p = 1 - 1/L.
            # clip at 0.01 so a length-1 variant still has SOME outgoing mass.
            p_extend_new = max(0.01, 1.0 - 1.0 / max(1.0, designed_len))

            old_p_extend = math.exp(log_trans[m_idx, m_idx])
            old_non_self = 1.0 - old_p_extend
            new_non_self = 1.0 - p_extend_new

            log_trans[m_idx, m_idx] = math.log(p_extend_new)

            if old_non_self > 1e-12:
                scale = new_non_self / old_non_self
                for j in range(n_states):
                    if j != m_idx and log_trans[m_idx, j] > -np.inf:
                        old_p = math.exp(log_trans[m_idx, j])
                        log_trans[m_idx, j] = math.log(old_p * scale)

        designed = hp_lengths.tolist()
        while len(designed) < pattern_length:
            designed.append(1)
        designed = designed[:pattern_length]

        library.add_variant(
            log_trans, template.log_emit, template.states,
            template.start_idx, template.end_idx, designed,
        )

    return library


def iter_fastq(path):
    """yield (read_id, sequence) from a fastq file or directory of fastqs."""
    p = Path(path)
    files = sorted(p.glob('*.fastq*')) if p.is_dir() else [p]

    for fq in files:
        open_func = gzip.open if fq.suffix == '.gz' else open
        mode = 'rt' if fq.suffix == '.gz' else 'r'
        with open_func(fq, mode) as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break
                sequence = f.readline().strip()
                f.readline()  # +
                f.readline()  # quality line, ignored
                if not header or not sequence:
                    break
                read_id = header[1:].split()[0] if header.startswith('@') else header
                yield read_id, sequence


def seq_to_idx(sequence):
    """DNA string -> int32 index array. drops anything outside ACGT."""
    valid = set('ACGT')
    filtered = ''.join(b for b in sequence.upper() if b in valid)
    if not filtered:
        return None
    return np.array([BASE_TO_IDX[b] for b in filtered], dtype=np.int32)


def run_variant_alignment(constant_name, input_path, fasta_path, output_path,
                          batch_size=500):
    """score every read against one constant's variants, write the tsv."""
    print(f"loading fasta from {fasta_path}...", file=sys.stderr)
    ref_variants = parse_fasta_references(fasta_path)

    if constant_name not in ref_variants:
        print(f"error: '{constant_name}' not in {fasta_path}", file=sys.stderr)
        print(f"available: {sorted(ref_variants.keys())}", file=sys.stderr)
        sys.exit(1)

    variants = ref_variants[constant_name]
    print(f"  {constant_name}: {len(variants)} variants", file=sys.stderr)

    pattern = dict(CONSTANTS).get(constant_name)
    if pattern is None:
        print(f"error: pattern for '{constant_name}' not in CONSTANTS", file=sys.stderr)
        sys.exit(1)

    print(f"building variant library ({len(variants)} variants, "
          f"{len(pattern)} positions)...", file=sys.stderr)
    library = build_variant_library(constant_name, pattern, variants)
    print(f"  library has {library.len()} variants", file=sys.stderr)

    pattern_length = len(pattern)
    header_cols = ['read_id', 'best_variant_idx', 'viterbi_score']
    for i in range(pattern_length):
        header_cols.extend([f'designed_{i}', f'observed_{i}'])

    n_reads = 0
    n_written = 0
    batch_seqs = []
    batch_ids = []

    with open(output_path, 'w') as out:
        out.write('\t'.join(header_cols) + '\n')

        for read_id, sequence in iter_fastq(input_path):
            n_reads += 1
            idx = seq_to_idx(sequence)
            if idx is None or len(idx) < 10:
                continue

            batch_seqs.append(idx)
            batch_ids.append(read_id)

            if len(batch_seqs) >= batch_size:
                _flush_batch(library, batch_seqs, batch_ids, out)
                n_written += len(batch_seqs)
                batch_seqs = []
                batch_ids = []
                if n_reads % 5000 == 0:
                    print(f"  {n_reads} reads, {n_written} aligned...",
                          file=sys.stderr)

        if batch_seqs:
            _flush_batch(library, batch_seqs, batch_ids, out)
            n_written += len(batch_seqs)

    print(f"done. {n_reads} reads, {n_written} aligned.", file=sys.stderr)
    print(f"output: {output_path}", file=sys.stderr)


def _flush_batch(library, batch_seqs, batch_ids, out):
    """write one batch of alignments to the open tsv handle."""
    results = library.align_batch(batch_seqs, batch_ids)
    for rid, vidx, score, hp_flat in results:
        row = [rid, str(vidx), f'{score:.2f}']
        for k in range(0, len(hp_flat), 2):
            row.append(str(hp_flat[k]))
            row.append(str(hp_flat[k + 1]))
        out.write('\t'.join(row) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='variant-level HMM alignment for babbage reads')
    parser.add_argument('constant_name', help='name of constant (e.g., pi)')
    parser.add_argument('input', help='fastq file or directory of fastqs')
    parser.add_argument('--fasta', default='data/babbage.fasta',
                        help='babbage fasta reference')
    parser.add_argument('-o', '--output', default=None,
                        help='output tsv (default: <constant>.variant_alignment.tsv)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='alignment batch size')
    args = parser.parse_args()

    output = args.output or f'{args.constant_name}.variant_alignment.tsv'
    run_variant_alignment(
        args.constant_name, args.input, args.fasta, output,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
