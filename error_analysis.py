#!/usr/bin/env python3
"""
error analysis: real reads vs ground-truth reference.

walks each read against the expected sequence and tallies the four kinds
of errors we care about - insertions, deletions, substitutions, and hp
length errors. the last one is the interesting one for tdt; the rest
look like generic nanopore noise.

run-level alignment treats the reference as a sequence of homopolymer
runs and decides per-run whether each was extended, truncated, or
shifted. that's the raw input to the channel matrix and per-base hp
parameter fits in profile_hmm.
"""

import argparse
import gzip
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tsv_records import write_record

# Import from existing modules
from babbage_sequence_list import CONSTANTS, DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER
from nanopore_pipeline import reverse_complement, trim_primers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# data classes


@dataclass
class HomopolymerStats:
    """per-base hp run-length stats. expected vs observed lists, plus a
    base -> expected_len -> [observed_len, ...] table for the channel matrix."""
    expected_lengths: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    observed_lengths: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    length_errors: Dict[str, Dict[int, List[int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )


@dataclass
class SubstitutionStats:
    """4x4-ish base confusion matrix. ignores hp expansions/truncations."""
    matrix: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total_bases: int = 0
    total_errors: int = 0


@dataclass
class PositionalStats:
    """error rate as a function of position in the read."""
    error_counts: List[int] = field(default_factory=list)
    total_counts: List[int] = field(default_factory=list)
    quality_by_position: List[List[float]] = field(default_factory=list)


@dataclass
class QualityCorrelation:
    """error rate binned by mean read phred."""
    quality_bins: List[float] = field(default_factory=list)
    error_rates: List[float] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)


@dataclass
class StrandStats:
    """forward vs reverse-complement reads. pcr slippage should be ~symmetric."""
    forward_errors: int = 0
    forward_total: int = 0
    reverse_errors: int = 0
    reverse_total: int = 0


@dataclass
class AlignmentErrorStats:
    """alignment-based per-error tallies. the workhorse output of this module."""
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    homopolymer_length_errors: int = 0  # correct base, wrong run length
    total_aligned_bases: int = 0
    total_reference_bases: int = 0

    # Per-base breakdown
    insertions_by_base: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    deletions_by_base: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    substitutions_by_pair: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # "A>T" format

    # Homopolymer-specific
    hp_length_diffs: List[int] = field(default_factory=list)  # observed - expected
    hp_length_by_base: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )  # base -> [(expected, observed), ...]

    # pcr slippage analysis: track errors by reference hp length
    hp_errors_by_ref_length: Dict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list)
    )  # ref_length -> list of length_diffs


@dataclass
class PCRAnalysisStats:
    """statistics for pcr vs sequencing error discrimination."""
    # strand symmetry (pcr errors should be symmetric)
    forward_hp_diffs: List[int] = field(default_factory=list)
    reverse_hp_diffs: List[int] = field(default_factory=list)

    # ±1 bp bias (pcr slippage typically causes single-base indels)
    plus_minus_one_count: int = 0
    larger_deviation_count: int = 0

    # length-dependent slippage rate
    error_rate_by_hp_length: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: (0, 0))
    )  # hp_length -> (errors, total)


@dataclass
class ReferenceSequence:
    """one entry from babbage.fasta. core_sequence has primers removed."""
    name: str
    constant: str
    sequence: str
    core_sequence: str


@dataclass
class ErrorAnalysisResults:
    """everything we collected. one of these per fastq."""
    homopolymer_stats: HomopolymerStats = field(default_factory=HomopolymerStats)
    substitution_stats: SubstitutionStats = field(default_factory=SubstitutionStats)
    positional_stats: PositionalStats = field(default_factory=PositionalStats)
    quality_correlation: QualityCorrelation = field(default_factory=QualityCorrelation)
    strand_stats: StrandStats = field(default_factory=StrandStats)

    # alignment-based error stats
    alignment_stats: AlignmentErrorStats = field(default_factory=AlignmentErrorStats)

    # pcr slippage analysis
    pcr_stats: PCRAnalysisStats = field(default_factory=PCRAnalysisStats)

    # Summary counts
    total_reads: int = 0
    analyzed_reads: int = 0
    pattern_matches: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


# reference sequence parsing

def parse_fasta(filepath: Path) -> List[ReferenceSequence]:
    """parse babbage.fasta. expects headers like
    `>pool1_pi_0 length=244 constant=pi`."""
    references = []
    current_name = None
    current_constant = None
    current_seq_parts = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_name is not None:
                    seq = ''.join(current_seq_parts)
                    # Extract core (without primers)
                    core = extract_core_sequence(seq)
                    references.append(ReferenceSequence(
                        name=current_name,
                        constant=current_constant,
                        sequence=seq,
                        core_sequence=core
                    ))

                # Parse header
                parts = line[1:].split()
                current_name = parts[0]
                current_constant = "unknown"
                for part in parts:
                    if part.startswith("constant="):
                        current_constant = part.split("=")[1]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Don't forget the last sequence
        if current_name is not None:
            seq = ''.join(current_seq_parts)
            core = extract_core_sequence(seq)
            references.append(ReferenceSequence(
                name=current_name,
                constant=current_constant,
                sequence=seq,
                core_sequence=core
            ))

    return references


def extract_core_sequence(sequence: str) -> str:
    """trim the fwd/rev primers if present. doesn't try to handle reverse-
    complement orientation; trim_primers() does that."""
    seq_upper = sequence.upper()
    fwd = DEFAULT_FORWARD_PRIMER.upper()
    rev = DEFAULT_REVERSE_PRIMER.upper()

    core = seq_upper
    if core.startswith(fwd):
        core = core[len(fwd):]
    if core.endswith(rev):
        core = core[:-len(rev)]

    return core


def build_reference_index(references: List[ReferenceSequence]) -> Dict[str, List[ReferenceSequence]]:
    """group references by constant name."""
    index = defaultdict(list)
    for ref in references:
        index[ref.constant].append(ref)
    return dict(index)


def find_best_reference(
    observed_core: str,
    constant_name: str,
    reference_index: Dict[str, List[ReferenceSequence]]
) -> Optional[ReferenceSequence]:
    """given a constant and its primer-trimmed read, find the closest variant
    in the reference. compares collapsed (homopolymer-free) sequences so we
    don't penalize hp expansion when picking the variant."""
    if constant_name not in reference_index:
        return None

    candidates = reference_index[constant_name]
    if not candidates:
        return None

    # Collapse homopolymers for comparison
    obs_collapsed = collapse_homopolymers_simple(observed_core.rstrip('G'))

    best_ref = None
    best_score = 0

    for ref in candidates:
        ref_collapsed = collapse_homopolymers_simple(ref.core_sequence.rstrip('G'))

        # Simple matching score
        score = sum(1 for a, b in zip(obs_collapsed, ref_collapsed) if a == b)
        if score > best_score:
            best_score = score
            best_ref = ref

    return best_ref


def collapse_homopolymers_simple(seq: str) -> str:
    """rle, drop the run lengths."""
    if not seq:
        return ""
    result = [seq[0]]
    for base in seq[1:]:
        if base != result[-1]:
            result.append(base)
    return ''.join(result)


# alignment-based error analysis

def extract_homopolymer_runs(sequence: str) -> List[Tuple[str, int, int]]:
    """rle. returns [(base, length, start_position), ...]."""
    if not sequence:
        return []

    runs = []
    current_base = sequence[0]
    current_start = 0
    current_len = 1

    for i, base in enumerate(sequence[1:], 1):
        if base == current_base:
            current_len += 1
        else:
            runs.append((current_base, current_len, current_start))
            current_base = base
            current_start = i
            current_len = 1

    runs.append((current_base, current_len, current_start))
    return runs


def align_homopolymer_runs(
    observed_runs: List[Tuple[str, int, int]],
    reference_runs: List[Tuple[str, int, int]]
) -> Tuple[List[Tuple], AlignmentErrorStats]:
    """walk both run lists in parallel and emit per-run alignments.

    when the bases agree we record the (expected, observed) length pair;
    when they don't, we look ahead one run on each side and decide whether
    it's an insertion, deletion, or substitution. naive but works because
    the collapsed sequences usually match. returns (alignments, stats).
    """
    stats = AlignmentErrorStats()

    # Simple run-by-run alignment (assumes runs are mostly in order)
    # This works well when the collapsed sequences match

    obs_idx = 0
    ref_idx = 0
    alignments = []

    while obs_idx < len(observed_runs) and ref_idx < len(reference_runs):
        obs_base, obs_len, obs_pos = observed_runs[obs_idx]
        ref_base, ref_len, ref_pos = reference_runs[ref_idx]

        if obs_base == ref_base:
            # matching base - check length
            alignments.append(('match', ref_base, ref_len, obs_len))
            stats.total_reference_bases += ref_len
            stats.total_aligned_bases += obs_len

            length_diff = obs_len - ref_len
            stats.hp_length_diffs.append(length_diff)
            stats.hp_length_by_base[ref_base].append((ref_len, obs_len))

            # track errors by reference hp length for pcr analysis
            stats.hp_errors_by_ref_length[ref_len].append(length_diff)

            if length_diff != 0:
                stats.homopolymer_length_errors += abs(length_diff)

            obs_idx += 1
            ref_idx += 1

        else:
            # bases differ - use lookahead to distinguish ins/del/sub
            skip_obs = (obs_idx + 1 < len(observed_runs)
                        and observed_runs[obs_idx + 1][0] == ref_base)
            skip_ref = (ref_idx + 1 < len(reference_runs)
                        and reference_runs[ref_idx + 1][0] == obs_base)

            if skip_obs and not skip_ref:
                # extra observed run = insertion
                alignments.append(('insertion', obs_base, 0, obs_len))
                stats.insertions += obs_len
                stats.insertions_by_base[obs_base] += obs_len
                stats.total_aligned_bases += obs_len
                obs_idx += 1
            elif skip_ref and not skip_obs:
                # missing reference run = deletion
                alignments.append(('deletion', ref_base, ref_len, 0))
                stats.deletions += ref_len
                stats.deletions_by_base[ref_base] += ref_len
                stats.total_reference_bases += ref_len
                ref_idx += 1
            elif skip_obs and skip_ref:
                # ambiguous - treat as insertion (skip obs)
                alignments.append(('insertion', obs_base, 0, obs_len))
                stats.insertions += obs_len
                stats.insertions_by_base[obs_base] += obs_len
                stats.total_aligned_bases += obs_len
                obs_idx += 1
            else:
                # neither lookahead matches - substitution
                alignments.append(('substitution', ref_base, ref_len, obs_len))
                stats.substitutions += 1
                pair_key = f"{ref_base}>{obs_base}"
                stats.substitutions_by_pair[pair_key] += 1
                stats.total_reference_bases += ref_len
                stats.total_aligned_bases += obs_len
                obs_idx += 1
                ref_idx += 1

    # Handle remaining runs
    while obs_idx < len(observed_runs):
        obs_base, obs_len, _ = observed_runs[obs_idx]
        stats.insertions += obs_len
        stats.insertions_by_base[obs_base] += obs_len
        stats.total_aligned_bases += obs_len
        obs_idx += 1

    while ref_idx < len(reference_runs):
        ref_base, ref_len, _ = reference_runs[ref_idx]
        stats.deletions += ref_len
        stats.deletions_by_base[ref_base] += ref_len
        stats.total_reference_bases += ref_len
        ref_idx += 1

    return alignments, stats


def analyze_read_against_reference(
    observed_core: str,
    reference: ReferenceSequence,
    global_stats: AlignmentErrorStats
) -> AlignmentErrorStats:
    """run-level alignment for one read. mutates global_stats and also
    returns the read-level numbers."""
    # strip g-tail from both
    obs_trimmed = observed_core.upper().rstrip('G')
    ref_trimmed = reference.core_sequence.upper().rstrip('G')

    # extract homopolymer runs
    obs_runs = extract_homopolymer_runs(obs_trimmed)
    ref_runs = extract_homopolymer_runs(ref_trimmed)

    # align and count errors
    _, read_stats = align_homopolymer_runs(obs_runs, ref_runs)

    # accumulate into global stats
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

    global_stats.hp_length_diffs.extend(read_stats.hp_length_diffs)
    for base, pairs in read_stats.hp_length_by_base.items():
        global_stats.hp_length_by_base[base].extend(pairs)

    # accumulate hp errors by ref length
    for ref_len, diffs in read_stats.hp_errors_by_ref_length.items():
        global_stats.hp_errors_by_ref_length[ref_len].extend(diffs)

    return read_stats


def analyze_pcr_signatures(results: ErrorAnalysisResults) -> Dict[str, any]:
    """try to tell pcr slippage from sequencing error.

    pcr slippage shows up as: strand symmetry (it's in the DNA, not the
    basecaller), most deviations are +-1 bp (polymerase stutter), and the
    error rate climbs with hp length. sequencing error tends to be more
    base-biased and asymmetric. we score each hypothesis and pick a winner.
    """
    analysis = {
        "strand_symmetry": {},
        "single_base_bias": {},
        "length_dependent_rate": {},
        "conclusion": "",
        "pcr_score": 0.0,
        "sequencing_score": 0.0,
    }

    pcr_stats = results.pcr_stats
    align_stats = results.alignment_stats

    # 1. strand symmetry analysis
    fwd_diffs = pcr_stats.forward_hp_diffs
    rev_diffs = pcr_stats.reverse_hp_diffs

    if len(fwd_diffs) > 10 and len(rev_diffs) > 10:
        fwd_mean = np.mean(fwd_diffs)
        rev_mean = np.mean(rev_diffs)
        fwd_std = np.std(fwd_diffs)
        rev_std = np.std(rev_diffs)

        analysis["strand_symmetry"] = {
            "forward_mean": float(fwd_mean),
            "forward_std": float(fwd_std),
            "forward_n": len(fwd_diffs),
            "reverse_mean": float(rev_mean),
            "reverse_std": float(rev_std),
            "reverse_n": len(rev_diffs),
        }

        # pcr errors should show similar distributions on both strands
        mean_diff = abs(fwd_mean - rev_mean)
        std_ratio = max(fwd_std, rev_std) / min(fwd_std, rev_std) if min(fwd_std, rev_std) > 0 else 999

        if mean_diff < 0.5 and std_ratio < 1.5:
            analysis["pcr_score"] += 2.0
            analysis["strand_symmetry"]["interpretation"] = "symmetric - consistent with pcr errors"
        else:
            analysis["sequencing_score"] += 2.0
            analysis["strand_symmetry"]["interpretation"] = "asymmetric - suggests sequencing errors"

    # 2. ±1 bp bias analysis (pcr slippage signature)
    all_diffs = align_stats.hp_length_diffs
    if all_diffs:
        plus_minus_one = sum(1 for d in all_diffs if abs(d) == 1)
        larger = sum(1 for d in all_diffs if abs(d) > 1)
        zero = sum(1 for d in all_diffs if d == 0)
        total_errors = plus_minus_one + larger

        analysis["single_base_bias"] = {
            "exact_matches": zero,
            "plus_minus_one": plus_minus_one,
            "larger_deviations": larger,
        }

        if total_errors > 0:
            pm1_fraction = plus_minus_one / total_errors
            analysis["single_base_bias"]["pm1_fraction"] = float(pm1_fraction)

            # pcr slippage typically shows >60% ±1 bp errors
            if pm1_fraction > 0.6:
                analysis["pcr_score"] += 1.5
                analysis["single_base_bias"]["interpretation"] = "±1 bp dominant - consistent with pcr slippage"
            else:
                analysis["sequencing_score"] += 1.0
                analysis["single_base_bias"]["interpretation"] = "larger deviations common - suggests sequencing errors"

    # 3. length-dependent error rate (pcr signature: rate increases with hp length)
    hp_errors_by_len = align_stats.hp_errors_by_ref_length
    if hp_errors_by_len:
        length_error_rates = []
        for ref_len in sorted(hp_errors_by_len.keys()):
            diffs = hp_errors_by_len[ref_len]
            if len(diffs) >= 5:  # require minimum sample size
                error_count = sum(1 for d in diffs if d != 0)
                error_rate = error_count / len(diffs)
                length_error_rates.append((ref_len, error_rate, len(diffs)))

        analysis["length_dependent_rate"]["data"] = length_error_rates

        if len(length_error_rates) >= 3:
            lengths = [x[0] for x in length_error_rates]
            rates = [x[1] for x in length_error_rates]

            # check if error rate increases with length
            correlation = np.corrcoef(lengths, rates)[0, 1] if np.std(rates) > 0 else 0
            analysis["length_dependent_rate"]["correlation"] = float(correlation)

            if correlation > 0.5:
                analysis["pcr_score"] += 1.5
                analysis["length_dependent_rate"]["interpretation"] = "error rate increases with hp length - pcr signature"
            elif correlation < -0.3:
                analysis["sequencing_score"] += 1.0
                analysis["length_dependent_rate"]["interpretation"] = "error rate decreases with length - unusual"
            else:
                analysis["length_dependent_rate"]["interpretation"] = "no clear length dependence"

    # 4. final conclusion
    total_score = analysis["pcr_score"] + analysis["sequencing_score"]
    if total_score > 0:
        pcr_fraction = analysis["pcr_score"] / total_score
        if pcr_fraction > 0.6:
            analysis["conclusion"] = "pcr_likely"
        elif pcr_fraction < 0.4:
            analysis["conclusion"] = "sequencing_likely"
        else:
            analysis["conclusion"] = "mixed"
    else:
        analysis["conclusion"] = "insufficient_data"

    return analysis


# pattern matching and alignment

def get_pattern_dict() -> Dict[str, str]:
    """name -> pattern dict from babbage_sequence_list.CONSTANTS."""
    return {name: pattern for name, pattern in CONSTANTS}

def align_to_pattern(observed_seq: str, pattern: str) -> Optional[List[Tuple[str, str, int, int]]]:
    """collapse homopolymers in observed_seq, align the collapsed string to
    the pattern by trying small offsets, then map back to per-run positions.

    returns [(expected_base, observed_base, observed_run_len, run_start), ...]
    of length len(pattern), or None if even the best offset matches < 50%.
    """
    if not observed_seq or not pattern:
        return None

    # Collapse homopolymers and track positions
    collapsed = []
    run_lengths = []
    run_starts = []

    if observed_seq:
        current_base = observed_seq[0]
        current_start = 0
        current_len = 1

        for i, base in enumerate(observed_seq[1:], 1):
            if base == current_base:
                current_len += 1
            else:
                collapsed.append(current_base)
                run_lengths.append(current_len)
                run_starts.append(current_start)
                current_base = base
                current_start = i
                current_len = 1

        # Don't forget the last run
        collapsed.append(current_base)
        run_lengths.append(current_len)
        run_starts.append(current_start)

    collapsed_seq = ''.join(collapsed)

    # Simple alignment: find best offset match
    # Allow for insertions/deletions at start/end
    best_score = -1
    best_offset = 0

    for offset in range(-5, 6):
        score = 0
        for i, expected_base in enumerate(pattern):
            obs_idx = i + offset
            if 0 <= obs_idx < len(collapsed_seq):
                if collapsed_seq[obs_idx] == expected_base:
                    score += 1
        if score > best_score:
            best_score = score
            best_offset = offset

    # Require at least 50% match
    if best_score < len(pattern) * 0.5:
        return None

    # Build alignment
    alignment = []
    for i, expected_base in enumerate(pattern):
        obs_idx = i + best_offset
        if 0 <= obs_idx < len(collapsed_seq):
            alignment.append((
                expected_base,
                collapsed_seq[obs_idx],
                run_lengths[obs_idx],
                run_starts[obs_idx]
            ))
        else:
            alignment.append((expected_base, '-', 0, -1))

    return alignment


def identify_pattern(sequence: str, patterns: Dict[str, str]) -> Optional[Tuple[str, str, float]]:
    """rough pattern picker: collapse, strip g-tail, score against each
    pattern by best-offset match. for downstream we use the hmm-based
    identifier in profile_hmm; this one is a lightweight prefilter."""
    # Collapse homopolymers
    if not sequence:
        return None

    collapsed = []
    prev = None
    for base in sequence.upper():
        if base != prev:
            collapsed.append(base)
            prev = base
    collapsed_seq = ''.join(collapsed)

    # Strip G-tail if present
    collapsed_seq = collapsed_seq.rstrip('G')

    best_match = None
    best_score = 0

    for name, pattern in patterns.items():
        # Try different alignments
        for offset in range(-3, 4):
            matches = 0
            total = 0
            for i, expected in enumerate(pattern):
                obs_idx = i + offset
                if 0 <= obs_idx < len(collapsed_seq):
                    total += 1
                    if collapsed_seq[obs_idx] == expected:
                        matches += 1

            if total > 0 and len(pattern) > 0:
                score = matches / len(pattern)
                if score > best_score:
                    best_score = score
                    best_match = (name, pattern, score)

    if best_score >= 0.6:  # Require 60% match
        return best_match
    return None



# error analysis functions


def analyze_homopolymers(
    alignment: List[Tuple[str, str, int, int]],
    stats: HomopolymerStats,
    expected_hp_length: int = 3
) -> None:
    """
    Analyze homopolymer length accuracy.

    Synthesis errors manifest as:
    - Systematic over/under-extension for specific bases
    - Length-dependent error rates

    Sequencing errors manifest as:
    - Random length variations
    - Longer homopolymers have higher error rates (nanopore characteristic)
    """
    for expected_base, observed_base, observed_length, _ in alignment:
        if observed_base == '-':
            continue

        # For each base in pattern, we expect ~expected_hp_length nucleotides
        # due to TdT processivity
        expected_len = expected_hp_length

        stats.expected_lengths[expected_base].append(expected_len)
        stats.observed_lengths[expected_base].append(observed_length)
        stats.length_errors[expected_base][expected_len].append(observed_length)


def analyze_substitutions(
    alignment: List[Tuple[str, str, int, int]],
    stats: SubstitutionStats
) -> None:
    """
    Analyze base substitution patterns.

    Synthesis errors:
    - TdT has nucleotide-specific incorporation preferences
    - Systematic biases for certain substitutions

    Sequencing errors:
    - Nanopore has characteristic substitution patterns
    - Context-dependent errors

    Storage/oxidative damage:
    - G -> T (8-oxoguanine)
    - C -> T (cytosine deamination)
    """
    for expected_base, observed_base, _, _ in alignment:
        if observed_base == '-':
            continue

        stats.total_bases += 1
        stats.matrix[expected_base][observed_base] += 1

        if expected_base != observed_base:
            stats.total_errors += 1


def analyze_positional_errors(
    alignment: List[Tuple[str, str, int, int]],
    quality_scores: List[int],
    stats: PositionalStats,
    observed_seq: str
) -> None:
    """
    Analyze error rates by position in the sequence.

    Synthesis errors:
    - May show position-dependent patterns due to synthesis kinetics

    Sequencing errors:
    - Often higher at read starts/ends
    - Quality scores correlate with position

    Storage degradation:
    - End-degradation from exonuclease activity
    - Random throughout if oxidative damage
    """
    # Ensure stats lists are long enough
    while len(stats.error_counts) < len(alignment):
        stats.error_counts.append(0)
        stats.total_counts.append(0)
        stats.quality_by_position.append([])

    for i, (expected_base, observed_base, _, obs_start) in enumerate(alignment):
        if observed_base == '-':
            continue

        stats.total_counts[i] += 1
        if expected_base != observed_base:
            stats.error_counts[i] += 1

        # Get quality score for this position
        if obs_start >= 0 and obs_start < len(quality_scores):
            stats.quality_by_position[i].append(quality_scores[obs_start])


def analyze_quality_correlation(
    sequence: str,
    quality_scores: List[int],
    alignment: List[Tuple[str, str, int, int]],
    quality_bins: Dict[int, Tuple[int, int]]  # bin -> (errors, total)
) -> None:
    """
    Analyze correlation between quality scores and error rates.

    Strong correlation indicates sequencing errors.
    Weak correlation suggests synthesis or storage errors.
    """
    for expected_base, observed_base, _, obs_start in alignment:
        if observed_base == '-' or obs_start < 0:
            continue

        if obs_start < len(quality_scores):
            q = quality_scores[obs_start]
            q_bin = (q // 5) * 5  # Bin by 5

            if q_bin not in quality_bins:
                quality_bins[q_bin] = [0, 0]

            quality_bins[q_bin][1] += 1
            if expected_base != observed_base:
                quality_bins[q_bin][0] += 1


def detect_strand_orientation(sequence: str) -> str:
    """
    Detect if read is in forward or reverse orientation.

    Forward: starts with FWD primer
    Reverse: starts with RC(REV primer)
    """
    seq_upper = sequence.upper()
    fwd = DEFAULT_FORWARD_PRIMER.upper()
    rc_rev = reverse_complement(DEFAULT_REVERSE_PRIMER.upper())

    # Check first 35 bases
    if seq_upper[:len(fwd)] == fwd or seq_upper[:20] == fwd[:20]:
        return "forward"
    elif seq_upper[:len(rc_rev)] == rc_rev or seq_upper[:20] == rc_rev[:20]:
        return "reverse"
    return "unknown"



# fastq parsing


def parse_fastq(filepath: Path):
    """
    Parse FASTQ file (gzipped or plain).

    Yields (read_id, sequence, quality_scores) tuples.
    """
    opener = gzip.open if filepath.suffix == '.gz' else open

    with opener(filepath, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break

            sequence = f.readline().strip()
            _ = f.readline()  # + line
            quality = f.readline().strip()

            # Parse read ID
            read_id = header[1:].split()[0]

            # Convert quality string to Phred scores
            quality_scores = [ord(c) - 33 for c in quality]

            yield read_id, sequence, quality_scores



# main analysis pipeline


def run_analysis(
    fastq_path: Path,
    output_dir: Path,
    reference_fasta: Optional[Path] = None,
    min_phred: float = 7.0,
    min_length: int = 80,
    max_length: int = 500
) -> ErrorAnalysisResults:
    """
    Run comprehensive error analysis on FASTQ file.

    If reference_fasta is provided, performs alignment-based error analysis
    against the ground truth sequences.
    """
    results = ErrorAnalysisResults()
    patterns = get_pattern_dict()
    quality_bins: Dict[int, List[int]] = {}

    # Load reference sequences if provided
    reference_index = None
    if reference_fasta and reference_fasta.exists():
        logger.info(f"Loading reference sequences from {reference_fasta}...")
        references = parse_fasta(reference_fasta)
        reference_index = build_reference_index(references)
        logger.info(f"  Loaded {len(references)} reference sequences for {len(reference_index)} constants")

    logger.info(f"Analyzing {fastq_path}...")

    for read_id, sequence, quality_scores in parse_fastq(fastq_path):
        results.total_reads += 1

        # Basic filtering
        mean_phred = np.mean(quality_scores) if quality_scores else 0
        if mean_phred < min_phred:
            continue
        if len(sequence) < min_length or len(sequence) > max_length:
            continue

        # Detect strand orientation
        strand = detect_strand_orientation(sequence)

        # Trim primers
        trimmed = trim_primers(sequence)
        if trimmed is None or len(trimmed) < 10:
            continue

        # Get quality scores for trimmed region (approximate)
        # This is simplified - in practice would need exact position tracking
        trim_start = sequence.upper().find(trimmed[:20].upper()) if len(trimmed) >= 20 else 0
        if trim_start == -1:
            trim_start = 0
        trimmed_quality = quality_scores[trim_start:trim_start + len(trimmed)]

        # Identify pattern
        match = identify_pattern(trimmed, patterns)
        if match is None:
            continue

        pattern_name, pattern, score = match
        results.pattern_matches[pattern_name] += 1
        results.analyzed_reads += 1

        # reference-based alignment if available
        if reference_index:
            best_ref = find_best_reference(trimmed, pattern_name, reference_index)
            if best_ref:
                read_stats = analyze_read_against_reference(trimmed, best_ref, results.alignment_stats)

                # track strand-specific hp diffs for pcr analysis
                if strand == "forward":
                    results.pcr_stats.forward_hp_diffs.extend(read_stats.hp_length_diffs)
                elif strand == "reverse":
                    results.pcr_stats.reverse_hp_diffs.extend(read_stats.hp_length_diffs)

                # track ±1 bp vs larger deviations
                for diff in read_stats.hp_length_diffs:
                    if abs(diff) == 1:
                        results.pcr_stats.plus_minus_one_count += 1
                    elif abs(diff) > 1:
                        results.pcr_stats.larger_deviation_count += 1

        # align to pattern
        alignment = align_to_pattern(trimmed, pattern)
        if alignment is None:
            continue

        # Analyze errors
        analyze_homopolymers(alignment, results.homopolymer_stats)
        analyze_substitutions(alignment, results.substitution_stats)
        analyze_positional_errors(alignment, trimmed_quality, results.positional_stats, trimmed)
        analyze_quality_correlation(trimmed, trimmed_quality, alignment, quality_bins)

        # Track strand-specific errors
        errors = sum(1 for e, o, _, _ in alignment if e != o and o != '-')
        total = sum(1 for _, o, _, _ in alignment if o != '-')

        if strand == "forward":
            results.strand_stats.forward_errors += errors
            results.strand_stats.forward_total += total
        elif strand == "reverse":
            results.strand_stats.reverse_errors += errors
            results.strand_stats.reverse_total += total

    # Compile quality correlation
    for q_bin in sorted(quality_bins.keys()):
        errors, total = quality_bins[q_bin]
        results.quality_correlation.quality_bins.append(q_bin)
        results.quality_correlation.error_rates.append(errors / total if total > 0 else 0)
        results.quality_correlation.counts.append(total)

    logger.info(f"Analyzed {results.analyzed_reads}/{results.total_reads} reads")

    return results



# reporting


def classify_error_source(results: ErrorAnalysisResults) -> Dict[str, any]:
    """
    Classify errors as likely synthesis vs sequencing based on patterns.
    """
    classification = {
        "synthesis_indicators": [],
        "sequencing_indicators": [],
        "synthesis_score": 0.0,
        "sequencing_score": 0.0,
        "analysis": {}
    }

    # 1. Homopolymer length analysis
    hp_stats = results.homopolymer_stats
    length_variance_by_base = {}
    for base in ['A', 'C', 'T', 'G']:
        if hp_stats.observed_lengths[base]:
            lengths = hp_stats.observed_lengths[base]
            variance = np.var(lengths) if len(lengths) > 1 else 0
            mean_len = np.mean(lengths)
            length_variance_by_base[base] = {
                "mean": float(mean_len),
                "variance": float(variance),
                "cv": float(np.sqrt(variance) / mean_len) if mean_len > 0 else 0
            }

    # High variance in homopolymer lengths suggests synthesis variation
    avg_cv = np.mean([v["cv"] for v in length_variance_by_base.values()]) if length_variance_by_base else 0
    if avg_cv > 0.5:
        classification["synthesis_indicators"].append(
            f"High homopolymer length variability (CV={avg_cv:.2f}) suggests TdT processivity variation"
        )
        classification["synthesis_score"] += 1.0

    classification["analysis"]["homopolymer_stats"] = length_variance_by_base

    # 2. Substitution pattern analysis
    sub_stats = results.substitution_stats
    if sub_stats.total_bases > 0:
        error_rate = sub_stats.total_errors / sub_stats.total_bases
        classification["analysis"]["overall_error_rate"] = float(error_rate)

        # Build substitution matrix percentages
        sub_matrix = {}
        for expected in sub_stats.matrix:
            sub_matrix[expected] = {}
            row_total = sum(sub_stats.matrix[expected].values())
            for observed, count in sub_stats.matrix[expected].items():
                sub_matrix[expected][observed] = count / row_total if row_total > 0 else 0

        classification["analysis"]["substitution_matrix"] = sub_matrix

        # Check for oxidative damage signatures (would indicate storage issues)
        # G->T (8-oxoguanine) and C->T (deamination)
        g_to_t = sub_matrix.get('G', {}).get('T', 0)
        c_to_t = sub_matrix.get('C', {}).get('T', 0)

        # Note: patterns don't use G, so we mainly check C->T
        if c_to_t > 0.1:
            classification["synthesis_indicators"].append(
                f"Elevated C->T transitions ({c_to_t:.1%}) may indicate cytosine deamination"
            )

    # 3. Quality score correlation
    qc = results.quality_correlation
    if len(qc.quality_bins) >= 3 and len(qc.error_rates) >= 3:
        # Calculate correlation between quality and error rate
        # Higher quality should have lower error rate
        q_vals = np.array(qc.quality_bins)
        e_vals = np.array(qc.error_rates)

        if len(q_vals) > 2:
            correlation = np.corrcoef(q_vals, e_vals)[0, 1] if np.std(e_vals) > 0 else 0
            classification["analysis"]["quality_error_correlation"] = float(correlation)

            if correlation < -0.5:
                classification["sequencing_indicators"].append(
                    f"Strong quality-error correlation (r={correlation:.2f}) indicates sequencing errors"
                )
                classification["sequencing_score"] += 1.5
            elif correlation > -0.2:
                classification["synthesis_indicators"].append(
                    f"Weak quality-error correlation (r={correlation:.2f}) suggests errors independent of sequencing"
                )
                classification["synthesis_score"] += 0.5

    # 4. Positional error analysis
    pos_stats = results.positional_stats
    if pos_stats.error_counts and pos_stats.total_counts:
        error_rates = []
        for i in range(len(pos_stats.error_counts)):
            if pos_stats.total_counts[i] > 0:
                error_rates.append(pos_stats.error_counts[i] / pos_stats.total_counts[i])
            else:
                error_rates.append(0)

        if len(error_rates) >= 10:
            # Check if errors are higher at ends (sequencing artifact)
            n = len(error_rates)
            start_rate = np.mean(error_rates[:n//5]) if n >= 5 else 0
            middle_rate = np.mean(error_rates[n//5:4*n//5]) if n >= 5 else 0
            end_rate = np.mean(error_rates[4*n//5:]) if n >= 5 else 0

            classification["analysis"]["positional_error_rates"] = {
                "start": float(start_rate),
                "middle": float(middle_rate),
                "end": float(end_rate)
            }

            # require middle_rate > 0.01 to make comparison meaningful
            if middle_rate > 0.01 and (start_rate > middle_rate * 1.5 or end_rate > middle_rate * 1.5):
                classification["sequencing_indicators"].append(
                    "Higher error rates at sequence ends suggest sequencing quality degradation"
                )
                classification["sequencing_score"] += 1.0
            elif middle_rate > 0.01:
                classification["synthesis_indicators"].append(
                    "Uniform error distribution suggests synthesis-related errors"
                )
                classification["synthesis_score"] += 0.5

    # 5. Strand bias analysis
    strand = results.strand_stats
    if strand.forward_total > 100 and strand.reverse_total > 100:
        fwd_rate = strand.forward_errors / strand.forward_total
        rev_rate = strand.reverse_errors / strand.reverse_total

        classification["analysis"]["strand_error_rates"] = {
            "forward": float(fwd_rate),
            "reverse": float(rev_rate),
            "ratio": float(fwd_rate / rev_rate) if rev_rate > 0 else 0
        }

        ratio = fwd_rate / rev_rate if rev_rate > 0 else 1
        if ratio > 1.3 or ratio < 0.7:
            classification["sequencing_indicators"].append(
                f"Strand bias (ratio={ratio:.2f}) indicates strand-specific sequencing errors"
            )
            classification["sequencing_score"] += 1.0

    # Final classification
    total_score = classification["synthesis_score"] + classification["sequencing_score"]
    if total_score > 0:
        classification["synthesis_fraction"] = classification["synthesis_score"] / total_score
        classification["sequencing_fraction"] = classification["sequencing_score"] / total_score
    else:
        classification["synthesis_fraction"] = 0.5
        classification["sequencing_fraction"] = 0.5

    return classification


def generate_report(results: ErrorAnalysisResults, output_dir: Path) -> None:
    """generate text report."""
    classification = classify_error_source(results)

    report_path = output_dir / "error_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("Error analysis report\n\n")

        # summary
        f.write("Summary\n")
        f.write(f"  Total reads processed: {results.total_reads}\n")
        f.write(f"  Reads analyzed: {results.analyzed_reads}\n")
        f.write(f"  Analysis rate: {results.analyzed_reads/results.total_reads*100:.1f}%\n\n")

        # pattern distribution
        f.write("Pattern distribution\n")
        for name, count in sorted(results.pattern_matches.items(), key=lambda x: -x[1]):
            f.write(f"  {name:25s}: {count:5d}\n")
        f.write("\n")

        # error classification
        f.write("Error source classification\n")
        f.write(f"  Estimated synthesis-related: {classification['synthesis_fraction']*100:.1f}%\n")
        f.write(f"  Estimated sequencing-related: {classification['sequencing_fraction']*100:.1f}%\n\n")

        # indicators
        f.write("Synthesis error indicators\n")
        if classification["synthesis_indicators"]:
            for indicator in classification["synthesis_indicators"]:
                f.write(f"  * {indicator}\n")
        else:
            f.write("  (none detected)\n")
        f.write("\n")

        f.write("Sequencing error indicators\n")
        if classification["sequencing_indicators"]:
            for indicator in classification["sequencing_indicators"]:
                f.write(f"  * {indicator}\n")
        else:
            f.write("  (none detected)\n")
        f.write("\n")

        # alignment-based error breakdown (if reference was used)
        align_stats = results.alignment_stats
        if align_stats.total_reference_bases > 0:
            f.write("Error type breakdown (vs reference)\n\n")

            # count hp runs with errors vs correct
            hp_diffs = align_stats.hp_length_diffs
            hp_correct = sum(1 for d in hp_diffs if d == 0)
            hp_total = len(hp_diffs)
            hp_accuracy = hp_correct / hp_total * 100 if hp_total > 0 else 0

            # alignment length = reference + net insertions
            alignment_len = align_stats.total_reference_bases + align_stats.insertions - align_stats.deletions

            f.write("Homopolymer run statistics\n")
            f.write(f"  Total HP runs aligned:   {hp_total:>8,}\n")
            f.write(f"  Correct length:          {hp_correct:>8,}\n")
            f.write(f"  HP accuracy:             {hp_accuracy:>7.1f}%\n")
            f.write(f"  Unmatched runs (ins):    {align_stats.insertions:>8,} bases\n")
            f.write(f"  Unmatched runs (del):    {align_stats.deletions:>8,} bases\n\n")

            f.write("Base counts\n")
            f.write(f"  Reference bases:         {align_stats.total_reference_bases:>8,}\n")
            f.write(f"  Observed bases:          {align_stats.total_aligned_bases:>8,}\n")
            f.write(f"  Net length change:       {align_stats.total_aligned_bases - align_stats.total_reference_bases:>+8,}\n\n")

            f.write("Unmatched runs by base (insertions)\n")
            for base in ['A', 'C', 'T', 'G']:
                count = align_stats.insertions_by_base.get(base, 0)
                if count > 0:
                    f.write(f"  {base}: {count:>6,}\n")
            f.write("\n")

            f.write("Unmatched runs by base (deletions)\n")
            for base in ['A', 'C', 'T', 'G']:
                count = align_stats.deletions_by_base.get(base, 0)
                if count > 0:
                    f.write(f"  {base}: {count:>6,}\n")
            f.write("\n")

            # homopolymer length error distribution
            if align_stats.hp_length_diffs:
                f.write("Homopolymer length deviation distribution\n")
                diffs = align_stats.hp_length_diffs
                f.write(f"  Mean deviation: {np.mean(diffs):+.2f} bases\n")
                f.write(f"  Std deviation:  {np.std(diffs):.2f} bases\n")
                f.write(f"  Min: {min(diffs):+d}, Max: {max(diffs):+d}\n\n")

                # histogram of deviations
                from collections import Counter
                diff_counts = Counter(diffs)
                f.write("  Deviation  Count\n")
                for diff in sorted(diff_counts.keys()):
                    if abs(diff) <= 5:
                        bar = '#' * min(50, diff_counts[diff] // 10)
                        f.write(f"    {diff:+3d}:  {diff_counts[diff]:>6,}  {bar}\n")
            f.write("\n")

            # interpretation
            f.write("Interpretation\n")

            # classify based on hp accuracy and unmatched runs
            unmatched_runs = align_stats.insertions + align_stats.deletions
            if hp_total > 0:
                unmatched_fraction = unmatched_runs / (hp_total * 3 + unmatched_runs)  # rough estimate

                if hp_accuracy > 80:
                    f.write("  * High HP accuracy - good synthesis fidelity\n")
                elif hp_accuracy > 50:
                    f.write("  * Moderate HP accuracy - some length variation\n")
                    f.write("    Common in TdT synthesis or nanopore homopolymer calling\n")
                else:
                    f.write("  * Low HP accuracy - significant length errors\n")
                    f.write("    May indicate synthesis issues or basecalling difficulty\n")

                # check for systematic bias
                if hp_diffs:
                    mean_diff = np.mean(hp_diffs)
                    if mean_diff > 0.5:
                        f.write(f"\n  * Systematic over-extension (+{mean_diff:.1f} bases avg)\n")
                        f.write("    TdT adding too many nucleotides per cycle\n")
                    elif mean_diff < -0.5:
                        f.write(f"\n  * Systematic under-extension ({mean_diff:.1f} bases avg)\n")
                        f.write("    TdT adding too few nucleotides or nanopore undercalling\n")
            f.write("\n")

        # pcr vs sequencing analysis
        pcr_analysis = analyze_pcr_signatures(results)
        f.write("PCR vs sequencing error analysis\n\n")

        # strand symmetry
        strand_sym = pcr_analysis.get("strand_symmetry", {})
        if strand_sym:
            f.write("Strand symmetry\n")
            f.write(f"  Forward strand: mean={strand_sym.get('forward_mean', 0):+.2f}, "
                   f"std={strand_sym.get('forward_std', 0):.2f}, n={strand_sym.get('forward_n', 0)}\n")
            f.write(f"  Reverse strand: mean={strand_sym.get('reverse_mean', 0):+.2f}, "
                   f"std={strand_sym.get('reverse_std', 0):.2f}, n={strand_sym.get('reverse_n', 0)}\n")
            if "interpretation" in strand_sym:
                f.write(f"  Interpretation: {strand_sym['interpretation']}\n")
            f.write("\n")

        # single-base (±1 bp) bias
        sb_bias = pcr_analysis.get("single_base_bias", {})
        if sb_bias:
            f.write("Homopolymer deviation pattern\n")
            f.write(f"  Exact matches: {sb_bias.get('exact_matches', 0):,}\n")
            f.write(f"  ±1 bp deviations: {sb_bias.get('plus_minus_one', 0):,}\n")
            f.write(f"  Larger deviations: {sb_bias.get('larger_deviations', 0):,}\n")
            if "pm1_fraction" in sb_bias:
                f.write(f"  ±1 bp fraction of errors: {sb_bias['pm1_fraction']*100:.1f}%\n")
            if "interpretation" in sb_bias:
                f.write(f"  Interpretation: {sb_bias['interpretation']}\n")
            f.write("\n")

        # length-dependent error rate
        len_dep = pcr_analysis.get("length_dependent_rate", {})
        if len_dep and len_dep.get("data"):
            f.write("Error rate by homopolymer length\n")
            f.write("  HP length  Error rate    n\n")
            for hp_len, error_rate, n in len_dep["data"]:
                f.write(f"      {hp_len:3d}      {error_rate*100:5.1f}%    {n:4d}\n")
            if "correlation" in len_dep:
                f.write(f"\n  Length-error correlation: r={len_dep['correlation']:+.3f}\n")
            if "interpretation" in len_dep:
                f.write(f"  Interpretation: {len_dep['interpretation']}\n")
            f.write("\n")

        # overall pcr conclusion
        f.write("PCR analysis conclusion\n")
        f.write(f"  PCR signature score: {pcr_analysis.get('pcr_score', 0):.1f}\n")
        f.write(f"  Sequencing signature score: {pcr_analysis.get('sequencing_score', 0):.1f}\n")
        conclusion = pcr_analysis.get("conclusion", "unknown")
        if conclusion == "pcr_likely":
            f.write("  Verdict: errors are consistent with PCR slippage\n")
        elif conclusion == "sequencing_likely":
            f.write("  Verdict: errors are consistent with sequencing artifacts\n")
        elif conclusion == "mixed":
            f.write("  Verdict: mixed signature - both PCR and sequencing errors likely present\n")
        else:
            f.write("  Verdict: insufficient data for determination\n")
        f.write("\n")

        # detailed statistics
        f.write("Detailed statistics\n\n")

        # substitution matrix
        f.write("Substitution matrix (row=expected, col=observed)\n")
        sub_stats = results.substitution_stats
        bases = ['A', 'C', 'T', 'G']
        f.write("     " + "".join(f"{b:>8s}" for b in bases) + "\n")
        for exp in bases:
            row_total = sum(sub_stats.matrix[exp].values())
            f.write(f"  {exp}  ")
            for obs in bases:
                count = sub_stats.matrix[exp][obs]
                pct = count / row_total * 100 if row_total > 0 else 0
                f.write(f"{pct:7.1f}%")
            f.write(f"  (n={row_total})\n")

        f.write(f"\nOverall error rate: {sub_stats.total_errors/sub_stats.total_bases*100:.2f}%\n"
                if sub_stats.total_bases > 0 else "")

        # Homopolymer stats
        f.write("\nHomopolymer Length Statistics:\n")
        hp_stats = results.homopolymer_stats
        for base in ['A', 'C', 'T']:
            if hp_stats.observed_lengths[base]:
                lengths = hp_stats.observed_lengths[base]
                f.write(f"  {base}: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
                       f"min={min(lengths)}, max={max(lengths)}, n={len(lengths)}\n")

        # Quality correlation
        f.write("\nQuality Score vs Error Rate:\n")
        qc = results.quality_correlation
        for i, (q, e, n) in enumerate(zip(qc.quality_bins, qc.error_rates, qc.counts)):
            f.write(f"  Q{q:2.0f}-{q+5:.0f}: {e*100:5.2f}% error (n={n})\n")

        # Strand stats
        f.write("\nStrand-specific Statistics:\n")
        strand = results.strand_stats
        if strand.forward_total > 0:
            f.write(f"  Forward: {strand.forward_errors}/{strand.forward_total} "
                   f"({strand.forward_errors/strand.forward_total*100:.2f}% error)\n")
        if strand.reverse_total > 0:
            f.write(f"  Reverse: {strand.reverse_errors}/{strand.reverse_total} "
                   f"({strand.reverse_errors/strand.reverse_total*100:.2f}% error)\n")

    logger.info(f"Report written to {report_path}")


def generate_tsv_output(results: ErrorAnalysisResults, output_dir: Path) -> None:
    """Generate machine-readable TSV output (key/type/value record)."""
    classification = classify_error_source(results)

    output = {
        "summary": {
            "total_reads": results.total_reads,
            "analyzed_reads": results.analyzed_reads,
            "pattern_matches": dict(results.pattern_matches)
        },
        "classification": classification,
        "homopolymer_analysis": {
            base: {
                "observed_lengths": results.homopolymer_stats.observed_lengths[base],
                "mean": float(np.mean(results.homopolymer_stats.observed_lengths[base]))
                    if results.homopolymer_stats.observed_lengths[base] else 0,
                "std": float(np.std(results.homopolymer_stats.observed_lengths[base]))
                    if results.homopolymer_stats.observed_lengths[base] else 0
            }
            for base in ['A', 'C', 'T', 'G']
            if results.homopolymer_stats.observed_lengths[base]
        },
        "substitution_matrix": {
            exp: dict(obs_counts)
            for exp, obs_counts in results.substitution_stats.matrix.items()
        },
        "quality_correlation": {
            "bins": results.quality_correlation.quality_bins,
            "error_rates": results.quality_correlation.error_rates,
            "counts": results.quality_correlation.counts
        },
        "strand_analysis": {
            "forward_errors": results.strand_stats.forward_errors,
            "forward_total": results.strand_stats.forward_total,
            "reverse_errors": results.strand_stats.reverse_errors,
            "reverse_total": results.strand_stats.reverse_total
        },
        "positional_errors": {
            "error_counts": results.positional_stats.error_counts[:50],  # First 50 positions
            "total_counts": results.positional_stats.total_counts[:50]
        }
    }

    tsv_path = output_dir / "error_analysis.tsv"
    write_record(tsv_path, output)

    logger.info(f"TSV output written to {tsv_path}")


def generate_tsv_output(results: ErrorAnalysisResults, output_dir: Path) -> None:
    """Generate detailed TSV output."""
    # Substitution matrix TSV
    sub_path = output_dir / "substitution_matrix.tsv"
    with open(sub_path, 'w') as f:
        bases = ['A', 'C', 'T', 'G']
        f.write("expected\t" + "\t".join(bases) + "\ttotal\n")
        for exp in bases:
            row_total = sum(results.substitution_stats.matrix[exp].values())
            counts = [str(results.substitution_stats.matrix[exp][obs]) for obs in bases]
            f.write(f"{exp}\t" + "\t".join(counts) + f"\t{row_total}\n")

    # Homopolymer lengths TSV
    hp_path = output_dir / "homopolymer_lengths.tsv"
    with open(hp_path, 'w') as f:
        f.write("base\tlength\tcount\n")
        for base in ['A', 'C', 'T', 'G']:
            lengths = results.homopolymer_stats.observed_lengths[base]
            if lengths:
                from collections import Counter
                for length, count in sorted(Counter(lengths).items()):
                    f.write(f"{base}\t{length}\t{count}\n")

    # Quality correlation TSV
    qc_path = output_dir / "quality_correlation.tsv"
    with open(qc_path, 'w') as f:
        f.write("quality_bin\terror_rate\tcount\n")
        for q, e, n in zip(
            results.quality_correlation.quality_bins,
            results.quality_correlation.error_rates,
            results.quality_correlation.counts
        ):
            f.write(f"{q}\t{e:.6f}\t{n}\n")

    # Positional errors TSV
    pos_path = output_dir / "positional_errors.tsv"
    with open(pos_path, 'w') as f:
        f.write("position\terror_count\ttotal_count\terror_rate\n")
        for i, (err, tot) in enumerate(zip(
            results.positional_stats.error_counts,
            results.positional_stats.total_counts
        )):
            rate = err / tot if tot > 0 else 0
            f.write(f"{i}\t{err}\t{tot}\t{rate:.6f}\n")

    logger.info(f"TSV files written to {output_dir}")


def generate_plots(results: ErrorAnalysisResults, output_dir: Path) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Homopolymer length distribution
    ax = axes[0, 0]
    colors = {'A': 'green', 'C': 'blue', 'T': 'red', 'G': 'orange'}
    for base in ['A', 'C', 'T']:
        lengths = results.homopolymer_stats.observed_lengths[base]
        if lengths:
            ax.hist(lengths, bins=range(1, max(lengths)+2), alpha=0.5,
                   label=f'{base} (n={len(lengths)})', color=colors[base])
    ax.set_xlabel('Homopolymer length [bp]')
    ax.set_ylabel('Count')
    ax.set_title('Homopolymer length distribution')
    ax.legend()

    # 2. Quality vs Error Rate
    ax = axes[0, 1]
    qc = results.quality_correlation
    if qc.quality_bins and qc.error_rates:
        ax.bar(qc.quality_bins, [e*100 for e in qc.error_rates], width=4, alpha=0.7)
        ax.set_xlabel('Quality score [Phred]')
        ax.set_ylabel('Error rate [%]')
        ax.set_title('Quality score vs error rate')

        # Add trend line
        if len(qc.quality_bins) >= 3:
            z = np.polyfit(qc.quality_bins, qc.error_rates, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(qc.quality_bins), max(qc.quality_bins), 100)
            ax.plot(x_line, [v*100 for v in p(x_line)], 'r--', alpha=0.8, label='Trend')
            ax.legend()

    # 3. Positional Error Rate
    ax = axes[1, 0]
    pos_stats = results.positional_stats
    if pos_stats.error_counts and pos_stats.total_counts:
        positions = range(len(pos_stats.error_counts))
        error_rates = [
            pos_stats.error_counts[i] / pos_stats.total_counts[i] * 100
            if pos_stats.total_counts[i] > 0 else 0
            for i in positions
        ]
        ax.plot(positions, error_rates, 'b-', alpha=0.7)
        ax.fill_between(positions, error_rates, alpha=0.3)
        ax.set_xlabel('Position in pattern')
        ax.set_ylabel('Error rate [%]')
        ax.set_title('Positional error distribution')

    # 4. Substitution Heatmap
    ax = axes[1, 1]
    bases = ['A', 'C', 'T', 'G']
    matrix = np.zeros((4, 4))
    for i, exp in enumerate(bases):
        row_total = sum(results.substitution_stats.matrix[exp].values())
        for j, obs in enumerate(bases):
            if row_total > 0:
                matrix[i, j] = results.substitution_stats.matrix[exp][obs] / row_total * 100

    im = ax.imshow(matrix, cmap='YlOrRd')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(bases)
    ax.set_yticklabels(bases)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Expected')
    ax.set_title('Substitution matrix [%]')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', color=color)

    plt.colorbar(im, ax=ax, label='%')

    plt.tight_layout()
    plot_path = output_dir / "error_analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {plot_path}")



# main entry point


def main():
    parser = argparse.ArgumentParser(
        description="Analyze errors in TdT-synthesized DNA sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input FASTQ file (or folder containing FASTQ files)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--min-phred",
        type=float,
        default=7.0,
        help="Minimum mean Phred score"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=80,
        help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "-r", "--reference",
        type=Path,
        default=None,
        help="Reference FASTA file (e.g., babbage.fasta) for ground truth comparison"
    )

    args = parser.parse_args()

    # Find FASTQ files
    input_path = args.input
    if input_path.is_dir():
        fastq_files = list(input_path.glob("*.fastq.gz")) + list(input_path.glob("*.fastq"))
        if not fastq_files:
            logger.error(f"No FASTQ files found in {input_path}")
            return 1
        fastq_path = fastq_files[0]
        output_dir = args.output or input_path
    else:
        fastq_path = input_path
        output_dir = args.output or input_path.parent

    # Auto-detect reference FASTA if not specified
    reference_fasta = args.reference
    if reference_fasta is None:
        # Look for babbage.fasta in common locations
        for candidate in [
            input_path.parent / "data" / "babbage.fasta",
            input_path.parent.parent / "data" / "babbage.fasta",
            Path("data/babbage.fasta"),
            input_path.parent / "babbage.fasta",
            input_path.parent.parent / "babbage.fasta",
            Path("babbage.fasta"),
        ]:
            if candidate.exists():
                reference_fasta = candidate
                logger.info(f"Auto-detected reference: {reference_fasta}")
                break

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = run_analysis(
        fastq_path,
        output_dir,
        reference_fasta=reference_fasta,
        min_phred=args.min_phred,
        min_length=args.min_length,
        max_length=args.max_length
    )

    # Generate outputs
    generate_report(results, output_dir)
    generate_tsv_output(results, output_dir)
    generate_tsv_output(results, output_dir)

    if not args.no_plots:
        generate_plots(results, output_dir)

    logger.info("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
