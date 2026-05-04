#!/usr/bin/env python3
"""
nanopore decoding pipeline for the babbage codec.

reads a folder (or one file) of fastqs, viterbi-aligns each read against
all 17 constants, decodes 32 transition bits per read, descrambles them,
reconstructs an IEEE 754 float, and (after grouping reads by predicted
constant) does confidence-weighted bit-level consensus.

usage:
    python3 nanopore_pipeline.py /path/to/fastq/folder
    python3 nanopore_pipeline.py /path/to/fastq -o ./out --detailed
"""

import os
import sys
import gzip
import struct
import hashlib
import argparse
from multiprocessing import Pool
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict
import math
import numpy as np

from logging_config import setup_logging, get_logger
from exceptions import PipelineIOError
from tsv_records import write_record

logger = get_logger(__name__)

from babbage_sequence_list import (
    CONSTANTS, DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER
)
from profile_hmm import (
    ProfileHMM, ProfileHMMDecoder, StateType, BASE_TO_IDX,
)


# lookup table for the binary_dna_codec transitions. only A,C,T appear in
# encoded payloads - a G means we left the encoded region.
TRANSITION_DECODE = {
    ('A', 'C'): 0, ('A', 'T'): 1,
    ('C', 'T'): 0, ('C', 'A'): 1,
    ('T', 'A'): 0, ('T', 'C'): 1,
}


@dataclass
class ReadResult:
    """everything we extracted from one read."""
    read_id: str
    sequence_length: int
    mean_phred: float
    passed_filter: bool
    filter_reason: Optional[str] = None

    # decoding output (only populated if passed_filter)
    predicted_constant: Optional[str] = None
    confidence: float = 0.0
    margin: float = 0.0

    bits: Optional[List[int]] = None
    n_valid_bits: int = 0
    n_invalid_bits: int = 0
    invalid_positions: List[int] = field(default_factory=list)

    decoded_float: Optional[float] = None
    float_category: str = "undecodable"  # valid / corrupted / nan / inf / undecodable

    # all 17 forward scores. handy for downstream analysis.
    all_scores: Optional[Dict[str, float]] = None


@dataclass
class ConsensusResult:
    """consensus output for one constant after we've grouped its reads."""
    name: str
    n_reads: int
    bits: List[int]               # 32 consensus bits
    bit_confidences: List[float]  # vote fraction per bit, 0..1
    decoded_float: Optional[float]
    mean_read_confidence: float


@dataclass
class PipelineStats:
    """rolling stats accumulated as the pipeline reads fastqs."""
    total_reads: int = 0
    passed_filter: int = 0
    failed_phred: int = 0
    failed_length: int = 0
    failed_no_primers: int = 0

    # decoding stats
    valid_floats: int = 0
    corrupted_floats: int = 0  # decoded but had 1-2 invalid bits
    nan_floats: int = 0
    inf_floats: int = 0
    undecodable: int = 0

    # per-constant accumulation
    constant_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    constant_confidences: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    constant_values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # per-constant bit accumulation for consensus
    constant_bits: Dict[str, List[Tuple[List[int], float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # consensus results (populated after all reads processed)
    consensus_results: Dict[str, ConsensusResult] = field(default_factory=dict)


def parse_fastq(filepath: Path) -> Iterator[Tuple[str, str, str]]:
    """yield (read_id, sequence, quality) per record. handles gzipped files."""
    open_func = gzip.open if filepath.suffix == '.gz' else open
    mode = 'rt' if filepath.suffix == '.gz' else 'r'

    try:
        with open_func(filepath, mode) as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break

                sequence = f.readline().strip()
                plus = f.readline().strip()
                quality = f.readline().strip()

                if not all([header, sequence, plus, quality]):
                    break

                # validate quality string
                if not validate_phred_quality(quality):
                    logger.warning(f"invalid phred quality in read, skipping")
                    continue

                # extract read ID from header (remove @ prefix)
                read_id = header[1:].split()[0] if header.startswith('@') else header

                yield read_id, sequence, quality
    except (IOError, gzip.BadGzipFile) as e:
        logger.error(f"failed to read {filepath}: {e}")
        raise PipelineIOError(f"cannot read FASTQ file: {filepath}") from e


def compute_mean_phred(quality_string: str) -> float:
    """mean phred score. assumes phred+33."""
    if not quality_string:
        return 0.0
    scores = [ord(c) - 33 for c in quality_string]
    return sum(scores) / len(scores)


def validate_phred_quality(quality_string: str) -> bool:
    """phred+33 in [0, 93]. anything else is junk and we skip the record."""
    for char in quality_string:
        score = ord(char) - 33
        if score < 0 or score > 93:
            return False
    return True


def find_fastq_files(folder: Path) -> List[Path]:
    """all *.fastq[.gz] / *.fq[.gz] in the folder, sorted."""
    files = []
    for pattern in ['*.fastq', '*.fq', '*.fastq.gz', '*.fq.gz']:
        files.extend(folder.glob(pattern))
    return sorted(files)


def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ACGT', 'TGCA')
    return seq.upper().translate(complement)[::-1]


def _find_primer_at_start(seq: str, primer: str, max_search: int = 50) -> int:
    """exact match in the first max_search bases, falling back to a partial
    match at position 0 if the primer was truncated. returns the index just
    past the primer, or -1."""
    # exact match within first max_search bases
    for i in range(min(max_search, len(seq) - len(primer) + 1)):
        if seq[i:i+len(primer)] == primer:
            return i + len(primer)

    # partial match at very start (primer may be truncated)
    for trim_len in range(5, 0, -1):
        if seq[:len(primer)-trim_len] == primer[trim_len:]:
            return len(primer) - trim_len

    return -1


def _find_primer_at_end(seq: str, primer: str, min_start: int = 0, max_search: int = 50) -> int:
    """mirror of _find_primer_at_start; returns the start index of the primer
    near the end of seq, or -1."""
    # exact match within last max_search bases
    search_start = max(min_start, len(seq) - max_search - len(primer))
    for i in range(len(seq) - len(primer), search_start - 1, -1):
        if i >= min_start and seq[i:i+len(primer)] == primer:
            return i

    # partial match at very end (primer may be truncated)
    for trim_len in range(5, 0, -1):
        if seq[-(len(primer)-trim_len):] == primer[:len(primer)-trim_len]:
            return len(seq) - (len(primer) - trim_len)

    return -1


def trim_primers(sequence: str) -> Optional[str]:
    """strip the fwd/rev primers off the read and return the core in forward
    orientation. tries forward first; if that fails, reverse-complements
    the read and retries. None if neither orientation has a primer match.
    """
    seq_upper = sequence.upper()
    fwd = DEFAULT_FORWARD_PRIMER.upper()
    rev = DEFAULT_REVERSE_PRIMER.upper()
    rc_fwd = reverse_complement(fwd)
    rc_rev = reverse_complement(rev)

    # Try forward orientation first: FWD -> core -> REV
    fwd_start = _find_primer_at_start(seq_upper, fwd)
    if fwd_start != -1:
        rev_end = _find_primer_at_end(seq_upper, rev, min_start=fwd_start)
        if rev_end == -1:
            rev_end = len(seq_upper)
        if rev_end > fwd_start:
            return seq_upper[fwd_start:rev_end]

    # Try reverse orientation: RC(REV) -> core_rc -> RC(FWD)
    rc_rev_start = _find_primer_at_start(seq_upper, rc_rev)
    if rc_rev_start != -1:
        rc_fwd_end = _find_primer_at_end(seq_upper, rc_fwd, min_start=rc_rev_start)
        if rc_fwd_end == -1:
            rc_fwd_end = len(seq_upper)
        if rc_fwd_end > rc_rev_start:
            # Extract core and reverse complement to get forward orientation
            core_rc = seq_upper[rc_rev_start:rc_fwd_end]
            return reverse_complement(core_rc)

    return None


def strip_gtail(sequence: str) -> str:
    """drop the terminal poly-G."""
    return sequence.rstrip('G')


def extract_bits_from_alignment(
    sequence: str,
    path: List[Tuple[StateType, int, Optional[str]]],
    pattern_length: int,
    start_base: str = 'A'
) -> Tuple[List[int], List[int]]:
    """pull 32 bits out of the viterbi alignment.

    encoding uses START_BASE = 'A' as the implicit start. for each pattern
    position i we keep the first and last base observed in match-state i;
    bit i is decoded from the transition (last_at_{i-1} -> first_at_i),
    with bit 0 using START_BASE for the left side.

    using first/last (rather than just one base) handles tdt carryover
    cleanly: if the hmm assigned a spurious base to an insert state, the
    last base at position i and the first base at position i+1 are still
    the right anchors for the transition and the bit decodes correctly.

    returns (bits, invalid_positions). bits has -1 wherever the transition
    couldn't be decoded; invalid_positions lists those indices.
    """
    # Track first and last observed base at each pattern position
    first_base_at_pos: Dict[int, str] = {}
    last_base_at_pos: Dict[int, str] = {}

    for t, state in enumerate(path):
        state_type, pos, _ = state

        if state_type == StateType.MATCH and pos is not None:
            observed = sequence[t]

            # Track first observation at this position
            if pos not in first_base_at_pos:
                first_base_at_pos[pos] = observed
            # Always update last (will end up with final observation)
            last_base_at_pos[pos] = observed

    # Decode transitions to get 32 bits
    bits = []
    invalid_positions = []

    # bit 0: START_BASE -> first base at position 0
    if 0 in first_base_at_pos:
        to_base = first_base_at_pos[0]
        transition = (start_base, to_base)
        if transition in TRANSITION_DECODE:
            bits.append(TRANSITION_DECODE[transition])
        else:
            bits.append(-1)
            invalid_positions.append(0)
    else:
        bits.append(-1)
        invalid_positions.append(0)

    # Bits 1-31: transitions between consecutive positions
    for i in range(pattern_length - 1):
        bit_idx = i + 1  # bit index (1 to 31)
        if i in last_base_at_pos and (i + 1) in first_base_at_pos:
            from_base = last_base_at_pos[i]
            to_base = first_base_at_pos[i + 1]

            transition = (from_base, to_base)
            if transition in TRANSITION_DECODE:
                bits.append(TRANSITION_DECODE[transition])
            else:
                # Invalid transition (G involved, or unexpected)
                bits.append(-1)
                invalid_positions.append(bit_idx)
        else:
            # Missing position (deletion in observed sequence)
            bits.append(-1)
            invalid_positions.append(bit_idx)

    return bits, invalid_positions


def get_scramble_sequence() -> List[int]:
    """sha256('babbage'), first 32 bits. matches binary_dna_codec."""
    digest = hashlib.sha256(b'babbage').digest()
    bits = []
    for byte in digest[:4]:  # 32 bits
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_float(bits: List[int]) -> Optional[float]:
    """descramble + repack 32 bits as an IEEE 754 float (big-endian).

    returns None if there are invalid bits (-1) - callers can substitute
    zeros and try anyway; we mark those as 'corrupted' downstream.
    """
    if len(bits) != 32:
        return None

    if any(b not in (0, 1) for b in bits):
        return None

    # descramble with SHA256 sequence
    scramble = get_scramble_sequence()
    descrambled = [bits[i] ^ scramble[i] for i in range(32)]

    # pack bits to bytes
    byte_vals = []
    for i in range(0, 32, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | descrambled[i + j]
        byte_vals.append(byte_val)

    # convert to float (big-endian)
    packed = bytes(byte_vals)
    value = struct.unpack('>f', packed)[0]

    return value


def categorize_float(value: Optional[float]) -> str:
    """one of: undecodable / nan / inf / valid."""
    if value is None:
        return "undecodable"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return "valid"


# minimum reads per file before multiprocessing kicks in
PARALLEL_THRESHOLD = 50_000


def _chunk_reads(reads, chunk_size):
    """split a list of reads into chunks."""
    for i in range(0, len(reads), chunk_size):
        yield reads[i:i + chunk_size]


def _process_read_chunk(args):
    """worker function for multiprocessing. each worker builds its own decoder."""
    chunk, patterns, min_phred = args
    hmm_decoder = ProfileHMMDecoder(patterns)
    return [
        process_read(read_id, seq, qual, patterns, hmm_decoder, min_phred=min_phred)
        for read_id, seq, qual in chunk
    ]


def _accumulate_result(result, stats):
    """accumulate a single ReadResult into PipelineStats."""
    stats.total_reads += 1

    if not result.passed_filter:
        if result.filter_reason == "low_phred":
            stats.failed_phred += 1
        elif result.filter_reason in ("too_short", "too_long"):
            stats.failed_length += 1
        elif result.filter_reason == "no_primers":
            stats.failed_no_primers += 1
        return

    stats.passed_filter += 1

    if result.predicted_constant:
        stats.constant_counts[result.predicted_constant] += 1
        stats.constant_confidences[result.predicted_constant].append(result.confidence)

        # accumulate bits for consensus (only 32-bit reads)
        if result.bits and len(result.bits) == 32:
            stats.constant_bits[result.predicted_constant].append(
                (result.bits, result.confidence)
            )

        if result.float_category == "valid" and result.decoded_float is not None:
            stats.constant_values[result.predicted_constant].append(result.decoded_float)
            stats.valid_floats += 1
        elif result.float_category == "corrupted" and result.decoded_float is not None:
            stats.constant_values[result.predicted_constant].append(result.decoded_float)
            stats.corrupted_floats += 1
        elif result.float_category == "nan":
            stats.nan_floats += 1
        elif result.float_category == "inf":
            stats.inf_floats += 1
        else:
            stats.undecodable += 1


def process_read(
    read_id: str,
    sequence: str,
    quality: str,
    patterns: Dict[str, str],
    hmm_decoder: ProfileHMMDecoder,
    min_phred: float = 7.0,
    min_length: int = 50,
    max_length: int = 500,
    max_invalid_bits: int = 5,
    max_bit_deviation: int = 4
) -> ReadResult:
    """one read through the whole pipeline.

    filter (phred, length, primers) -> identify which of the 17 patterns
    via forward score -> viterbi against the winner -> extract 32 bits
    -> descramble -> reconstruct float. categorizes the result and stuffs
    everything into a ReadResult for the consensus step later.
    """
    result = ReadResult(
        read_id=read_id,
        sequence_length=len(sequence),
        mean_phred=compute_mean_phred(quality),
        passed_filter=False
    )

    # pre-filter: phred quality
    if result.mean_phred < min_phred:
        result.filter_reason = "low_phred"
        return result

    # pre-filter: length
    if len(sequence) < min_length:
        result.filter_reason = "too_short"
        return result

    if len(sequence) > max_length:
        result.filter_reason = "too_long"
        return result

    # trim primers
    trimmed = trim_primers(sequence)
    if trimmed is None or len(trimmed) < 10:
        result.filter_reason = "no_primers"
        return result

    result.passed_filter = True

    # identify the pattern, then reconstruct the path for the winner only.
    # the viterbi backtrack (a python loop over T observations) is the bulk of
    # the per-pattern overhead and is discarded for 16 of 17 patterns, so we
    # score-only against all 17 (njit kernel, no backtrack) and backtrack just
    # the winner. ~30% off the decode loop (~20% off process_read end-to-end),
    # bit-identical to backtracking every pattern.
    best_score = -float('inf')
    best_pattern_name = None
    all_scores = {}
    for pattern_name, hmm in hmm_decoder.hmms.items():
        score = hmm.score_viterbi(trimmed)
        all_scores[pattern_name] = score
        if score > best_score:
            best_score = score
            best_pattern_name = pattern_name
    if best_score == -float('inf'):
        best_pattern_name = None
    best_path = (hmm_decoder.hmms[best_pattern_name].viterbi(trimmed)[0]
                 if best_pattern_name is not None else None)

    result.predicted_constant = best_pattern_name
    result.all_scores = all_scores if all_scores else None

    # compute confidence as softmax of finite scores
    if all_scores:
        scores_arr = np.array(list(all_scores.values()))
        finite_mask = np.isfinite(scores_arr)
        if np.any(finite_mask):
            finite_scores = scores_arr[finite_mask]
            max_score = np.max(finite_scores)
            exp_scores = np.exp(finite_scores - max_score)
            softmax = exp_scores / np.sum(exp_scores)
            result.confidence = float(np.max(softmax))

        # margin: difference between top-2 finite scores
        finite_sorted = sorted((s for s in all_scores.values() if np.isfinite(s)), reverse=True)
        if len(finite_sorted) >= 2:
            result.margin = finite_sorted[0] - finite_sorted[1]

    # === Extract bits from best path (already computed above) ===
    if best_pattern_name and best_path:
        pattern = patterns.get(best_pattern_name, "")

        # extract bits from alignment using best_path from pattern identification
        bits, invalid_positions = extract_bits_from_alignment(
            trimmed, best_path, len(pattern)
        )

        result.bits = bits
        result.n_valid_bits = sum(1 for b in bits if b >= 0)
        result.n_invalid_bits = len(invalid_positions)
        result.invalid_positions = invalid_positions

        # attempt float reconstruction
        # allow up to max_invalid_bits (substitute 0), useful for distribution analysis
        # higher threshold accommodates noisy nanopore reads
        if result.n_invalid_bits <= max_invalid_bits and abs(len(bits) - 32) <= max_bit_deviation:
            # substitute 0 for invalid bits
            reconstructed = [b if b >= 0 else 0 for b in bits]

            # pad or truncate to 32 bits
            if len(reconstructed) < 32:
                # pad with zeros at the end
                reconstructed = reconstructed + [0] * (32 - len(reconstructed))
            elif len(reconstructed) > 32:
                # truncate
                reconstructed = reconstructed[:32]

            result.decoded_float = bits_to_float(reconstructed)
            result.float_category = categorize_float(result.decoded_float)

            # mark as corrupted if we had invalid bits
            if result.n_invalid_bits > 0:
                result.float_category = "corrupted"
        else:
            result.float_category = "undecodable"
    else:
        result.float_category = "undecodable"

    return result


def detect_constants(
    stats: PipelineStats,
    min_count: int = 5,
    min_confidence: float = 0.5
) -> Dict[str, Dict]:
    """which constants do we think are present?

    keep anything that hit min_count reads with mean confidence above
    min_confidence. returns a dict of name -> stats blob (count,
    weighted_score, mean_confidence, value list, mean/std).
    """
    detected = {}

    for name, confidences in stats.constant_confidences.items():
        if not confidences:
            continue

        count = stats.constant_counts[name]
        if count < min_count:
            continue

        mean_conf = sum(confidences) / len(confidences)
        if mean_conf < min_confidence:
            continue

        values = stats.constant_values[name]
        weighted_score = sum(confidences)

        detected[name] = {
            'count': count,
            'weighted_score': weighted_score,
            'mean_confidence': mean_conf,
            'values': values,
            'mean_value': sum(values) / len(values) if values else None,
            'std_value': (
                (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5
                if len(values) > 1 else 0
            )
        }

    return detected


def run_consensus(stats: PipelineStats, detected: Dict) -> Dict[str, ConsensusResult]:
    """bit-level consensus, one ConsensusResult per detected constant.

    confidence-weighted majority vote at each of the 32 bit positions, then
    reconstruct the float. weighting matters more than you'd think - even
    the worst sign-bit position (88.6% per-read accuracy in our data)
    decodes correctly here because high-confidence reads pull the vote.
    """
    results = {}

    for name in detected:
        bit_entries = stats.constant_bits.get(name, [])
        if not bit_entries:
            continue

        # confidence-weighted majority vote per bit
        bit_votes = np.zeros(32)
        bit_weights = np.zeros(32)

        for bits, confidence in bit_entries:
            for i, b in enumerate(bits[:32]):
                if b >= 0:  # valid bit
                    bit_votes[i] += b * confidence
                    bit_weights[i] += confidence

        consensus_bits = []
        bit_confidences = []
        for i in range(32):
            if bit_weights[i] > 0:
                vote_frac = bit_votes[i] / bit_weights[i]
                consensus_bits.append(1 if vote_frac > 0.5 else 0)
                # confidence = how far from 0.5 (higher = more certain)
                bit_confidences.append(abs(vote_frac - 0.5) * 2)
            else:
                consensus_bits.append(0)
                bit_confidences.append(0.0)

        decoded = bits_to_float(consensus_bits)
        confs = stats.constant_confidences.get(name, [])
        mean_conf = sum(confs) / len(confs) if confs else 0.0

        results[name] = ConsensusResult(
            name=name,
            n_reads=len(bit_entries),
            bits=consensus_bits,
            bit_confidences=bit_confidences,
            decoded_float=decoded,
            mean_read_confidence=mean_conf,
        )

    return results


def run_pipeline(
    input_folder: Path,
    output_folder: Optional[Path] = None,
    detailed: bool = False,
    min_phred: float = 7.0,
    verbose: bool = True,
    workers: int = 0,
) -> PipelineStats:
    """top-level entry. read fastqs, decode every read, run consensus,
    write reports. workers=0 picks one based on cpu count; workers=1 forces
    serial; large files (>= PARALLEL_THRESHOLD reads) get split into
    chunks across a process pool."""
    if output_folder is None:
        output_folder = input_folder

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # get patterns
    patterns = {name: pattern for name, pattern in CONSTANTS}

    # find FASTQ files
    fastq_files = find_fastq_files(input_folder)

    if not fastq_files:
        logger.warning(f"no FASTQ files found in {input_folder}")
        return PipelineStats()

    if verbose:
        logger.info(f"found {len(fastq_files)} FASTQ file(s)")
        for f in fastq_files:
            logger.info(f"  - {f.name}")

    # resolve worker count
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    # initialize decoder for two-pass decoding
    if verbose:
        logger.info(f"initializing decoder for {len(patterns)} constants...")

    hmm_decoder = ProfileHMMDecoder(patterns)

    if verbose:
        logger.info("  ProfileHMMDecoder: ready (pattern ID + Viterbi bit extraction)")
        if workers > 1:
            logger.info(f"  parallel mode: {workers} workers (threshold: {PARALLEL_THRESHOLD} reads)")

    stats = PipelineStats()
    detailed_results: List[ReadResult] = []

    # process each file
    for fastq_file in fastq_files:
        if verbose:
            logger.info(f"processing {fastq_file.name}...")

        all_reads = list(parse_fastq(fastq_file))

        if workers > 1 and len(all_reads) >= PARALLEL_THRESHOLD:
            # parallel: chunk reads across workers
            if verbose:
                logger.info(f"  {len(all_reads)} reads, using {workers} workers...")
            chunk_size = max(1000, len(all_reads) // workers)
            chunks = list(_chunk_reads(all_reads, chunk_size))
            with Pool(workers) as pool:
                chunk_results = pool.map(
                    _process_read_chunk,
                    [(c, patterns, min_phred) for c in chunks]
                )
            file_results = [r for batch in chunk_results for r in batch]
        else:
            # serial: process reads one by one
            file_results = []
            for read_id, sequence, quality in all_reads:
                file_results.append(
                    process_read(read_id, sequence, quality, patterns,
                                 hmm_decoder=hmm_decoder, min_phred=min_phred)
                )

        for result in file_results:
            _accumulate_result(result, stats)
            if detailed:
                detailed_results.append(result)

        if verbose:
            logger.info(f"  processed {len(all_reads)} reads total")

    # detect constants
    detected = detect_constants(stats)

    # run bit-level consensus for detected constants
    stats.consensus_results = run_consensus(stats, detected)

    if verbose and stats.consensus_results:
        n_decoded = sum(1 for cr in stats.consensus_results.values() if cr.decoded_float is not None)
        logger.info(f"consensus: {n_decoded}/{len(stats.consensus_results)} constants decoded")

    # write output files
    write_detected_constants(output_folder / "detected_constants.txt", detected, patterns)
    write_decoding_report(output_folder / "decoding_report.txt", stats, detected)
    write_consensus_results(output_folder / "consensus_results.txt", stats.consensus_results)
    write_value_distributions(output_folder / "value_distributions.tsv", stats)

    if detailed:
        write_detailed_reads(output_folder / "detailed_reads.tsv", detailed_results)

    if verbose:
        logger.info("pipeline complete!")
        logger.info(f"  total reads: {stats.total_reads}")
        logger.info(f"  passed filter: {stats.passed_filter}")
        logger.info(f"  valid floats decoded: {stats.valid_floats}")
        logger.info(f"  corrupted floats: {stats.corrupted_floats}")
        logger.info(f"  constants detected: {len(detected)}")
        logger.info(f"output written to {output_folder}")

    return stats


def write_detected_constants(filepath: Path, detected: Dict, patterns: Dict[str, str]):
    """detected_constants.txt - short list of what we think we found."""
    try:
        with open(filepath, 'w') as f:
            f.write("Detected constants\n")
            f.write("-" * 60 + "\n\n")

            if not detected:
                f.write("No constants detected with sufficient confidence.\n")
                return

            f.write(f"{'Constant':<30} {'Count':>8} {'Confidence':>12} {'Mean Value':>15}\n")
            f.write("-" * 70 + "\n")

            for name, info in sorted(detected.items(), key=lambda x: -x[1]['count']):
                mean_val = info['mean_value']
                mean_str = f"{mean_val:.6g}" if mean_val is not None else "N/A"
                f.write(f"{name:<30} {info['count']:>8} {info['mean_confidence']:>12.3f} {mean_str:>15}\n")

            f.write("\n\nDetailed statistics\n")
            f.write("-" * 60 + "\n\n")

            for name, info in sorted(detected.items(), key=lambda x: -x[1]['count']):
                f.write(f"{name}:\n")
                f.write(f"  Pattern: {patterns.get(name, 'unknown')}\n")
                f.write(f"  Read count: {info['count']}\n")
                f.write(f"  Weighted score: {info['weighted_score']:.2f}\n")
                f.write(f"  Mean confidence: {info['mean_confidence']:.4f}\n")
                if info['values']:
                    f.write(f"  Decoded values: {len(info['values'])}\n")
                    f.write(f"  Mean value: {info['mean_value']:.6g}\n")
                    f.write(f"  Std value: {info['std_value']:.6g}\n")
                f.write("\n")
    except IOError as e:
        logger.error(f"failed to write {filepath}: {e}")
        raise PipelineIOError(f"cannot write file: {filepath}") from e


def write_consensus_results(filepath: Path, consensus: Dict[str, ConsensusResult]):
    """consensus_results.txt - per-constant consensus floats and bit confs."""
    try:
        with open(filepath, 'w') as f:
            f.write("Bit-level consensus results\n")
            f.write("-" * 60 + "\n\n")

            if not consensus:
                f.write("No consensus results.\n")
                return

            f.write(f"{'Constant':<25} {'Reads':>6} {'Value':>15} {'Min bit conf':>13}\n")
            f.write("-" * 65 + "\n")

            for name, cr in sorted(consensus.items(), key=lambda x: -x[1].n_reads):
                val_str = f"{cr.decoded_float:.6g}" if cr.decoded_float is not None else "None"
                min_conf = min(cr.bit_confidences) if cr.bit_confidences else 0
                f.write(f"{name:<25} {cr.n_reads:>6} {val_str:>15} {min_conf:>13.3f}\n")

            f.write(f"\n\nPer-constant detail\n")
            f.write("-" * 60 + "\n\n")

            for name, cr in sorted(consensus.items(), key=lambda x: -x[1].n_reads):
                f.write(f"{name}:\n")
                f.write(f"  Reads used: {cr.n_reads}\n")
                f.write(f"  Mean read confidence: {cr.mean_read_confidence:.4f}\n")
                val_str = f"{cr.decoded_float:.6g}" if cr.decoded_float is not None else "None"
                f.write(f"  Consensus value: {val_str}\n")
                f.write(f"  Bits: {''.join(str(b) for b in cr.bits)}\n")

                # flag weak bits (confidence < 0.5)
                weak = [i for i, c in enumerate(cr.bit_confidences) if c < 0.5]
                if weak:
                    f.write(f"  Weak bits (conf < 0.5): {weak}\n")

                f.write(f"  Min/mean/max bit confidence: "
                        f"{min(cr.bit_confidences):.3f} / "
                        f"{sum(cr.bit_confidences)/32:.3f} / "
                        f"{max(cr.bit_confidences):.3f}\n")
                f.write("\n")
    except IOError as e:
        logger.error(f"failed to write {filepath}: {e}")
        raise PipelineIOError(f"cannot write file: {filepath}") from e


def write_decoding_report(filepath: Path, stats: PipelineStats, detected: Dict):
    """decoding_report.txt - counts, per-constant tallies, the whole spread."""
    try:
        with open(filepath, 'w') as f:
            f.write("Nanopore decoding report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Read statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total reads processed: {stats.total_reads}\n")
            f.write(f"Passed filter: {stats.passed_filter}\n")
            f.write(f"Failed (low Phred): {stats.failed_phred}\n")
            f.write(f"Failed (length): {stats.failed_length}\n")
            f.write(f"Failed (no primers): {stats.failed_no_primers}\n")

            if stats.total_reads > 0:
                pass_rate = stats.passed_filter / stats.total_reads * 100
                f.write(f"Pass rate: {pass_rate:.1f}%\n")

            f.write("\n")
            f.write("Decoding statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"Valid floats decoded: {stats.valid_floats}\n")
            f.write(f"Corrupted floats (up to 5 bad bits): {stats.corrupted_floats}\n")
            f.write(f"NaN values: {stats.nan_floats}\n")
            f.write(f"Inf values: {stats.inf_floats}\n")
            f.write(f"Undecodable: {stats.undecodable}\n")

            if stats.passed_filter > 0:
                decode_rate = (stats.valid_floats + stats.corrupted_floats) / stats.passed_filter * 100
                f.write(f"Decode rate (valid + corrupted): {decode_rate:.1f}%\n")

            f.write("\n")
            f.write("Per-constant breakdown\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Constant':<30} {'Count':>8} {'Valid':>8} {'Confidence':>12}\n")
            f.write("-" * 60 + "\n")

            for name in sorted(stats.constant_counts.keys()):
                count = stats.constant_counts[name]
                n_valid = len(stats.constant_values[name])
                confidences = stats.constant_confidences[name]
                mean_conf = sum(confidences) / len(confidences) if confidences else 0
                f.write(f"{name:<30} {count:>8} {n_valid:>8} {mean_conf:>12.3f}\n")

            f.write("\n")
            f.write("Detection summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Constants detected: {len(detected)}\n")
            if detected:
                f.write("Detected: " + ", ".join(sorted(detected.keys())) + "\n")

            # consensus summary
            if stats.consensus_results:
                f.write("\n")
                f.write("Consensus decoding\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Constant':<25} {'Reads':>6} {'Value':>15} {'Min bit conf':>13}\n")
                f.write("-" * 65 + "\n")
                for name, cr in sorted(stats.consensus_results.items(),
                                       key=lambda x: -x[1].n_reads):
                    val_str = f"{cr.decoded_float:.6g}" if cr.decoded_float is not None else "None"
                    min_conf = min(cr.bit_confidences)
                    f.write(f"{name:<25} {cr.n_reads:>6} {val_str:>15} {min_conf:>13.3f}\n")
    except IOError as e:
        logger.error(f"failed to write {filepath}: {e}")
        raise PipelineIOError(f"cannot write file: {filepath}") from e


def write_value_distributions(filepath: Path, stats: PipelineStats):
    """value_distributions.tsv - every decoded float per constant, for plotting (key/type/value record)."""
    try:
        data = {
            'per_constant': {},
            'summary': {
                'total_valid': stats.valid_floats,
                'total_corrupted': stats.corrupted_floats,
                'total_nan': stats.nan_floats,
                'total_inf': stats.inf_floats,
                'total_undecodable': stats.undecodable
            }
        }

        for name in stats.constant_counts.keys():
            values = stats.constant_values[name]
            entry = {
                'count': stats.constant_counts[name],
                'valid_values': values,
                'n_valid': len(values),
                'mean': sum(values) / len(values) if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None
            }
            # add consensus value if available
            cr = stats.consensus_results.get(name)
            if cr:
                entry['consensus_value'] = cr.decoded_float
                entry['consensus_reads'] = cr.n_reads
                entry['consensus_min_bit_confidence'] = min(cr.bit_confidences)
            data['per_constant'][name] = entry

        write_record(filepath, data)
    except IOError as e:
        logger.error(f"failed to write {filepath}: {e}")
        raise PipelineIOError(f"cannot write file: {filepath}") from e


def write_detailed_reads(filepath: Path, results: List[ReadResult]):
    """detailed_results.tsv - one row per read, opt-in via --detailed."""
    try:
        with open(filepath, 'w') as f:
            # header
            f.write("\t".join([
                "read_id", "seq_length", "mean_phred", "passed_filter", "filter_reason",
                "predicted_constant", "confidence", "margin",
                "n_valid_bits", "n_invalid_bits", "decoded_float", "float_category"
            ]) + "\n")

            for r in results:
                f.write("\t".join([
                    r.read_id,
                    str(r.sequence_length),
                    f"{r.mean_phred:.1f}",
                    str(r.passed_filter),
                    r.filter_reason or "",
                    r.predicted_constant or "",
                    f"{r.confidence:.4f}" if r.confidence else "",
                    f"{r.margin:.4f}" if r.margin else "",
                    str(r.n_valid_bits),
                    str(r.n_invalid_bits),
                    f"{r.decoded_float:.6g}" if r.decoded_float is not None else "",
                    r.float_category
                ]) + "\n")
    except IOError as e:
        logger.error(f"failed to write {filepath}: {e}")
        raise PipelineIOError(f"cannot write file: {filepath}") from e


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Nanopore decoding pipeline for TdT-synthesized DNA"
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing FASTQ files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output folder (default: input folder)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed per-read TSV output"
    )
    parser.add_argument(
        "--min-phred",
        type=float,
        default=7.0,
        help="Minimum mean Phred score (default: 7.0)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: auto-detect, 1 = serial)"
    )

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        logger.error(f"{args.input_folder} is not a directory")
        sys.exit(1)

    stats = run_pipeline(
        input_folder=args.input_folder,
        output_folder=args.output,
        detailed=args.detailed,
        min_phred=args.min_phred,
        verbose=not args.quiet,
        workers=args.workers,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
