"""
shared infrastructure for paper-figure analyses.

provides DataLoader, constants, plotting helpers, and common metrics
used across the analysis scripts under analysis/.
"""

import os
import sys
import math
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

import logging
logging.getLogger('fontTools').setLevel(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nanopore_pipeline import (
    parse_fastq, compute_mean_phred, trim_primers, process_read,
    _accumulate_result, detect_constants, run_consensus,
    bits_to_float, extract_bits_from_alignment,
    ReadResult, PipelineStats, ConsensusResult,
    find_fastq_files,
)
from profile_hmm import (
    ProfileHMM, ProfileHMMDecoder, ProfileHMMParameters, StateType,
    BASE_TO_IDX,
)
from error_analysis import (
    extract_homopolymer_runs, align_homopolymer_runs, AlignmentErrorStats,
)
from babbage_sequence_list import CONSTANTS
from binary_dna_codec import decode_dna

# publication defaults
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

PHI = (1 + 5 ** 0.5) / 2
WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

# data paths - merged reads from feb24 + feb25 runs
DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_reads')
RUNS = {
    'merged': os.path.join(DATA_DIR, 'merged', 'fastq_pass'),
}

BARCODE_CONDITIONS = {
    'barcode01': {'polymerase': 'SpeedStar HS', 'group': 'polymerase', 'cycles': 25},
    'barcode02': {'polymerase': 'PrimeStar GXL', 'group': 'polymerase', 'cycles': 25},
    'barcode03': {'polymerase': 'KAPA HiFi', 'group': 'polymerase', 'cycles': 25},
    'barcode04': {'polymerase': 'PrimeStar GXL', 'group': 'cycle_titration', 'cycles': 15},
    'barcode05': {'polymerase': 'PrimeStar GXL', 'group': 'cycle_titration', 'cycles': 25},
    'barcode06': {'polymerase': 'PrimeStar GXL', 'group': 'cycle_titration', 'cycles': 40},
}
POLYMERASE_BARCODES = ['barcode01', 'barcode02', 'barcode03']
CYCLE_BARCODES = ['barcode04', 'barcode05', 'barcode06']

PATTERNS = {name: pattern for name, pattern in CONSTANTS}

# ground truth float values (decoded from reference patterns)
GROUND_TRUTH = {}
for _name, _pattern in CONSTANTS:
    _bits = decode_dna(_pattern)
    GROUND_TRUTH[_name] = bits_to_float(_bits)

# ground truth bits per constant (for bit-level analyses)
GROUND_TRUTH_BITS = {}
for _name, _pattern in CONSTANTS:
    GROUND_TRUTH_BITS[_name] = list(decode_dna(_pattern))


def hamming_distance(pat1, pat2):
    """hamming distance between two equal-length DNA patterns."""
    return sum(c1 != c2 for c1, c2 in zip(pat1, pat2))


# default per-base HP params (fitted from real nanopore data)
DEFAULT_HP_EXPECTED = {'A': 3.54, 'C': 3.38, 'T': 4.35}
DEFAULT_HP_SCALE = {'A': 1.83, 'C': 1.61, 'T': 2.55}


@dataclass
class ParsedRead:
    """processed read with all derived data."""
    read_id: str
    sequence: str
    quality: str
    mean_phred: float
    seq_length: int
    result: ReadResult
    trimmed: Optional[str] = None
    run_lengths: List[Tuple[str, int, int]] = field(default_factory=list)


@dataclass
class BarcodeData:
    """all data for one (run, barcode) combination."""
    run: str
    barcode: str
    reads: List[ParsedRead] = field(default_factory=list)
    stats: PipelineStats = field(default_factory=PipelineStats)


class DataLoader:
    """loads FASTQ data, runs pipeline, caches results."""

    def __init__(self, quick=False, quick_n=500):
        self._cache: Dict[Tuple[str, str], BarcodeData] = {}
        self._hmm_decoder = None
        self.quick = quick
        self.quick_n = quick_n

    def _get_decoder(self) -> ProfileHMMDecoder:
        if self._hmm_decoder is None:
            self._hmm_decoder = ProfileHMMDecoder(PATTERNS)
        return self._hmm_decoder

    def load(self, run: str, barcode: str) -> BarcodeData:
        """load and process reads for one (run, barcode). cached."""
        key = (run, barcode)
        if key in self._cache:
            return self._cache[key]

        data = BarcodeData(run=run, barcode=barcode)
        run_dir = RUNS.get(run)
        if run_dir is None or not os.path.isdir(run_dir):
            self._cache[key] = data
            return data

        barcode_dir = Path(os.path.join(run_dir, barcode))
        if not barcode_dir.is_dir():
            self._cache[key] = data
            return data

        fastq_files = find_fastq_files(barcode_dir)
        if not fastq_files:
            self._cache[key] = data
            return data

        decoder = self._get_decoder()
        stats = PipelineStats()
        reads = []
        count = 0

        for fq in fastq_files:
            for read_id, sequence, quality in parse_fastq(fq):
                if self.quick and count >= self.quick_n:
                    break

                phred = compute_mean_phred(quality)
                result = process_read(
                    read_id, sequence, quality, PATTERNS, decoder
                )
                _accumulate_result(result, stats)

                # get trimmed and run lengths
                trimmed = trim_primers(sequence) if result.passed_filter else None
                run_lengths = []
                if trimmed:
                    stripped = trimmed.rstrip('G')
                    if stripped:
                        run_lengths = extract_homopolymer_runs(stripped)

                reads.append(ParsedRead(
                    read_id=read_id,
                    sequence=sequence,
                    quality=quality,
                    mean_phred=phred,
                    seq_length=len(sequence),
                    result=result,
                    trimmed=trimmed,
                    run_lengths=run_lengths,
                ))
                count += 1

            if self.quick and count >= self.quick_n:
                break

        # finalize stats
        detected = detect_constants(stats)
        stats.consensus_results = run_consensus(stats, detected)

        data.reads = reads
        data.stats = stats
        self._cache[key] = data
        return data

    def load_all(self):
        """load all (run, barcode) combinations."""
        for run in RUNS:
            for barcode in BARCODE_CONDITIONS:
                print(f"  loading {run}/{barcode}...", end='', flush=True)
                bd = self.load(run, barcode)
                print(f" {len(bd.reads)} reads")


def label(barcode: str) -> str:
    """human-readable label for a barcode."""
    info = BARCODE_CONDITIONS[barcode]
    return f"{info['polymerase']} ({info['cycles']}c)"


def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved {os.path.basename(path)}")


def copy_script(script_path, dest_dir):
    """copy the generating script alongside its figures."""
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(script_path, dest_dir)


def compute_accuracy(data: BarcodeData) -> float:
    """fraction of passed reads with correct predicted constant."""
    passed = [r for r in data.reads if r.result.passed_filter and r.result.predicted_constant]
    if not passed:
        return 0.0
    correct = sum(
        1 for r in passed
        if r.result.decoded_float is not None
        and r.result.predicted_constant in GROUND_TRUTH
        and r.result.decoded_float == GROUND_TRUTH[r.result.predicted_constant]
    )
    return correct / len(passed)


def compute_error_stats(data: BarcodeData) -> AlignmentErrorStats:
    """compute alignment error stats for all reads with predicted constant."""
    global_stats = AlignmentErrorStats()
    for r in data.reads:
        if not r.result.passed_filter or not r.result.predicted_constant:
            continue
        if r.trimmed is None:
            continue
        ref_pattern = PATTERNS.get(r.result.predicted_constant)
        if ref_pattern is None:
            continue
        obs_stripped = r.trimmed.rstrip('G')
        ref_stripped = ref_pattern
        if not obs_stripped or not ref_stripped:
            continue
        obs_runs = extract_homopolymer_runs(obs_stripped)
        ref_runs = extract_homopolymer_runs(ref_stripped)
        _, read_stats = align_homopolymer_runs(obs_runs, ref_runs)
        # accumulate
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
    return global_stats


def error_rates(stats: AlignmentErrorStats) -> Dict[str, float]:
    """compute sub/ins/del rates from alignment stats."""
    total = stats.total_reference_bases
    if total == 0:
        return {'sub': 0.0, 'ins': 0.0, 'del': 0.0}
    return {
        'sub': stats.substitutions / total,
        'ins': stats.insertions / total,
        'del': stats.deletions / total,
    }


def consensus_correct(data: BarcodeData) -> int:
    """how many of 17 consensus results match ground truth."""
    correct = 0
    for name, cr in data.stats.consensus_results.items():
        if cr.decoded_float is not None and name in GROUND_TRUTH:
            gt = GROUND_TRUTH[name]
            if math.isnan(gt) and math.isnan(cr.decoded_float):
                correct += 1
            elif cr.decoded_float == gt:
                correct += 1
    return correct


def add_quick_arg(parser):
    """add --quick and --quick-n arguments to an argparse parser."""
    parser.add_argument('--quick', action='store_true',
                        help='quick mode: limit reads per barcode')
    parser.add_argument('--quick-n', type=int, default=500,
                        help='reads per barcode in quick mode (default: 500)')
