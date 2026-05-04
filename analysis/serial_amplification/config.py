"""
configuration and data loading for section 2.11 - serial re-amplification.

12 generations of PCR re-amplification from TdT-synthesized DNA.
barcode07 (gen 1) through barcode18 (gen 12).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ANALYSIS_DIR)

from nanopore_pipeline import (
    parse_fastq, compute_mean_phred, trim_primers, process_read,
    ReadResult, PipelineStats, find_fastq_files,
)
from profile_hmm import ProfileHMMDecoder
from error_analysis import (
    extract_homopolymer_runs, align_homopolymer_runs, AlignmentErrorStats,
)
from babbage_sequence_list import (
    CONSTANTS, DEFAULT_FORWARD_PRIMER, DEFAULT_REVERSE_PRIMER,
)
from binary_dna_codec import decode_dna
from nanopore_pipeline import bits_to_float

# data path (merged feb26 + feb27 runs)
FASTQ_DIR = os.path.join(
    PROJECT_ROOT, 'raw_reads', '2026_02_26_27_merged', 'fastq_pass',
)

# pre-decoded TSV directory (populated by aws_decode.sh)
TSV_DIR = os.path.join(SCRIPT_DIR, 'per_barcode_reads')

N_GENERATIONS = 12
BARCODES = [f'barcode{i:02d}' for i in range(7, 19)]
GENERATIONS = {bc: i + 1 for i, bc in enumerate(BARCODES)}

# qubit concentrations [ng/uL] per generation
QUBIT_NG_UL = {
    1: 11.3, 2: 13.0, 3: 10.8, 4: 15.6, 5: 34.4, 6: 39.2,
    7: 58.2, 8: 46.8, 9: 35.0, 10: 39.0, 11: 38.6, 12: 45.4,
}

# target amplicon
TARGET_MIN = 200
TARGET_MAX = 300
EXPECTED_LENGTH = 250

# codeword patterns and ground truth
PATTERNS = {name: pattern for name, pattern in CONSTANTS}
GROUND_TRUTH = {}
GROUND_TRUTH_BITS = {}
for _name, _pattern in CONSTANTS:
    _bits = decode_dna(_pattern)
    GROUND_TRUTH[_name] = bits_to_float(_bits)
    GROUND_TRUTH_BITS[_name] = list(_bits)

# primers
FWD_PRIMER = DEFAULT_FORWARD_PRIMER.upper()
REV_PRIMER = DEFAULT_REVERSE_PRIMER.upper()


def _reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(c, 'N') for c in reversed(seq))


RC_FWD = _reverse_complement(FWD_PRIMER)
RC_REV = _reverse_complement(REV_PRIMER)


@dataclass
class RawRead:
    """minimal parsed read."""
    read_id: str
    sequence: str
    quality: str
    length: int
    mean_phred: float


@dataclass
class DecodedRead:
    """read processed through the HMM pipeline."""
    raw: RawRead
    result: ReadResult
    trimmed: Optional[str] = None


@dataclass
class GenerationData:
    """all data for one generation."""
    generation: int
    barcode: str
    raw_reads: List[RawRead] = field(default_factory=list)
    decoded_reads: List[DecodedRead] = field(default_factory=list)


def load_raw_reads(barcode, quick=False, quick_n=500):
    """parse FASTQ files for a barcode. no HMM decoding."""
    barcode_dir = Path(os.path.join(FASTQ_DIR, barcode))
    if not barcode_dir.is_dir():
        return []
    fastq_files = find_fastq_files(barcode_dir)
    reads = []
    for fq in fastq_files:
        for read_id, sequence, quality in parse_fastq(fq):
            if quick and len(reads) >= quick_n:
                break
            reads.append(RawRead(
                read_id=read_id,
                sequence=sequence,
                quality=quality,
                length=len(sequence),
                mean_phred=compute_mean_phred(quality),
            ))
        if quick and len(reads) >= quick_n:
            break
    return reads


def load_decoded_reads(barcode, decoder, quick=False, quick_n=500):
    """parse and decode reads through HMM pipeline."""
    raw_reads = load_raw_reads(barcode, quick=quick, quick_n=quick_n)
    decoded = []
    for r in raw_reads:
        result = process_read(r.read_id, r.sequence, r.quality, PATTERNS, decoder)
        trimmed = trim_primers(r.sequence) if result.passed_filter else None
        decoded.append(DecodedRead(raw=r, result=result, trimmed=trimmed))
    return decoded


def _parse_tsv_float(s):
    """parse float from TSV field, handling nan/inf/empty."""
    if not s:
        return None
    if s == 'nan':
        return float('nan')
    if s == 'inf':
        return float('inf')
    if s == '-inf':
        return float('-inf')
    return float(s)


def _tsvs_available():
    """check if pre-decoded TSVs exist for all 12 barcodes."""
    if not os.path.isdir(TSV_DIR):
        return False
    return all(
        os.path.isfile(os.path.join(TSV_DIR, f'{bc}.tsv'))
        for bc in BARCODES
    )


def _load_decoded_from_tsv(barcode):
    """load pre-decoded reads from per-barcode TSV.

    reconstructs DecodedRead objects without needing HMM or raw FASTQ.
    raw.sequence and raw.quality are empty strings (not needed for analysis).
    """
    tsv_path = os.path.join(TSV_DIR, f'{barcode}.tsv')
    if not os.path.isfile(tsv_path):
        return None

    decoded = []
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 14:
                continue

            (read_id, length, mean_phred, passed_filter, filter_reason,
             predicted_constant, confidence, margin,
             n_valid_bits, n_invalid_bits, decoded_float, float_category,
             bits_str, trimmed) = fields[:14]

            bits = None
            if bits_str:
                bits = [int(b) for b in bits_str.split(',')]

            raw = RawRead(
                read_id=read_id,
                sequence='',
                quality='',
                length=int(length),
                mean_phred=float(mean_phred),
            )

            result = ReadResult(
                read_id=read_id,
                sequence_length=int(length),
                mean_phred=float(mean_phred),
                passed_filter=(passed_filter == 'True'),
                filter_reason=filter_reason or None,
                predicted_constant=predicted_constant or None,
                confidence=float(confidence),
                margin=float(margin),
                bits=bits,
                n_valid_bits=int(n_valid_bits),
                n_invalid_bits=int(n_invalid_bits),
                decoded_float=_parse_tsv_float(decoded_float),
                float_category=float_category,
            )

            decoded.append(DecodedRead(
                raw=raw,
                result=result,
                trimmed=trimmed or None,
            ))

    return decoded


def load_all_generations(decode=False, quick=False, quick_n=500):
    """load data for all 12 generations.

    if decode=True and pre-decoded TSVs exist (from aws_decode.sh),
    loads instantly from TSVs instead of running HMM decoding.

    returns dict: generation_number -> GenerationData
    """
    # fast path: pre-decoded TSVs (skip HMM entirely)
    if decode and not quick and _tsvs_available():
        print("  loading from pre-decoded TSVs...", flush=True)
        data = {}
        for barcode in BARCODES:
            gen = GENERATIONS[barcode]
            gd = GenerationData(generation=gen, barcode=barcode)
            gd.decoded_reads = _load_decoded_from_tsv(barcode)
            gd.raw_reads = [d.raw for d in gd.decoded_reads]
            print(f"  gen {gen:2d} ({barcode}): "
                  f"{len(gd.raw_reads)} reads (from TSV)", flush=True)
            data[gen] = gd
        return data

    decoder = ProfileHMMDecoder(PATTERNS) if decode else None
    data = {}
    for barcode in BARCODES:
        gen = GENERATIONS[barcode]
        gd = GenerationData(generation=gen, barcode=barcode)
        if decode:
            gd.decoded_reads = load_decoded_reads(
                barcode, decoder, quick=quick, quick_n=quick_n)
            gd.raw_reads = [d.raw for d in gd.decoded_reads]
        else:
            gd.raw_reads = load_raw_reads(
                barcode, quick=quick, quick_n=quick_n)
        print(f"  gen {gen:2d} ({barcode}): {len(gd.raw_reads)} reads", flush=True)
        data[gen] = gd
    return data
