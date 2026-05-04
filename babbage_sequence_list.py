"""
tdt synthesis simulator + the constant table.

tdt is a template-independent polymerase that tacks homopolymer runs onto
the 3' end with base-specific kinetics. we model it as a per-base
negative binomial extension whose parameters come from the literature
(kuznetsova 2022 pmc9348634 for k_obs and processivity, deibel & coleman
1980 for Km of dGTP/dATP). dTTP and dCTP Km are guesses; the synthesis
table is robust to small changes anyway.

things the model captures, roughly: dGTP is processive (~6 nt/binding
with Mg2+); others are distributive (~1-2 nt). rate ordering is dGTP >
dTTP > dATP > dCTP. Mn2+/Co2+ make everything ~10x more processive.
poly-G tail at the end because tdt likes dGTP.

CONSTANTS at the bottom is the actual list of patterns used in the paper.
"""

import random
import math
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from statistics import mean, stdev
from enum import Enum

from logging_config import get_logger

logger = get_logger(__name__)


VALID_BASES = frozenset('ACGT')


class MetalIon(Enum):
    """divalent cofactor. Mg2+ is the default; Mn2+/Co2+ are more processive."""
    MG = "Mg2+"
    MN = "Mn2+"
    CO = "Co2+"   # similar to Mn2+, not used in the paper


@dataclass
class NucleotideKinetics:
    """per-nucleotide kinetics. lit values; see module docstring for sources."""
    k_obs_mg: float      # rate constant with Mg²⁺ (s⁻¹)
    k_obs_mn: float      # rate constant with Mn²⁺ (s⁻¹)
    processivity_mg: int  # max nucleotides per binding (Mg²⁺)
    processivity_mn: int  # max nucleotides per binding (Mn²⁺)
    km: float            # Michaelis constant (μM)


# literature-derived kinetic parameters (Kuznetsova et al. 2022, Deibel & Coleman 1980)
NUCLEOTIDE_KINETICS: Dict[str, NucleotideKinetics] = {
    'G': NucleotideKinetics(
        k_obs_mg=0.07,    # fastest incorporation
        k_obs_mn=0.30,
        processivity_mg=6, # processive mode
        processivity_mn=8,
        km=120.0          # lowest Km = highest affinity
    ),
    'T': NucleotideKinetics(
        k_obs_mg=0.03,
        k_obs_mn=0.20,
        processivity_mg=2, # distributive
        processivity_mn=4,
        km=200.0           # estimated, no published source
    ),
    'A': NucleotideKinetics(
        k_obs_mg=0.01,
        k_obs_mn=0.10,
        processivity_mg=2, # distributive
        processivity_mn=4,
        km=540.0          # highest Km = lowest affinity
    ),
    'C': NucleotideKinetics(
        k_obs_mg=0.008,   # slowest incorporation
        k_obs_mn=0.10,
        processivity_mg=1, # highly distributive
        processivity_mn=3,
        km=300.0           # estimated, no published source
    ),
}


@dataclass
class TdTParameters:
    """knobs for the synthesis model. defaults match Mg2+ at room temp."""
    # metal ion cofactor (affects processivity and rates)
    metal_ion: MetalIon = MetalIon.MG

    # scaling factor for processivity (enzyme variant, temperature, etc.)
    processivity_scale: float = 1.0

    # minimum bases per incorporation cycle (experimental control)
    min_bases: int = 3

    # G-tail parameters (TdT naturally favors G at end)
    min_g_tail: int = 3
    g_tail_scale: float = 1.0

    # leftover/carryover from incomplete washing between cycles
    p_leftover: float = 0.1
    leftover_scale: float = 0.5

    # sequence constraints
    max_total_length: int = 300
    max_attempts: int = 10000

    def validate(self) -> None:
        """sanity-check the knobs are in range."""
        if self.processivity_scale <= 0:
            raise ValueError(f"processivity_scale must be > 0, got {self.processivity_scale}")
        if self.min_bases < 1:
            raise ValueError(f"min_bases must be >= 1, got {self.min_bases}")
        if not (0 <= self.p_leftover <= 1):
            raise ValueError(f"p_leftover must be in [0, 1], got {self.p_leftover}")

    def get_processivity(self, base: str) -> int:
        """processivity for `base` under the current metal_ion + scale."""
        kinetics = NUCLEOTIDE_KINETICS[base]
        if self.metal_ion == MetalIon.MG:
            base_proc = kinetics.processivity_mg
        else:
            base_proc = kinetics.processivity_mn
        return max(1, int(base_proc * self.processivity_scale))

    def get_relative_rate(self, base: str) -> float:
        """rate for `base` relative to dGTP. dGTP is fastest so this is <= 1."""
        kinetics = NUCLEOTIDE_KINETICS[base]
        g_kinetics = NUCLEOTIDE_KINETICS['G']
        if self.metal_ion == MetalIon.MG:
            return kinetics.k_obs_mg / g_kinetics.k_obs_mg
        else:
            return kinetics.k_obs_mn / g_kinetics.k_obs_mn


@dataclass
class SynthesisStats:
    """rolling stats from a synthesis run; mostly for the rejection-rate report."""
    total_sequences: int = 0
    total_attempts: int = 0
    lengths: List[int] = field(default_factory=list)
    rejection_count: int = 0

    @property
    def rejection_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.rejection_count / self.total_attempts

    @property
    def mean_length(self) -> float:
        return mean(self.lengths) if self.lengths else 0.0

    @property
    def stdev_length(self) -> float:
        return stdev(self.lengths) if len(self.lengths) > 1 else 0.0

    def summary(self) -> str:
        return (
            f"Sequences: {self.total_sequences}, "
            f"Rejection rate: {self.rejection_rate:.2%}, "
            f"Length: {self.mean_length:.1f} ± {self.stdev_length:.1f} bp"
        )


def sample_negative_binomial(r: int, p: float) -> int:
    """sample NB(r, p): successes before r failures.

    in this model, a "success" is one more base incorporated and a
    "failure" is the enzyme dissociating. r=1 is geometric (distributive,
    e.g. dCTP/Mg2+); r>=3 is processive (dGTP, or anything with Mn2+).
    mean is r*p/(1-p), variance is r*p/(1-p)^2.
    """
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}")

    failures = 0
    successes = 0
    while failures < r:
        if random.random() < p:
            successes += 1
        else:
            failures += 1
    return successes


def sample_tdt_extension(base: str, params: TdTParameters) -> int:
    """one extension event for `base`. NB(r, p) where r is the literature
    processivity (already scaled by processivity_scale) and p is a function
    of relative incorporation rate. clamped to params.min_bases at the
    bottom because we want at least one base added per cycle in practice."""
    processivity = params.get_processivity(base)
    rel_rate = params.get_relative_rate(base)

    # convert processivity to negative binomial r parameter
    # higher processivity = higher r = longer runs before dissociation
    # note: processivity already includes processivity_scale via get_processivity()
    r = max(1, processivity)

    # p derived from relative rate: faster incorporation = higher p
    # base p=0.6, scaled by relative rate (G=1.0, others lower)
    p = 0.55 + (0.30 * rel_rate)
    p = min(0.89, max(0.5, p))

    extension = sample_negative_binomial(r, p)
    return max(extension, params.min_bases)


def sample_leftover(previous_base: str, params: TdTParameters) -> int:
    """carryover from incomplete washing between cycles. with probability
    p_leftover we add a short geometric run of `previous_base` before
    starting the next cycle. matches the insert-state behavior in the
    profile hmm."""
    if random.random() >= params.p_leftover:
        return 0

    # leftover is typically short (distributive, enzyme already partially dissociated)
    r = 1  # Geometric distribution
    p = params.leftover_scale
    return max(1, sample_negative_binomial(r, p))


def validate_pattern(pattern: str) -> None:
    """only A/C/G/T allowed."""
    invalid = set(pattern.upper()) - VALID_BASES
    if invalid:
        raise ValueError(f"Invalid bases in pattern: {invalid}. Use only A, C, G, T.")


def tdt_stepwise_synthesis(pattern: str, params: TdTParameters) -> str:
    """walk the pattern base by base, sampling an extension length per base
    and an optional carryover before each cycle. tacks a poly-G on the end.
    returns the core sequence; primers are added by generate_dna_sequence.
    follows the kinetics in Loc'h 2022 (life sci alliance).
    """
    validate_pattern(pattern)
    pattern = pattern.upper()

    segments = []
    previous_base = None

    for base in pattern:
        # leftover effect: incomplete washing leaves residual dNTP + TdT
        if previous_base is not None:
            leftover = sample_leftover(previous_base, params)
            if leftover > 0:
                segments.append(previous_base * leftover)

        # main synthesis: nucleotide-specific extension
        extension = sample_tdt_extension(base, params)
        segments.append(base * extension)

        previous_base = base

    # G-tail: TdT has highest affinity for dGTP (Km=120μM vs 540μM for dATP)
    # this creates a natural poly-G tail at the end of synthesis
    g_tail_length = sample_tdt_extension('G', params)
    g_tail_length = max(g_tail_length, params.min_g_tail)
    g_tail_length = int(g_tail_length * params.g_tail_scale)
    segments.append('G' * g_tail_length)

    return ''.join(segments)

def generate_dna_sequence(
    pattern: str,
    forward_primer: str,
    reverse_primer: str,
    params: TdTParameters,
    stats: Optional[SynthesisStats] = None
) -> str:
    """rejection-sampled tdt synthesis. retries until the synthesized core
    plus primers fits in params.max_total_length. matches what experimental
    protocols do via size-selection beads or gel cuts. raises RuntimeError
    if max_attempts is exhausted, which usually means turning down
    processivity_scale or bumping max_total_length.
    """
    for attempt in range(params.max_attempts):
        core = tdt_stepwise_synthesis(pattern, params)
        full_seq = forward_primer + core + reverse_primer

        if stats:
            stats.total_attempts += 1

        if len(full_seq) <= params.max_total_length:
            if stats:
                stats.lengths.append(len(full_seq))
                stats.total_sequences += 1
            return full_seq

        if stats:
            stats.rejection_count += 1

    raise RuntimeError(
        f"Could not generate sequence <= {params.max_total_length} bp "
        f"after {params.max_attempts} attempts. Consider adjusting parameters: "
        f"decrease processivity_scale or increase max_total_length."
    )

# the 17 constants used in the paper. each pattern is the ternary {A,C,T}
# barcode that comes out of binary_dna_codec.encode_float_to_dna with
# scrambling on. scramble seed is sha256(b"babbage") so simple inputs
# (zero, one) don't collapse to repetitive attractor patterns.
CONSTANTS: List[Tuple[str, str]] = [
    ("pi",                      "TCATCTCTACTACTCATACACTCATCATACAT"),
    ("e",                       "TCATCTCTATCTATCACATACACTATCATCTA"),
    ("G",                       "TACACACTCACATACTATATCATCTCACTACT"),
    ("Avogadro",                "TCTCATACACATATATCTATACTACTACTCAC"),
    ("hbar",                    "TATCATATATACTCACACATACTCTACACTAC"),
    ("electron_charge",         "TACATATACATCTCATCTCATATATCATCTAT"),
    ("electron_mass",           "TATCTCATCACATACACACACTCTCACTCATA"),
    ("proton_mass",             "TATATACACACTCATACTATCTACATCTATAT"),
    ("golden_ratio",            "TACTATATCTACTCTCACTACATCACTCTCTC"),
    ("epsilon_0",               "TACACATCTCTCTACACACATCATCTATACTA"),
    ("sqrt_2",                  "TACTATATCATCATCATACATACTCACATATC"),
    ("fine_structure",          "TACTACTCACACTCTCACTACTCTCTCACACT"),
    ("hydrogen_frequency",      "TCATATACATCTACATCACTCATCTCATACTA"),
    ("boltzmann",               "TATACTCATCTATCATACTCACATACTCATCA"),
    ("millionth_prime",         "TCATACTCTATACATATCTCTCATCTACATCA"),
    ("zero",                    "TATCACACTCTATATATACATCTACTCTCTAC"),
    ("calculator_easter_egg",   "TCATACTATCACACTATCTCTCACATATCTAC"),
]

# standard primers for PCR amplification
DEFAULT_FORWARD_PRIMER = "AAAATGAACTCATGGGACCTTCCACCCCAT"
DEFAULT_REVERSE_PRIMER = "ATGAGACCACGTCGAAGTTTGCGAGAGGAT"


def write_fasta(sequences: List[Tuple[str, str, str]], filename: str, line_width: int = 80):
    """
    Write sequences to FASTA format.

    Args:
        sequences: List of (header, description, sequence) tuples
        filename: Output filename
        line_width: Characters per line for sequence (default 80)
    """
    with open(filename, 'w') as f:
        for header, description, seq in sequences:
            f.write(f">{header} {description}\n")
            # wrap sequence to line_width
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i + line_width] + "\n")


def parse_args() -> argparse.Namespace:
    """cli for synthesizing the paper's full oligo pool fasta."""
    parser = argparse.ArgumentParser(
        description="TdT DNA Synthesis Simulator for DNA data storage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o", "--output",
        default="data/babbage.fasta",
        help="Output filename (.fasta or .csv)"
    )
    parser.add_argument(
        "-n", "--num-variants",
        type=int,
        default=118,
        help="number of variants per constant (default 118 x 17 = 2006 oligos)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum total sequence length (bp)"
    )
    parser.add_argument(
        "--metal-ion",
        choices=["Mg", "Mn", "Co"],
        default="Mg",
        help="Divalent metal ion cofactor (affects processivity)"
    )
    parser.add_argument(
        "--processivity-scale",
        type=float,
        default=1.0,
        help="Scale factor for processivity (enzyme variant, temperature)"
    )
    parser.add_argument(
        "--min-bases",
        type=int,
        default=3,
        help="Minimum homopolymer run length per cycle"
    )
    parser.add_argument(
        "--p-leftover",
        type=float,
        default=0.1,
        help="Probability of carryover from previous base"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit CSV header row (CSV format only)"
    )
    parser.add_argument(
        "--format",
        choices=["fasta", "csv", "auto"],
        default="auto",
        help="Output format (auto detects from filename extension)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed statistics"
    )
    return parser.parse_args()


def main() -> int:
    """generate the oligo pool. one fasta record per (constant, variant)."""
    args = parse_args()

    # set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed: {args.seed}")

    # map CLI metal ion choice to enum
    metal_ion_map = {"Mg": MetalIon.MG, "Mn": MetalIon.MN, "Co": MetalIon.CO}
    metal_ion = metal_ion_map[args.metal_ion]

    # configure synthesis parameters with literature-based kinetics
    params = TdTParameters(
        metal_ion=metal_ion,
        processivity_scale=args.processivity_scale,
        min_bases=args.min_bases,
        min_g_tail=3,
        g_tail_scale=1.0,
        p_leftover=args.p_leftover,
        leftover_scale=0.5,
        max_total_length=args.max_length,
        max_attempts=10000
    )

    try:
        params.validate()
    except ValueError as e:
        logger.info(f"Error: {e}", file=sys.stderr)
        return 1

    # print configuration
    logger.info(f"TdT conditions: {metal_ion.value}, processivity scale: {args.processivity_scale}")

    # determine output format
    output_format = args.format
    if output_format == "auto":
        if args.output.endswith(".fasta") or args.output.endswith(".fa"):
            output_format = "fasta"
        else:
            output_format = "csv"

    logger.info(f"Output format: {output_format}")

    stats = SynthesisStats()
    all_sequences = []  # for FASTA output

    # generate sequences
    for pool_id, (constant_name, pattern) in enumerate(CONSTANTS, start=1):
        pool_label = f"pool{pool_id}"

        for variant in range(args.num_variants):
            seq = generate_dna_sequence(
                pattern=pattern,
                forward_primer=DEFAULT_FORWARD_PRIMER,
                reverse_primer=DEFAULT_REVERSE_PRIMER,
                params=params,
                stats=stats
            )
            all_sequences.append((pool_label, constant_name, seq, variant))

        if args.verbose:
            logger.info(f"  {pool_label} ({constant_name}): {args.num_variants} sequences")

    # write output
    if output_format == "fasta":
        fasta_records = []
        for pool_label, constant_name, seq, variant in all_sequences:
            header = f"{pool_label}_{constant_name}_{variant}"
            description = f"length={len(seq)} constant={constant_name}"
            fasta_records.append((header, description, seq))
        write_fasta(fasta_records, args.output)
    else:
        # csv format
        with open(args.output, "w") as f:
            if not args.no_header:
                f.write("pool,constant,sequence,length\n")
            for pool_label, constant_name, seq, variant in all_sequences:
                f.write(f"{pool_label},{constant_name},{seq},{len(seq)}\n")

    # print summary
    total = len(CONSTANTS) * args.num_variants
    logger.info(f"\nGenerated {total} sequences to '{args.output}'")
    logger.info(f"Statistics: {stats.summary()}")

    if stats.rejection_rate > 0.5:
        print(
            f"\nWarning: High rejection rate ({stats.rejection_rate:.1%}). "
            "Consider increasing --max-length or decreasing --processivity-scale.",
            file=sys.stderr
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())