"""regression test: the n-terminal flank state absorbs residual 5' primer.

imperfect primer trimming leaves a few primer bases (typically the C-run from
the forward-primer tail '...CACCCCAT') on the front of a read. without a
leading flank the first observed base is forced into match-state 0 (START's
only successor), so the residual base corrupts bit 0 and propagates to bit 1.
the flank lets viterbi skip the residue and lock onto the true codeword start.

asserts:
  1. clean read: flank unused, all bits recovered, with and without the flank.
  2. residual-primer read: flank-off flips bit 0; flank-on recovers it.

run with:  python test_flank_decoding.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_hmm import ProfileHMM, ProfileHMMParameters, StateType
from binary_dna_codec import decode_dna
from nanopore_pipeline import extract_bits_from_alignment
from babbage_sequence_list import CONSTANTS


# fine_structure codeword starts 'TACTAC...'; primer tail is '...CACCCCAT', so
# the residue that lands on the read front is a run of C.
PATTERN = dict(CONSTANTS)["fine_structure"]
GT_BITS = list(decode_dna(PATTERN))

# a clean read: every codeword base expanded to a length-3 homopolymer run,
# then the terminal poly-G tail. residue-free.
CLEAN = "".join(b * 3 for b in PATTERN) + "GGGGGG"
# the same read with the forward-primer tail 'CACCCCAT' left on the 5' end.
RESIDUAL = "CCCCAT" + CLEAN

FLANK_ON = ProfileHMMParameters()                       # default: flank enabled
FLANK_OFF = ProfileHMMParameters(p_start_to_flank=0.0)  # disabled


def _decode_bits(params, read):
    hmm = ProfileHMM(PATTERN, params)
    path, _ = hmm.viterbi(read)
    bits, _ = extract_bits_from_alignment(read, path, len(PATTERN))
    n_flank = sum(1 for st, _, _ in path if st == StateType.FLANK)
    return bits, n_flank


def test_clean_read_unaffected():
    """flank must not engage on a residue-free read, and bits round-trip with
    and without it."""
    for label, params in [("flank-off", FLANK_OFF), ("flank-on", FLANK_ON)]:
        bits, n_flank = _decode_bits(params, CLEAN)
        assert bits == GT_BITS, f"{label}: clean read misdecoded\n  {bits}\n  {GT_BITS}"
    bits_on, n_flank_on = _decode_bits(FLANK_ON, CLEAN)
    assert n_flank_on == 0, f"flank engaged on a clean read ({n_flank_on} obs)"
    print("  PASS: clean read fully recovered; flank stays out of the way")


def test_residual_primer_recovered():
    """flank-off flips bit 0 on residual primer; flank-on recovers it."""
    bits_off, _ = _decode_bits(FLANK_OFF, RESIDUAL)
    bits_on, n_flank_on = _decode_bits(FLANK_ON, RESIDUAL)

    assert bits_off[0] != GT_BITS[0], (
        "expected flank-off to flip bit 0 on residual primer, but it did not "
        f"(bit0={bits_off[0]}, gt={GT_BITS[0]})"
    )
    assert n_flank_on > 0, "flank did not engage on residual primer"
    assert bits_on == GT_BITS, (
        f"flank-on failed to recover residual-primer read\n  got {bits_on}\n  exp {GT_BITS}"
    )
    n_flip_off = sum(1 for i in range(32)
                     if bits_off[i] >= 0 and bits_off[i] != GT_BITS[i])
    print(f"  PASS: flank-off flips {n_flip_off} bit(s) on residual primer; "
          f"flank-on absorbs {n_flank_on} residue base(s) and recovers all 32")


if __name__ == "__main__":
    print("test_clean_read_unaffected:")
    test_clean_read_unaffected()
    print("test_residual_primer_recovered:")
    test_residual_primer_recovered()
    print("\nALL PASS")
