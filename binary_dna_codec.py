"""
binary -> DNA codec via transition encoding plus a sha256 scramble.

each bit is encoded as a transition between bases. there are no homopolymer
runs in the encoded sequence because the base always changes.

  bit 0: shift +1 in the cycle A -> C -> T -> A
  bit 1: shift +2 in the cycle A -> T -> C -> A

we xor the input bits against a fixed pseudorandom sequence (sha256 of
"babbage") before encoding. without scrambling, an all-zero input gives a
boring CTACTAC... pattern that's easy to confuse with another all-zero
read; the scramble breaks that. xor is self-inverse, so descrambling is
the same operation.

each base in the output sequence depends on the bit AND the previous base,
so the encoding is also a quick check for read sanity at decode time.
"""

import struct
from typing import Tuple, List
from dataclasses import dataclass

from logging_config import setup_logging, get_logger

logger = get_logger(__name__)


# transition tables: ENCODE[current][bit] -> next, DECODE[prev][curr] -> bit
ENCODE = {
    'A': {0: 'C', 1: 'T'},
    'C': {0: 'T', 1: 'A'},
    'T': {0: 'A', 1: 'C'},
}
DECODE = {
    'A': {'C': 0, 'T': 1},
    'C': {'T': 0, 'A': 1},
    'T': {'A': 0, 'C': 1},
}

# convention: encoding starts at A
START_BASE = 'A'


import hashlib


def _sha256_scramble_sequence(key: bytes, n_bits: int = 32) -> List[int]:
    """sha256(key) read out as a bit sequence. unlike LFSR this is aperiodic
    over the lengths we care about, which is plenty."""
    digest = hashlib.sha256(key).digest()
    n_bytes = (n_bits + 7) // 8
    bits = []
    for byte in digest[:n_bytes]:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
            if len(bits) >= n_bits:
                return bits
    return bits[:n_bits]


# the actual scramble: 32 bits from sha256("babbage")
SCRAMBLE_SEQUENCE_32 = _sha256_scramble_sequence(b"babbage", 32)


def scramble_bits(bits: List[int], sequence: List[int] = None) -> List[int]:
    """xor the bits against the scramble sequence. cycles the sequence if
    bits is longer."""
    if sequence is None:
        sequence = SCRAMBLE_SEQUENCE_32
    n = len(bits)
    if len(sequence) < n:
        sequence = (sequence * ((n // len(sequence)) + 1))[:n]
    return [b ^ sequence[i] for i, b in enumerate(bits)]


def descramble_bits(bits: List[int], sequence: List[int] = None) -> List[int]:
    """xor is self-inverse, so descramble == scramble."""
    return scramble_bits(bits, sequence)


def encode_bits(bits: List[int], start_base: str = START_BASE) -> str:
    """walk the transition table starting from start_base; one bit gives one base."""
    dna = []
    current = start_base
    for bit in bits:
        next_base = ENCODE[current][bit]
        dna.append(next_base)
        current = next_base
    return ''.join(dna)


def decode_dna(dna: str, start_base: str = START_BASE) -> List[int]:
    """invert encode_bits. raises if the sequence has an invalid transition
    (e.g. a homopolymer or a G), which is what we want - the caller can
    decide what to do with a malformed read."""
    bits = []
    current = start_base
    for base in dna.upper():
        if base not in DECODE[current]:
            raise ValueError(f"invalid transition: {current} -> {base}")
        bit = DECODE[current][base]
        bits.append(bit)
        current = base
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """msb-first; pads at the front with zeros if not a multiple of 8."""
    while len(bits) % 8 != 0:
        bits = [0] + bits
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def bytes_to_bits(data: bytes) -> List[int]:
    """msb-first."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


@dataclass
class Float32Parts:
    """1 sign / 8 exponent (biased by 127) / 23 mantissa, IEEE 754 32-bit."""
    sign: int
    exponent: int
    mantissa: int

    @classmethod
    def from_float(cls, value: float) -> 'Float32Parts':
        packed = struct.pack('>f', value)
        bits = struct.unpack('>I', packed)[0]
        return cls(
            sign=(bits >> 31) & 1,
            exponent=(bits >> 23) & 0xFF,
            mantissa=bits & 0x7FFFFF,
        )

    def to_float(self) -> float:
        bits = (self.sign << 31) | (self.exponent << 23) | self.mantissa
        return struct.unpack('>f', struct.pack('>I', bits))[0]

    def to_bits(self) -> List[int]:
        bits = [self.sign]
        for i in range(7, -1, -1):
            bits.append((self.exponent >> i) & 1)
        for i in range(22, -1, -1):
            bits.append((self.mantissa >> i) & 1)
        return bits

    @classmethod
    def from_bits(cls, bits: List[int]) -> 'Float32Parts':
        if len(bits) != 32:
            raise ValueError(f"expected 32 bits, got {len(bits)}")
        sign = bits[0]
        exponent = 0
        for i in range(8):
            exponent = (exponent << 1) | bits[1 + i]
        mantissa = 0
        for i in range(23):
            mantissa = (mantissa << 1) | bits[9 + i]
        return cls(sign=sign, exponent=exponent, mantissa=mantissa)


def encode_float_to_dna(value: float, scramble: bool = False
                        ) -> Tuple[str, Float32Parts]:
    """float -> 32-base DNA. scramble=True is the operating mode used in
    the paper; without it, simple inputs (zero, one) collapse to repetitive
    patterns that confuse the decoder."""
    parts = Float32Parts.from_float(value)
    bits = parts.to_bits()
    if scramble:
        bits = scramble_bits(bits)
    return encode_bits(bits), parts


def decode_dna_to_float(dna: str, scrambled: bool = False
                        ) -> Tuple[float, Float32Parts]:
    """inverse of encode_float_to_dna. set scrambled=True if the encoder did."""
    bits = decode_dna(dna)
    if scrambled:
        bits = descramble_bits(bits)
    parts = Float32Parts.from_bits(bits)
    return parts.to_float(), parts


def encode_float_to_dna_scrambled(value: float) -> Tuple[str, Float32Parts]:
    return encode_float_to_dna(value, scramble=True)


def decode_dna_to_float_scrambled(dna: str) -> Tuple[float, Float32Parts]:
    return decode_dna_to_float(dna, scrambled=True)


# a few constants we use in tests
CONSTANTS_ENCODED = {
    'B_1': -0.5,
    'pi': 3.1415927,
    'positive_zero': 0.0,
    'negative_zero': -0.0,
    'machine_epsilon': 2.0 ** -23,
}


def format_float_encoding(name: str, value: float) -> str:
    """pretty-print the IEEE 754 layout next to its DNA encoding. for slides."""
    dna, parts = encode_float_to_dna(value)
    bits = parts.to_bits()
    sign_bit = bits[0]
    exp_bits = ''.join(str(b) for b in bits[1:9])
    mant_bits = ''.join(str(b) for b in bits[9:])
    sign_dna = dna[0]
    exp_dna = dna[1:9]
    mant_dna = dna[9:]
    lines = [
        f"constant: {name} = {value}",
        f"",
        f"IEEE 754 32-bit float:",
        f"  sign:     {sign_bit}",
        f"  exponent: {exp_bits} (biased: {parts.exponent}, actual: {parts.exponent - 127})",
        f"  mantissa: {mant_bits}",
        f"",
        f"DNA encoding:",
        f"  sign:     {sign_dna}",
        f"  exponent: {exp_dna}",
        f"  mantissa: {mant_dna}",
        f"  full:     {dna}",
    ]
    return '\n'.join(lines)


def demo():
    """small sanity check + the zero-attractor demo."""
    logger.info("transition encoding (current_base + bit -> next_base):")
    logger.info("  current | bit 0 | bit 1")
    for base in ['A', 'C', 'T']:
        logger.info(f"     {base}    |   {ENCODE[base][0]}   |   {ENCODE[base][1]}")

    logger.info("\nscramble sequence (sha256('babbage'), 32 bits):")
    logger.info(f"  {''.join(str(b) for b in SCRAMBLE_SEQUENCE_32)}")

    # the all-zero attractor
    zero_bits = [0] * 32
    zero_unscrambled = encode_bits(zero_bits)
    zero_scrambled_bits = scramble_bits(zero_bits)
    zero_scrambled = encode_bits(zero_scrambled_bits)
    logger.info(f"\nall-zero input:")
    logger.info(f"  no scramble: bits {''.join(str(b) for b in zero_bits)} -> {zero_unscrambled}")
    logger.info(f"  scrambled:   bits {''.join(str(b) for b in zero_scrambled_bits)} -> {zero_scrambled}")

    # round-trip
    logger.info(f"\nround-trip with scrambling:")
    test_values = [
        ('0', 0.0), ('pi', 3.1415927), ('e', 2.7182818),
        ('phi', 1.6180340), ('sqrt2', 1.4142135), ('G', 6.674e-11),
    ]
    for name, value in test_values:
        dna, _ = encode_float_to_dna(value, scramble=True)
        decoded, _ = decode_dna_to_float(dna, scrambled=True)
        ok = "ok" if abs(value - decoded) < 1e-6 or value == decoded else "FAIL"
        logger.info(f"  {name:>6} {value:>15.7g} -> {dna} -> {decoded:>15.7g}  {ok}")

    # hamming distances between scrambled patterns
    logger.info(f"\nhamming distances (scrambled):")
    values = [0.0, 1.0, 2.0, 3.1415927, 2.7182818]
    names = ['0', '1', '2', 'pi', 'e']
    patterns = [encode_float_to_dna(v, scramble=True)[0] for v in values]
    header = "       " + "".join(f"{n:>6}" for n in names)
    logger.info(header)
    for i, n1 in enumerate(names):
        row = f"  {n1:>6}"
        for j in range(len(names)):
            if i <= j:
                d = sum(c1 != c2 for c1, c2 in zip(patterns[i], patterns[j]))
                row += f"{d:>6}"
            else:
                row += " " * 6
        logger.info(row)


if __name__ == "__main__":
    setup_logging()
    demo()
