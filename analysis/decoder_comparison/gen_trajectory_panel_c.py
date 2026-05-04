#!/usr/bin/env python3
"""
panel c -> head-to-head bootstrap across all 12 generations, fixed depth.

at each generation we load the gen-X TSV, pick a draw of d reads per codeword
(d capped at the per-codeword pool size), and feed both medaka smolecule and
the HMM bit-level consensus on the same draw. count constants recovered out
of 17. 20 reps per generation.

d is fixed at 25 because that's the highest common depth the 12 generations
can all support. gen 12 has two codewords below 25 (zero=22, proton_mass=24)
which get capped at all available reads.

writes panel_c_raw.tsv with one row per (generation, replicate).
"""

import os
import sys
import shutil
import subprocess
import time

import numpy as np
import edlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import PATTERNS, GROUND_TRUTH, plt
from binary_dna_codec import decode_dna, DECODE
from nanopore_pipeline import bits_to_float

GT_BITS = {n: list(decode_dna(p)) for n, p in PATTERNS.items()}

# gen N lives in barcode{N+6}. so gen 1 -> barcode07, gen 12 -> barcode18.
SERIAL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'serial_amplification')
TSV_DIR = os.path.join(SERIAL_DIR, 'per_barcode_reads')
OUT_TSV = os.path.join(SCRIPT_DIR, 'panel_c_raw.tsv')

GENS = list(range(1, 13))
DEPTH = 25
K_REPS = 20
MAX_PER_CW = 400  # cap on how many reads to pre-load per codeword


def rle(s):
    """collapse to one base per run."""
    if not s:
        return ''
    out = [s[0]]
    for c in s[1:]:
        if c != out[-1]:
            out.append(c)
    return ''.join(out)


def transitions_to_bits(rle_str, start_base='A'):
    """cyclic transition decode, base by base, from the start base."""
    bits = []
    prev = start_base
    for c in rle_str:
        bits.append(DECODE.get(prev, {}).get(c, -1))
        prev = c
    return bits


def load_gen(gen):
    """load up to MAX_PER_CW passing reads per codeword from gen X's TSV."""
    bc = f"barcode{gen + 6:02d}"
    path = os.path.join(TSV_DIR, f"{bc}.tsv")
    by_cw = {cw: [] for cw in PATTERNS}
    with open(path) as f:
        h = f.readline().rstrip().split('\t')
        rid_i = h.index('read_id')
        pf_i = h.index('passed_filter')
        pred_i = h.index('predicted_constant')
        conf_i = h.index('confidence')
        trim_i = h.index('trimmed')
        bits_i = h.index('bits')
        for line in f:
            fs = line.rstrip('\n').split('\t')
            if len(fs) <= max(pf_i, trim_i, bits_i):
                continue
            if fs[pf_i] != 'True':
                continue
            pred = fs[pred_i]
            if pred not in PATTERNS:
                continue
            if len(by_cw[pred]) >= MAX_PER_CW:
                continue
            trim = fs[trim_i]
            if not trim or len(trim) < 40:
                continue
            bv = fs[bits_i].split(',') if fs[bits_i] else []
            if len(bv) != 32:
                continue
            try:
                bits = [int(b) for b in bv]
            except ValueError:
                continue
            try:
                conf = float(fs[conf_i])
            except ValueError:
                conf = 0.0
            by_cw[pred].append((fs[rid_i], conf, trim, bits))
    cw_arrays = {}
    for cw in PATTERNS:
        rows = by_cw[cw]
        cw_arrays[cw] = {
            'ids':   [r[0] for r in rows],
            'confs': np.array([r[1] for r in rows], dtype=np.float32),
            'trims': [r[2] for r in rows],
            'bits':  np.array([r[3] for r in rows], dtype=np.int8) if rows
                      else np.empty((0, 32), dtype=np.int8),
            'n':     len(rows),
        }
    return cw_arrays


def hmm_consensus_recovery(idx_per_cw, cw_arrays):
    """confidence-weighted bit vote -> count of 17 recovered exactly."""
    n_ok = 0
    for cw in PATTERNS:
        idx = idx_per_cw[cw]
        if len(idx) == 0:
            continue
        b = cw_arrays[cw]['bits'][idx]
        c = cw_arrays[cw]['confs'][idx]
        votes = np.zeros(32)
        weights = np.zeros(32)
        for i in range(len(idx)):
            for p in range(32):
                if b[i, p] >= 0:
                    votes[p] += b[i, p] * c[i]
                    weights[p] += c[i]
        cb = [(1 if (weights[p] > 0 and votes[p] / weights[p] > 0.5) else 0)
              for p in range(32)]
        if bits_to_float(cb) == GROUND_TRUTH[cw]:
            n_ok += 1
    return n_ok


def smolecule_recovery(idx_per_cw, cw_arrays, work_dir):
    """write FASTAs, run smolecule, score consensuses against the design pattern."""
    pcf = os.path.join(work_dir, 'pcf')
    if os.path.exists(pcf):
        shutil.rmtree(pcf)
    os.makedirs(pcf)
    for cw in PATTERNS:
        idx = idx_per_cw[cw]
        if len(idx) == 0:
            continue
        ids = cw_arrays[cw]['ids']
        trims = cw_arrays[cw]['trims']
        with open(os.path.join(pcf, f"{cw}.fasta"), 'w') as f:
            for i in idx:
                f.write(f">{ids[i]}\n{trims[i]}\n")
    out = os.path.join(work_dir, 'smol_out')
    if os.path.exists(out):
        shutil.rmtree(out)
    fastas = sorted([os.path.join(pcf, fn) for fn in os.listdir(pcf)
                      if fn.endswith('.fasta')])
    if not fastas:
        return None
    # pinned setting for the paper, don't fiddle with these. depth 3 is the min
    # subread floor (the per-codeword draw sets the real depth so it basically
    # never bites) and length 100 is the min median subread length.
    cmd = ['medaka', 'smolecule', '--threads', '4', '--depth', '3',
           '--length', '100', '--quiet', out] + fastas
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    cons_path = os.path.join(out, 'consensus.fasta')
    if not os.path.exists(cons_path):
        return None
    seqs = {}
    name = None
    buf = []
    with open(cons_path) as f:
        for ln in f:
            ln = ln.rstrip()
            if ln.startswith('>'):
                if name:
                    seqs[name] = ''.join(buf)
                name = ln[1:].split()[0]
                buf = []
            else:
                buf.append(ln)
    if name:
        seqs[name] = ''.join(buf)
    n_ok = 0
    for cw, pat in PATTERNS.items():
        key = next((k for k in seqs if k.startswith(cw + '_')), None)
        if key is None:
            continue
        conseq = seqs[key].upper()
        cw_rle = rle(pat)
        conseq_rle = rle(conseq)
        rb = edlib.align(cw_rle, conseq_rle, task='path', mode='HW')
        if rb['editDistance'] < 0 or not rb['locations']:
            continue
        loc = rb['locations'][0]
        pred_rle = conseq_rle[loc[0]:loc[1] + 1] if loc[0] is not None else ''
        if not pred_rle:
            continue
        dec = transitions_to_bits(pred_rle, 'A')
        if len(dec) != 32:
            continue
        try:
            if bits_to_float(dec) == GROUND_TRUTH[cw]:
                n_ok += 1
        except Exception:
            pass
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(pcf, ignore_errors=True)
    return n_ok


def main():
    work_root = os.path.join(SCRIPT_DIR, '_panel_c_work')
    if os.path.exists(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root)

    with open(OUT_TSV, 'w') as f:
        f.write('generation\treplicate\ttime_s\tsmolecule_recovered\thmm_recovered\n')

    t_start = time.time()
    total = 0
    for gen in GENS:
        print(f"loading gen {gen}...", flush=True)
        cw_arrays = load_gen(gen)
        sizes = [a['n'] for a in cw_arrays.values()]
        print(f"  gen {gen} per-cw pool sizes -> min {min(sizes)}, "
              f"max {max(sizes)}", flush=True)
        for r in range(K_REPS):
            # seed by (gen, rep) so both pipelines see the same draw.
            rng = np.random.default_rng(seed=30000 + 100 * gen + r)
            idx_per_cw = {}
            for cw in PATTERNS:
                n = cw_arrays[cw]['n']
                take = min(DEPTH, n)
                if take == 0:
                    idx_per_cw[cw] = np.array([], dtype=np.int64)
                else:
                    idx_per_cw[cw] = rng.choice(n, size=take, replace=False)
            wd = os.path.join(work_root, f"g{gen}_r{r}")
            os.makedirs(wd, exist_ok=True)
            t0 = time.time()
            sm = smolecule_recovery(idx_per_cw, cw_arrays, wd)
            dt = time.time() - t0
            hmm = hmm_consensus_recovery(idx_per_cw, cw_arrays)
            with open(OUT_TSV, 'a') as f:
                f.write(f"{gen}\t{r}\t{dt:.2f}\t{sm}\t{hmm}\n")
            shutil.rmtree(wd, ignore_errors=True)
            total += 1
            print('.', end='', flush=True)
        elapsed = time.time() - t_start
        eta = elapsed / total * (len(GENS) * K_REPS - total)
        print(f" gen {gen} done ({elapsed:.0f}s, ETA {eta:.0f}s)")

    shutil.rmtree(work_root, ignore_errors=True)
    print(f"\nwrote {OUT_TSV}")


if __name__ == '__main__':
    main()
