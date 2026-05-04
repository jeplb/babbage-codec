#!/usr/bin/env python3
"""
download and merge variant alignment results from S3.

each of the 17 per-constant TSVs contains scores for ALL reads. for each
read, picks the constant with the highest viterbi score (argmax), keeping
only the winning row. outputs a single TSV with one row per read.

usage:
    python3 collect_variant_results.py [-o all_variant_alignments.tsv]
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

from babbage_sequence_list import CONSTANTS

S3_BUCKET = 'babbage-variant-alignment'


def download_results(output_dir):
    """download per-constant TSVs from S3."""
    os.makedirs(output_dir, exist_ok=True)
    constant_names = [name for name, _ in CONSTANTS]
    downloaded = []

    for name in constant_names:
        s3_path = f's3://{S3_BUCKET}/output/{name}.variant_alignment.tsv'
        local_path = os.path.join(output_dir, f'{name}.variant_alignment.tsv')

        try:
            subprocess.run(
                ['aws', 's3', 'cp', s3_path, local_path],
                check=True, capture_output=True,
            )
            downloaded.append((name, local_path))
            print(f"  {name}: downloaded", file=sys.stderr)
        except subprocess.CalledProcessError:
            print(f"  {name}: NOT FOUND on S3", file=sys.stderr)

    return downloaded


def merge_results(downloaded, output_path):
    """per-read argmax across all 17 per-constant TSVs.

    for each read_id, keeps the row from whichever constant gave the
    highest viterbi_score. outputs one row per read with a prepended
    constant_name column.
    """
    # read_id -> (constant_name, score, full_row_string)
    best_per_read = {}
    per_constant_stats = {}
    file_header = None

    for const_name, tsv_path in downloaded:
        n_rows = 0
        scores = []

        with open(tsv_path) as f:
            hdr = f.readline().strip()
            if file_header is None:
                file_header = hdr

            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                parts = line.split('\t')
                read_id = parts[0]
                try:
                    score = float(parts[2])
                except (ValueError, IndexError):
                    continue
                scores.append(score)

                prev = best_per_read.get(read_id)
                if prev is None or score > prev[1]:
                    best_per_read[read_id] = (const_name, score, line)

        mean_score = np.mean(scores) if scores else float('nan')
        per_constant_stats[const_name] = (n_rows, mean_score)

    # count how many reads were assigned to each constant
    assigned_counts = defaultdict(int)
    for rid, (cname, score, row) in best_per_read.items():
        assigned_counts[cname] += 1

    # write merged output
    with open(output_path, 'w') as out:
        out.write(f'constant_name\t{file_header}\n')
        for rid in sorted(best_per_read.keys()):
            cname, score, row = best_per_read[rid]
            out.write(f'{cname}\t{row}\n')

    # compute deletion stats per constant from winning rows
    del_stats = {}
    for const_name in [name for name, _ in CONSTANTS]:
        n_deletions = 0
        n_positions = 0
        for rid, (cname, score, row) in best_per_read.items():
            if cname != const_name:
                continue
            parts = row.split('\t')
            # columns: read_id, variant_idx, score, then pairs
            for i in range(3, len(parts) - 1, 2):
                try:
                    designed = int(parts[i])
                    observed = int(parts[i + 1])
                    if designed > 0:
                        n_positions += 1
                        if observed == 0:
                            n_deletions += 1
                except (ValueError, IndexError):
                    pass
        del_rate = n_deletions / max(1, n_positions)
        del_stats[const_name] = del_rate

    return best_per_read, assigned_counts, per_constant_stats, del_stats


def main():
    parser = argparse.ArgumentParser(description='collect variant alignment results')
    parser.add_argument('-o', '--output', default='all_variant_alignments.tsv',
                        help='output merged TSV (one row per read)')
    parser.add_argument('--download-dir', default='variant_jobs/results',
                        help='local directory for downloaded TSVs')
    parser.add_argument('--skip-download', action='store_true',
                        help='skip S3 download, use existing local files')

    args = parser.parse_args()

    # download
    if not args.skip_download:
        print("downloading results from S3...", file=sys.stderr)
        downloaded = download_results(args.download_dir)
    else:
        downloaded = []
        for name, _ in CONSTANTS:
            path = os.path.join(args.download_dir, f'{name}.variant_alignment.tsv')
            if os.path.exists(path):
                downloaded.append((name, path))

    if not downloaded:
        print("error: no result TSVs found", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(downloaded)}/17 constants collected", file=sys.stderr)

    # merge
    print("merging (per-read argmax across constants)...", file=sys.stderr)
    best, assigned, const_stats, del_stats = merge_results(downloaded, args.output)

    # summary
    print(f"\n{'constant':<25} {'assigned':>8} {'scored':>8} "
          f"{'mean_score':>12} {'del_rate':>10}", file=sys.stderr)
    print('-' * 66, file=sys.stderr)
    for name, _ in CONSTANTS:
        n_assigned = assigned.get(name, 0)
        n_scored, ms = const_stats.get(name, (0, float('nan')))
        dr = del_stats.get(name, 0.0)
        print(f"{name:<25} {n_assigned:>8} {n_scored:>8} "
              f"{ms:>12.1f} {dr:>9.4f}", file=sys.stderr)

    print(f"\ntotal reads: {len(best)}", file=sys.stderr)
    print(f"output: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
