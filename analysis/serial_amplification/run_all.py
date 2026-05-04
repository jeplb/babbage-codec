#!/usr/bin/env python3
"""
run all section 2.11 serial amplification analyses.

loads data once and passes to each analysis.
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from config import load_all_generations
from shared import add_quick_arg


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='run all serial amplification analyses')
    add_quick_arg(parser)
    parser.add_argument('--only', nargs='+',
                        help='only run these analyses (e.g., 01 03)')
    args = parser.parse_args()

    analyses_all = ['01', '02', '03', '04', '05', '06', '07']
    analyses = args.only if args.only else analyses_all

    # determine if we need HMM decoding
    need_decode = any(a in analyses for a in ['03', '04', '06', '07'])

    print("loading data...", flush=True)
    t0 = time.time()
    data = load_all_generations(
        decode=need_decode, quick=args.quick, quick_n=args.quick_n)
    print(f"loaded in {time.time() - t0:.1f}s\n", flush=True)

    scripts = {
        '01': ('01_read_inventory', 'read inventory'),
        '02': ('02_read_length_distributions', 'read length distributions'),
        '03': ('03_decoder_accuracy', 'decoder accuracy'),
        '04': ('04_error_modes', 'error modes'),
        '05': ('05_chimera_characterization', 'chimera characterization'),
        '06': ('06_summary_figure', 'summary figure'),
        '07': ('07_consensus_depth', 'consensus depth'),
    }

    for num in analyses:
        if num not in scripts:
            print(f"unknown analysis: {num}", flush=True)
            continue
        module_name, description = scripts[num]
        print(f"{'=' * 50}", flush=True)
        print(f"  analysis {num} - {description}", flush=True)
        print(f"{'=' * 50}", flush=True)
        t1 = time.time()
        try:
            import importlib
            sys.path.insert(0, SCRIPT_DIR)
            mod = importlib.import_module(module_name)
            mod.run_analysis(data, quick=args.quick, quick_n=args.quick_n)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
        print(f"  completed in {time.time() - t1:.1f}s\n", flush=True)

    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
