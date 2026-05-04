#!/usr/bin/env python3
"""
section 4.02 - noiseless capacity ladder.

horizontal bar chart comparing theoretical capacity bounds for different
alphabet and constraint classes against the Babbage codec's achieved rate.
pure computation, no data files needed.
"""

import os
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from shared import remove_spines, save_fig, PHI, WONG, plt, np

FIGURE_DIR = SCRIPT_DIR

# capacity values (bits per nucleotide)
# q-ary no-adjacent-repeat capacity = log2(q-1)
ENTRIES = [
    ('Quaternary unconstrained\n(ACGT)',            math.log2(4)),
    ('Quaternary no-repeat\n(ACGT, no HP)',          math.log2(3)),
    ('Ternary unconstrained\n(ACT)',                 math.log2(3)),
    ('Ternary no-repeat\n(ACT, no HP)',              math.log2(2)),
    ('Babbage codec\n(32 bits / 32 nt)',             32 / 32),
]


def main():
    print("section 4.02: noiseless capacity ladder")

    labels = [e[0] for e in ENTRIES]
    values = [e[1] for e in ENTRIES]

    fig_w = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))

    colors = [WONG[0], WONG[1], WONG[1], WONG[2], WONG[2]]
    y_pos = np.arange(len(labels))[::-1]

    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Capacity [bits/nt]')
    ax.set_xlim(0, 2.3)

    # annotate values
    for bar, val in zip(bars, values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    # highlight that ternary no-repeat == Babbage codec rate
    ax.axvline(x=1.0, color='gray', ls='--', lw=0.8, alpha=0.5)

    remove_spines(ax)
    fig.tight_layout()

    save_fig(fig, os.path.join(FIGURE_DIR, 'capacity_ladder.pdf'))
    print("  done.")


if __name__ == '__main__':
    main()
