"""
figure 1, panel A: encoding pipeline schematic.

vertical flow: IEEE 754 float -> XOR scramble -> transition encode -> DNA codeword.
worked example: pi = 3.1415927.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
})

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

# wong palette for DNA bases
BASE_COLORS = {
    'A': '#D55E00',
    'C': '#0072B2',
    'T': '#E69F00',
}

# worked example: pi
RAW_BITS     = '01000000010010010000111111011011'
SCRAMBLE_KEY = '10111010010010101001110000101000'
SCRAMBLED    = '11111010000000111001001111110011'
DNA_SEQ      = 'TCATCTCTACTACTCATACACTCATCATACAT'

SIGN_END = 1
EXP_END = 9

SIGN_COLOR = '#9E9E9E'
EXP_COLOR  = '#C8C8C8'
MANT_COLOR = '#E8E8E8'


def draw_bit_row(ax, y, bits, x0, cw, fs, field_shading=False, color='black'):
    for i, bit in enumerate(bits):
        x = x0 + i * cw
        if field_shading:
            if i < SIGN_END:
                bg = SIGN_COLOR
            elif i < EXP_END:
                bg = EXP_COLOR
            else:
                bg = MANT_COLOR
            ax.add_patch(FancyBboxPatch(
                (x - cw * 0.48, y - 0.028), cw * 0.96, 0.056,
                boxstyle="square,pad=0", facecolor=bg, edgecolor='none', zorder=1))
        ax.text(x, y, bit, fontsize=fs, fontfamily='monospace',
                ha='center', va='center', color=color, zorder=2)


def draw_dna_row(ax, y, seq, x0, cw, fs):
    for i, base in enumerate(seq):
        ax.text(x0 + i * cw, y, base, fontsize=fs, fontfamily='monospace',
                ha='center', va='center', color=BASE_COLORS[base],
                fontweight='bold', zorder=2)


def arrow_down(ax, y1, y2, xc):
    ax.annotate('', xy=(xc, y2), xytext=(xc, y1),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))


def draw_transition_diagram(ax, cx, cy, r=0.045):
    """compact A/C/T triangle with 0 and 1 arrows."""
    angles = {'A': 90, 'T': 210, 'C': 330}
    pos = {}
    for b, deg in angles.items():
        rad = np.radians(deg)
        pos[b] = (cx + r * np.cos(rad), cy + r * np.sin(rad))

    nr = 0.014

    for b, (px, py) in pos.items():
        ax.add_patch(plt.Circle((px, py), nr, fc='white',
                                ec=BASE_COLORS[b], lw=1.8, zorder=5))
        ax.text(px, py, b, fontsize=7.5, fontfamily='monospace', fontweight='bold',
                ha='center', va='center', color=BASE_COLORS[b], zorder=6)

    # 0: clockwise +1 (thin gray)
    zero = [('A', 'C'), ('C', 'T'), ('T', 'A')]
    # 1: skip +2 (thick dark)
    one  = [('A', 'T'), ('T', 'C'), ('C', 'A')]

    def _arrows(pairs, rad_curve, col, lw):
        for s, d in pairs:
            sx, sy = pos[s]; dx, dy = pos[d]
            v = np.array([dx - sx, dy - sy])
            u = v / np.linalg.norm(v)
            off = nr * 1.5
            ax.annotate('', xy=(dx - u[0]*off, dy - u[1]*off),
                        xytext=(sx + u[0]*off, sy + u[1]*off),
                        arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                        connectionstyle=f'arc3,rad={rad_curve}'),
                        zorder=3)

    _arrows(zero, 0.30, '#AAAAAA', 0.9)
    _arrows(one, -0.30, '#333333', 1.4)

    # "0" and "1" labels
    ax.text(cx + r + 0.028, cy, '0', fontsize=6.5, color='#AAAAAA',
            ha='center', va='center', fontweight='bold')
    ax.text(cx - r - 0.028, cy, '1', fontsize=6.5, color='#333333',
            ha='center', va='center', fontweight='bold')


def make_figure():
    fig_w = 3.4
    fig_h = 3.8
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cw = 0.026
    fs = 6.0
    lbl = 8
    x0 = 0.14
    xc = x0 + 15.5 * cw

    # panel label
    ax.text(0.005, 0.995, 'a', fontsize=12, fontweight='bold',
            va='top', ha='left', transform=ax.transAxes)

    # ---- stage 1: IEEE 754 ----
    y = 0.93
    ax.text(0.06, y, 'IEEE 754 binary32', fontsize=lbl,
            fontweight='bold', va='center', color='#333333')

    y -= 0.040
    # field labels
    ax.text(x0, y, 'S', fontsize=5.5, ha='center', va='center', color='#666666')
    ax.text(x0 + 4.5 * cw, y, 'exponent [8]', fontsize=5.5,
            ha='center', va='center', color='#666666')
    ax.text(x0 + 20 * cw, y, 'mantissa [23]', fontsize=5.5,
            ha='center', va='center', color='#666666')

    y -= 0.032
    draw_bit_row(ax, y, RAW_BITS, x0, cw, fs, field_shading=True)

    for p in [SIGN_END, EXP_END]:
        xs = x0 + (p - 0.5) * cw
        ax.plot([xs, xs], [y - 0.025, y + 0.025], color='#666666', lw=0.5, zorder=3)

    # value annotation
    rx = x0 + 32 * cw + 0.01
    ax.text(rx, y + 0.006, r'$\pi$ = 3.1415927', fontsize=6, va='center', color='#333333')
    ax.text(rx, y - 0.018, '0x40490FDB', fontsize=5, va='center',
            color='#888888', fontfamily='monospace')

    # ---- arrow ----
    y -= 0.040
    arrow_down(ax, y, y - 0.025, xc)
    y -= 0.048

    # ---- stage 2: XOR scramble ----
    ax.text(0.06, y, 'XOR scramble', fontsize=lbl,
            fontweight='bold', va='center', color='#333333')

    y -= 0.038
    ax.text(x0 - 0.015, y, 'raw', fontsize=5, ha='right', va='center', color='#999999')
    draw_bit_row(ax, y, RAW_BITS, x0, cw, fs)

    y -= 0.028
    ax.text(x0 - 0.015, y, r'$\oplus$', fontsize=8, ha='right', va='center', color='#555555')

    y -= 0.023
    ax.text(x0 - 0.015, y, 'key', fontsize=5, ha='right', va='center', color='#999999')
    draw_bit_row(ax, y, SCRAMBLE_KEY, x0, cw, fs, color='#777777')
    ax.text(x0 + 32 * cw + 0.01, y, "SHA-256(b'babbage')", fontsize=4.5,
            va='center', color='#999999', fontstyle='italic')

    y -= 0.020
    ax.plot([x0 - cw * 0.5, x0 + 31.5 * cw], [y, y], color='#AAAAAA', lw=0.5)

    y -= 0.020
    ax.text(x0 - 0.015, y, 'result', fontsize=5, ha='right', va='center', color='#999999')
    draw_bit_row(ax, y, SCRAMBLED, x0, cw, fs)

    # ---- arrow ----
    y -= 0.038
    arrow_down(ax, y, y - 0.025, xc)
    y -= 0.048

    # ---- stage 3: transition encode ----
    ax.text(0.06, y, 'Transition encode', fontsize=lbl,
            fontweight='bold', va='center', color='#333333')

    y_bits = y - 0.040
    y_dna = y_bits - 0.050

    # scrambled bits (muted)
    draw_bit_row(ax, y_bits, SCRAMBLED, x0, cw, fs, color='#BBBBBB')

    # dna row
    draw_dna_row(ax, y_dna, DNA_SEQ, x0, cw, fs + 0.5)

    # start base
    sx = x0 - cw * 1.3
    ax.text(sx, y_dna, 'A', fontsize=6.5, fontfamily='monospace',
            ha='center', va='center', color='#CCCCCC', fontweight='bold')
    ax.text(sx, y_dna - 0.020, 'start', fontsize=4, ha='center', va='center', color='#CCCCCC')
    ax.annotate('', xy=(x0 - cw * 0.5, y_dna),
                xytext=(sx + 0.012, y_dna),
                arrowprops=dict(arrowstyle='->', color='#DDDDDD', lw=0.5), zorder=1)

    # vertical arrows and shift labels for first 6 transitions
    ENCODE = {'A': {0: 'C', 1: 'T'}, 'C': {0: 'T', 1: 'A'}, 'T': {0: 'A', 1: 'C'}}
    mid_y = (y_bits + y_dna) / 2
    current = 'A'
    for i in range(6):
        x = x0 + i * cw
        ax.annotate('', xy=(x, y_dna + 0.015), xytext=(x, y_bits - 0.015),
                    arrowprops=dict(arrowstyle='->', color='#E0E0E0', lw=0.4), zorder=1)
        bit = int(SCRAMBLED[i])
        nxt = ENCODE[current][bit]
        shift = '+2' if bit == 1 else '+1'
        ax.text(x, mid_y, shift, fontsize=3.5, ha='center', va='center', color='#AAAAAA')
        current = nxt

    # transition rule diagram: bottom-right, clear of the main flow
    diag_cx = x0 + 28 * cw
    diag_cy = y_dna - 0.080
    draw_transition_diagram(ax, diag_cx, diag_cy, r=0.045)

    # ---- arrow (shifted left to avoid diagram) ----
    y = y_dna - 0.038
    arr_x = x0 + 7 * cw  # left of center to avoid diagram
    arrow_down(ax, y, y - 0.025, arr_x)
    y_arr_bot = y - 0.025

    # stage 4 positioned below both arrow and diagram
    y_s4 = min(y_arr_bot - 0.023, diag_cy - 0.045 - 0.028)

    # ---- stage 4: 32-base codeword ----
    ax.text(0.06, y_s4, '32-base codeword', fontsize=lbl,
            fontweight='bold', va='center', color='#333333')

    y_s4_dna = y_s4 - 0.038
    draw_dna_row(ax, y_s4_dna, DNA_SEQ, x0, cw, fs + 2)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        out = os.path.join(FIGURE_DIR, f'figure1a_encoding_pipeline.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    make_figure()
