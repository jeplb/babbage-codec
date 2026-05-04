"""
section 3 - run-level profile HMM

a profile HMM that operates on homopolymer run observations (base, run_length)
rather than individual bases. supports laplace and gaussian emission models,
per-base and position-specific HP parameters, MLE fitting, and ablation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from statistics import median, mean, stdev

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# publication defaults: arial, editable text in illustrator, golden ratio
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42

PHI = (1 + 5 ** 0.5) / 2

from babbage_sequence_list import CONSTANTS

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

NEG_INF = float('-inf')


# emission functions

def laplace_log_prob(x, mu, b):
    """normalized log-probability under laplace distribution."""
    return -math.log(2 * b) - abs(x - mu) / b


def gaussian_log_prob(x, mu, sigma):
    """normalized log-probability under gaussian distribution."""
    return -0.5 * math.log(2 * math.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2


# run-length encoding utilities (self-contained copies)

def encode_runs(sequence):
    """convert sequence to run-length encoding."""
    if not sequence:
        return []
    sequence = sequence.upper()
    runs = []
    current_base = sequence[0]
    current_len = 1
    for base in sequence[1:]:
        if base == current_base:
            current_len += 1
        else:
            runs.append((current_base, current_len))
            current_base = base
            current_len = 1
    runs.append((current_base, current_len))
    return runs


def strip_gtail(runs):
    """remove terminal G-tail from runs."""
    if not runs:
        return runs
    end_idx = len(runs)
    for i in range(len(runs) - 1, -1, -1):
        if runs[i][0] != 'G':
            end_idx = i + 1
            break
    else:
        end_idx = 1
    return runs[:end_idx]


# parameter dataclass

@dataclass
class RunHMMParams:
    """parameters for run-level profile HMM."""
    # transitions from match
    p_mm: float = 0.90
    p_mi: float = 0.07
    p_md: float = 0.03
    # transitions from insert
    p_im: float = 0.70
    p_ii: float = 0.30
    # transitions from delete
    p_dm: float = 0.90
    p_dd: float = 0.10
    # g-tail transitions
    p_gg: float = 0.60
    p_mg: float = 0.93
    # emission
    p_base_err: float = 0.02
    emission_model: str = 'laplace'
    # per-base HP params (fitted from nanopore data)
    hp_mu: Dict[str, float] = field(default_factory=lambda: {
        'A': 3.54, 'C': 3.38, 'T': 4.35, 'G': 5.0,
    })
    hp_scale: Dict[str, float] = field(default_factory=lambda: {
        'A': 1.83, 'C': 1.61, 'T': 2.55, 'G': 3.0,
    })
    # position-specific override: {position: {base: (mu, scale)}}
    position_params: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None
    # insert/gtail params
    insert_mu: float = 2.0
    insert_scale: float = 1.5
    gtail_mu: float = 5.0
    gtail_scale: float = 3.0


# MLE fitting

@dataclass
class AlignedRun:
    """a single aligned run from a viterbi alignment."""
    position: int       # pattern position, -1 for insert
    state_type: str     # 'match', 'insert', 'gtail'
    expected_base: str
    observed_base: str
    observed_length: int


def fit_parameters(aligned_runs, model='laplace'):
    """fit HP parameters from aligned runs via MLE.

    laplace: mu = median, scale = MAD
    gaussian: mu = mean, scale = std
    """
    # group match observations by base
    base_lengths = {}
    for ar in aligned_runs:
        if ar.state_type == 'match' and ar.observed_base == ar.expected_base:
            base_lengths.setdefault(ar.observed_base, []).append(ar.observed_length)

    hp_mu = {}
    hp_scale = {}

    for base, lengths in base_lengths.items():
        if len(lengths) < 2:
            continue
        if model == 'laplace':
            mu = median(lengths)
            mad = median([abs(x - mu) for x in lengths])
            # avoid zero scale
            scale = max(mad, 0.1)
        else:
            mu = mean(lengths)
            scale = max(stdev(lengths), 0.1)
        hp_mu[base] = mu
        hp_scale[base] = scale

    params = RunHMMParams(emission_model=model)
    params.hp_mu.update(hp_mu)
    params.hp_scale.update(hp_scale)
    return params


# run-level profile HMM

class RunProfileHMM:
    """profile HMM operating on (base, run_length) observations."""

    def __init__(self, pattern, params=None):
        """
        pattern: DNA string (e.g. 'TCATCTCTAC...'), no G-tail
        params: RunHMMParams instance
        """
        self.pattern = pattern
        self.L = len(pattern)
        self.params = params or RunHMMParams()

        # precompute log transition probabilities
        p = self.params
        self.log_p_mm = math.log(p.p_mm)
        self.log_p_mi = math.log(p.p_mi)
        self.log_p_md = math.log(p.p_md)
        self.log_p_im = math.log(p.p_im)
        self.log_p_ii = math.log(p.p_ii)
        self.log_p_dm = math.log(p.p_dm)
        self.log_p_dd = math.log(p.p_dd)
        self.log_p_gg = math.log(p.p_gg)
        self.log_p_ge = math.log(1 - p.p_gg)
        self.log_p_mg = math.log(p.p_mg)
        # last match can go to I_{L-1} with p_mi
        self.log_p_last_mi = math.log(p.p_mi)
        # I_{L-1} goes to G with p_im
        self.log_p_last_im = math.log(p.p_im)

    def _emission_log_prob(self, x, mu, scale):
        """compute emission log-prob for run length using configured model."""
        if self.params.emission_model == 'gaussian':
            return gaussian_log_prob(x, mu, scale)
        return laplace_log_prob(x, mu, scale)

    def _get_hp_params(self, position, base):
        """get (mu, scale) for a given position and base."""
        p = self.params
        if p.position_params and position in p.position_params:
            pos_dict = p.position_params[position]
            if base in pos_dict:
                return pos_dict[base]
        return (p.hp_mu.get(base, 3.5), p.hp_scale.get(base, 2.0))

    def emit_match(self, i, obs):
        """log emission probability for match state at position i."""
        base, length = obs
        expected = self.pattern[i]
        p = self.params

        # base component
        if base == expected:
            log_base = math.log(1 - p.p_base_err)
        else:
            log_base = math.log(p.p_base_err / 3)

        # run length component
        mu, scale = self._get_hp_params(i, base)
        log_len = self._emission_log_prob(length, mu, scale)

        return log_base + log_len

    def emit_insert(self, obs):
        """log emission probability for insert state."""
        base, length = obs
        p = self.params
        log_base = math.log(0.25)
        log_len = self._emission_log_prob(length, p.insert_mu, p.insert_scale)
        return log_base + log_len

    def emit_gtail(self, obs):
        """log emission probability for g-tail state."""
        base, length = obs
        p = self.params
        if base == 'G':
            log_base = math.log(1 - p.p_base_err)
        else:
            log_base = math.log(p.p_base_err / 3)
        log_len = self._emission_log_prob(length, p.gtail_mu, p.gtail_scale)
        return log_base + log_len

    def viterbi(self, observations):
        """run viterbi algorithm on a sequence of (base, run_length) observations.

        returns (log_score, path) where path is a list of (state_type, position)
        tuples aligned to observations. state_type is 'M', 'I', or 'G'.
        """
        L = self.L
        N = len(observations)
        if N == 0:
            return NEG_INF, []

        # DP matrices: V_M[i][j], V_I[i][j], V_D[i][j], V_G[j]
        # i = model position (0..L-1), j = observation index (0..N-1)
        V_M = [[NEG_INF] * N for _ in range(L)]
        V_I = [[NEG_INF] * N for _ in range(L)]
        V_D = [[NEG_INF] * N for _ in range(L)]
        V_G = [NEG_INF] * N

        # backpointers: (state_type, i, j) - previous state
        BP_M = [[None] * N for _ in range(L)]
        BP_I = [[None] * N for _ in range(L)]
        BP_D = [[None] * N for _ in range(L)]
        BP_G = [None] * N

        # initialization: S -> M_0 (probability 1.0, log = 0)
        V_M[0][0] = self.emit_match(0, observations[0])
        BP_M[0][0] = ('S', -1, -1)

        # D_1 can be reached from M_0 without consuming an observation
        # but D is silent, so we handle it after M

        # also allow I_0 from M_0 at j=1
        # and D states from M_0

        # main loop: for each observation j
        for j in range(N):
            # --- match states ---
            for i in range(L):
                if j == 0 and i == 0:
                    continue  # already initialized
                if j == 0:
                    # can only reach M_i from D_{i-1} at j=0 (silent path)
                    # handled in delete propagation below
                    continue

                em = self.emit_match(i, observations[j])
                best = NEG_INF
                best_bp = None

                if i > 0:
                    # from M_{i-1} at j-1
                    v = V_M[i - 1][j - 1] + self.log_p_mm
                    if v > best:
                        best = v
                        best_bp = ('M', i - 1, j - 1)

                    # from I_{i-1} at j-1
                    v = V_I[i - 1][j - 1] + self.log_p_im
                    if v > best:
                        best = v
                        best_bp = ('I', i - 1, j - 1)

                    # from D_{i-1} at j-1
                    v = V_D[i - 1][j - 1] + self.log_p_dm
                    if v > best:
                        best = v
                        best_bp = ('D', i - 1, j - 1)

                if best > NEG_INF:
                    V_M[i][j] = em + best
                    BP_M[i][j] = best_bp

            # --- delete states (silent, same j) ---
            for i in range(1, L):
                best = NEG_INF
                best_bp = None

                # from M_{i-1} at j
                v = V_M[i - 1][j] + self.log_p_md
                if v > best:
                    best = v
                    best_bp = ('M', i - 1, j)

                # from D_{i-1} at j
                v = V_D[i - 1][j] + self.log_p_dd
                if v > best:
                    best = v
                    best_bp = ('D', i - 1, j)

                if best > NEG_INF:
                    V_D[i][j] = best
                    BP_D[i][j] = best_bp

            # after delete propagation, check if any new M states are reachable
            # from D states at j=0 (silent chain S -> M_0 -> D_1 -> ... -> D_k -> M_{k+1})
            if j == 0:
                for i in range(1, L):
                    if V_D[i][0] > NEG_INF:
                        # D_i at j=0 can feed M_{i+1} at j=1 (handled in next j)
                        pass
                    # also check if D feeds M at j=0 (not possible, M consumes obs)
                    # but D_i at j=0 can be a source for M_{i+1} at j=1

            # --- insert states ---
            for i in range(L):
                if j == 0:
                    # I_0 not reachable at j=0 (would need M_0 at j=-1)
                    continue

                em = self.emit_insert(observations[j])
                best = NEG_INF
                best_bp = None

                # from M_i at j-1
                v = V_M[i][j - 1] + self.log_p_mi
                if v > best:
                    best = v
                    best_bp = ('M', i, j - 1)

                # from I_i at j-1 (self-loop)
                v = V_I[i][j - 1] + self.log_p_ii
                if v > best:
                    best = v
                    best_bp = ('I', i, j - 1)

                if best > NEG_INF:
                    V_I[i][j] = em + best
                    BP_I[i][j] = best_bp

            # --- g-tail state ---
            if j > 0:
                em = self.emit_gtail(observations[j])
                best = NEG_INF
                best_bp = None

                # from M_{L-1} at j-1
                v = V_M[L - 1][j - 1] + self.log_p_mg
                if v > best:
                    best = v
                    best_bp = ('M', L - 1, j - 1)

                # from I_{L-1} at j-1
                v = V_I[L - 1][j - 1] + self.log_p_last_im
                if v > best:
                    best = v
                    best_bp = ('I', L - 1, j - 1)

                # from G at j-1 (self-loop)
                v = V_G[j - 1] + self.log_p_gg
                if v > best:
                    best = v
                    best_bp = ('G', -1, j - 1)

                if best > NEG_INF:
                    V_G[j] = em + best
                    BP_G[j] = best_bp

        # also handle M states reachable from D at previous j
        # (already handled in main loop above for j > 0)

        # termination: best score from last observation
        best_score = NEG_INF
        best_end = None

        # can end at M_{L-1}, I_{L-1}, or G at j=N-1
        v = V_M[L - 1][N - 1]
        if v > best_score:
            best_score = v
            best_end = ('M', L - 1, N - 1)

        v = V_I[L - 1][N - 1]
        if v > best_score:
            best_score = v
            best_end = ('I', L - 1, N - 1)

        v = V_G[N - 1]
        if v > best_score:
            best_score = v
            best_end = ('G', -1, N - 1)

        # also allow ending at any M_i or D_i if read is truncated
        # (optional: for now require full pattern traversal or G-tail)

        if best_end is None:
            return NEG_INF, []

        # traceback
        path = []
        state_type, i, j = best_end

        while state_type != 'S':
            if state_type in ('M', 'I', 'G'):
                path.append((state_type, i, j))
            elif state_type == 'D':
                # silent state, not aligned to observation
                path.append((state_type, i, j))

            if state_type == 'M':
                bp = BP_M[i][j]
            elif state_type == 'I':
                bp = BP_I[i][j]
            elif state_type == 'D':
                bp = BP_D[i][j]
            elif state_type == 'G':
                bp = BP_G[j]
            else:
                break

            if bp is None:
                break
            state_type, i, j = bp

        path.reverse()

        # extract alignment: only emitting states map to observations
        alignment = []
        for st, pos, obs_j in path:
            if st in ('M', 'I', 'G'):
                alignment.append((st, pos))

        return best_score, alignment

    def score(self, observations):
        """return just the viterbi score."""
        score, _ = self.viterbi(observations)
        return score


def score_all_patterns(observations, params=None):
    """score observations against all 17 patterns.

    returns list of (name, pattern, score) sorted by score descending.
    """
    results = []
    for name, pattern in CONSTANTS:
        hmm = RunProfileHMM(pattern, params)
        s = hmm.score(observations)
        results.append((name, pattern, s))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def classify(observations, params=None):
    """classify observations - return (best_name, best_score, margin).

    margin is best_score - second_best_score.
    """
    results = score_all_patterns(observations, params)
    if len(results) < 2:
        return results[0][0], results[0][2], float('inf')
    return results[0][0], results[0][2], results[0][2] - results[1][2]


# figure generation

def plot_emission_comparison(outpath):
    """laplace vs gaussian PDFs for representative bases."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 9 / (PHI * 2)))

    bases = [('A', 3.54, 1.83), ('C', 3.38, 1.61), ('T', 4.35, 2.55)]
    # wong palette (colorblind-safe)
    colors = {'laplace': '#0072B2', 'gaussian': '#D55E00'}

    for ax, (base, mu, scale) in zip(axes, bases):
        x = np.linspace(0, 15, 300)

        # laplace
        y_lap = np.array([math.exp(laplace_log_prob(xi, mu, scale)) for xi in x])
        ax.plot(x, y_lap, color=colors['laplace'], linewidth=1.5, label='Laplace')

        # gaussian
        y_gau = np.array([math.exp(gaussian_log_prob(xi, mu, scale)) for xi in x])
        ax.plot(x, y_gau, color=colors['gaussian'], linewidth=1.5,
                linestyle='--', label='Gaussian')

        ax.set_xlabel('Run length [nt]')
        if ax == axes[0]:
            ax.set_ylabel('Probability density')
        ax.set_title(f'{base} (μ={mu}, b={scale})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {outpath}")


def plot_score_margins(outpath, params=None):
    """histogram of correct-model score margins across all 17 patterns.

    for each pattern, synthesize a perfect read, score against all 17,
    and record (best_score - second_best).
    """
    margins = []
    default_params = params or RunHMMParams()

    for name, pattern in CONSTANTS:
        # build a clean observation: one run per base + G-tail
        runs = encode_runs(pattern)
        obs = []
        for base, _ in runs:
            mu = default_params.hp_mu.get(base, 3.5)
            obs.append((base, round(mu)))
        # add g-tail
        obs.append(('G', round(default_params.hp_mu.get('G', 5.0))))

        _, best_score, margin = classify(obs, default_params)
        margins.append(margin)

    fig, ax = plt.subplots(figsize=(5.5, 5.5 / PHI))
    ax.hist(margins, bins=15, color='#0072B2', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Score margin [log-likelihood]')
    ax.set_ylabel('Frequency [count]')
    ax.set_title('Correct model score margin (n = 17 constants)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {outpath}")


if __name__ == '__main__':
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("section 3 - run-level profile HMM\n")

    # demo: score a perfect read for pi
    name, pattern = CONSTANTS[0]  # pi
    runs = encode_runs(pattern)
    params = RunHMMParams()
    obs = []
    for base, _ in runs:
        mu = params.hp_mu.get(base, 3.5)
        obs.append((base, round(mu)))
    obs.append(('G', 5))

    hmm = RunProfileHMM(pattern, params)
    score, path = hmm.viterbi(obs)
    print(f"pattern: {name}")
    print(f"observations: {len(obs)} runs")
    print(f"viterbi score: {score:.2f}")
    print(f"path length: {len(path)} states")
    print()

    # classify
    results = score_all_patterns(obs, params)
    print(f"{'rank':<6} {'name':<24} {'score':>10}")
    for rank, (n, p, s) in enumerate(results[:5], 1):
        marker = ' <-' if n == name else ''
        print(f"{rank:<6} {n:<24} {s:>10.2f}{marker}")

    margin = results[0][2] - results[1][2]
    print(f"\nmargin: {margin:.2f}")

    # generate figures
    print()
    plot_emission_comparison(os.path.join(FIGURE_DIR, 'emission_laplace_vs_gaussian.pdf'))
    plot_score_margins(os.path.join(FIGURE_DIR, 'score_margin_histogram.pdf'))

    print(f"figures saved to {FIGURE_DIR}")
