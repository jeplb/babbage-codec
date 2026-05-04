"""
profile HMM for tdt-synthesized DNA sequences.

each pattern gets its own HMM with self-loops on match states (geometric
hp-length), insert states for tdt carryover, delete states for skipped
bases, and a g-tail terminator. viterbi gives the MAP path; forward gives
the marginal likelihood; we use the latter to identify which of the 17
constants a read came from.

implementation is plain python + numpy. inner loops walk the sparse
log-trans matrix (~99x99, ~97% -inf) so each step touches only a couple
of predecessors. fast enough for the paper's read counts.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

import numba
from numba import njit

from logging_config import get_logger

logger = get_logger(__name__)


# sentinel used in the njit kernel; finite so numba doesn't choke on -inf
# arithmetic, and far enough below any realistic log-prob that real scores
# never reach it.
NEG_INF_SENTINEL = -1e308


BASES = ['A', 'C', 'G', 'T']
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for i, b in enumerate(BASES)}


def laplace_log_likelihood(observed: float, expected: float, scale: float) -> float:
    """laplace log-pdf, sans the -log(2b) constant.

    we use this for hp-length scoring in the run-based decoder. why laplace:
    nanopore hp-length errors are heavy-tailed (kurtosis ~4.3, shapiro-wilk
    p ~0). laplace beats gaussian by ~6000 AIC on the empirical fit. L1
    penalty also matches the long-tail behavior we see in the channel matrix.
    """
    return -abs(observed - expected) / scale


class StateType(Enum):
    START = "S"
    FLANK = "F"      # n-terminal insert; absorbs residual 5' primer before M_0
    MATCH = "M"      # emits the expected base; self-loop carries hp extension
    INSERT = "I"     # emits leftover/carryover from the previous position
    DELETE = "D"     # silent; lets us skip a base
    GTAIL = "G"      # terminal poly-G anchor
    END = "E"


@dataclass
class ProfileHMMParameters:
    """transition + emission knobs.

    these are the defaults used in the paper. fitted by hand against the
    convergence/error-rate tables; nothing fancy. the run-based scorer uses
    a laplace likelihood (see hp_laplace_scale) but the base-by-base
    viterbi uses the geometric implicit in the self-loop, which is what
    everyone does and is what everyone gets dinged for at review.
    """
    p_match_extend: float = 0.70        # M_i -> M_i (hp extension)
    p_match_to_next: float = 0.25       # M_i -> M_{i+1}
    p_match_to_insert: float = 0.03     # M_i -> I_i (leftover)
    p_match_to_delete: float = 0.02     # M_i -> D_{i+1}

    p_insert_to_match: float = 0.65     # I_i -> M_{i+1}
    p_insert_to_insert: float = 0.35    # I_i -> I_i

    p_delete_to_match: float = 0.90     # D_i -> M_i
    p_delete_to_delete: float = 0.10    # D_i -> D_{i+1}

    p_gtail_extend: float = 0.80        # G -> G
    p_gtail_from_last: float = 0.95     # last M -> G (vs M -> I)

    # n-terminal flank: a leading insert that absorbs residual 5' primer left
    # by imperfect trimming. without it the first observed base is forced into
    # M_0 (START's only successor), so residual primer flips bits 0-1. values
    # are deliberately small: the flank then only engages when the leading base
    # clearly mismatches M_0, fixing the primer artifact without over-absorbing
    # degraded codeword starts on high-error reads. tuned on real reads across
    # generations 1-12 (more aggressive values regress the late chimera-heavy
    # generations). set p_start_to_flank=0 to recover the pre-flank model.
    p_start_to_flank: float = 0.08      # START -> FLANK (vs START -> M_0)
    p_flank_extend: float = 0.35        # FLANK -> FLANK (geometric residue length)

    p_emission_error: float = 0.02      # rough nanopore base-call error rate

    # used by score_runs (run-based scorer); tuned on real data
    hp_expected_length: float = 3.0
    hp_laplace_scale: float = 1.8


def _build_sparse(log_trans: np.ndarray):
    """sparse adjacency lists from a dense log-trans matrix.

    predecessors[j] = [(i, log_p), ...] with log_p > -inf, ascending i.
    successors[i]   = [(j, log_p), ...] with log_p > -inf, ascending j.
    iteration order matters for tie-breaking in viterbi, see _run_viterbi.
    """
    n = log_trans.shape[0]
    predecessors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    successors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = float(log_trans[i, j])
            if v > -math.inf:
                predecessors[j].append((i, v))
                successors[i].append((j, v))
    return predecessors, successors


def _build_sparse_csr(log_trans: np.ndarray):
    """flatten predecessors/successors into CSR-style numpy arrays.

    returns six arrays:
        pred_indptr     int64, shape (n + 1,)
        pred_indices    int64, shape (nnz,)   - predecessor i for each j-column
        pred_log_trans  float64, shape (nnz,)
        succ_indptr     int64, shape (n + 1,)
        succ_indices    int64, shape (nnz,)   - successor j for each i-row
        succ_log_trans  float64, shape (nnz,)

    predecessor entries within each j-column are kept in ascending-i order
    so the njit viterbi reproduces the same lowest-i tie-breaking the
    list-based version uses.
    """
    n = log_trans.shape[0]
    pred_lists: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    succ_lists: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = float(log_trans[i, j])
            if v > -math.inf:
                pred_lists[j].append((i, v))
                succ_lists[i].append((j, v))

    pred_indptr = np.zeros(n + 1, dtype=np.int64)
    succ_indptr = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        pred_indptr[k + 1] = pred_indptr[k] + len(pred_lists[k])
        succ_indptr[k + 1] = succ_indptr[k] + len(succ_lists[k])

    pred_nnz = int(pred_indptr[-1])
    succ_nnz = int(succ_indptr[-1])
    pred_indices = np.zeros(pred_nnz, dtype=np.int64)
    pred_log_trans = np.zeros(pred_nnz, dtype=np.float64)
    succ_indices = np.zeros(succ_nnz, dtype=np.int64)
    succ_log_trans = np.zeros(succ_nnz, dtype=np.float64)

    for k in range(n):
        for off, (i, v) in enumerate(pred_lists[k]):
            pred_indices[pred_indptr[k] + off] = i
            pred_log_trans[pred_indptr[k] + off] = v
        for off, (j, v) in enumerate(succ_lists[k]):
            succ_indices[succ_indptr[k] + off] = j
            succ_log_trans[succ_indptr[k] + off] = v

    return (pred_indptr, pred_indices, pred_log_trans,
            succ_indptr, succ_indices, succ_log_trans)


def _logsumexp(values: List[float]) -> float:
    """numerically stable log-sum-exp. textbook."""
    if not values:
        return -math.inf
    max_val = max(values)
    if max_val == -math.inf:
        return -math.inf
    s = 0.0
    for v in values:
        s += math.exp(v - max_val)
    return max_val + math.log(s)


def _run_viterbi(seq_idx, n_states, predecessors, successors,
                 log_emit, emitting_mask, start_idx):
    """sparse viterbi forward pass. returns (V, bp).

    ties go to the lowest predecessor index because we iterate ascending
    and only replace on strict-greater. don't change this without rerunning
    the parity reference; small ordering changes flip a handful of paths
    and you'll spend an afternoon chasing it.
    """
    t_len = len(seq_idx)
    v = np.full((t_len + 1, n_states), -math.inf, dtype=np.float64)
    bp = np.zeros((t_len + 1, n_states), dtype=np.int64)

    v[0, start_idx] = 0.0

    # first observation: hop out of START via successors
    obs = int(seq_idx[0])
    for j, log_p in successors[start_idx]:
        emit = log_emit[j, obs]
        if emit > -math.inf:
            v[1, j] = log_p + emit
            bp[1, j] = start_idx

    for t in range(2, t_len + 1):
        obs = int(seq_idx[t - 1])
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit == -math.inf:
                continue

            best_score = -math.inf
            best_prev = 0
            for i, log_trans in predecessors[j]:
                prev_score = v[t - 1, i]
                if prev_score > -math.inf:
                    s = prev_score + log_trans
                    if s > best_score:  # strict >, ties stay with lowest i
                        best_score = s
                        best_prev = i

            if best_score > -math.inf:
                v[t, j] = best_score + emit
                bp[t, j] = best_prev

    return v, bp


@njit(cache=True)
def _run_viterbi_njit(seq_idx, n_states,
                      pred_indptr, pred_indices, pred_log_trans,
                      succ_indptr, succ_indices, succ_log_trans,
                      log_emit, emitting_mask, start_idx):
    """njit viterbi over CSR predecessors. mirrors _run_viterbi.

    log_emit and v use NEG_INF_SENTINEL (-1e308) instead of -inf so numba
    avoids -inf arithmetic. real log-probs never reach the sentinel.
    tie-breaking matches the list version: strict-greater update over
    ascending predecessor indices keeps the lowest i.
    """
    NEG = -1e308
    t_len = seq_idx.shape[0]
    v = np.full((t_len + 1, n_states), NEG, dtype=np.float64)
    bp = np.zeros((t_len + 1, n_states), dtype=np.int64)

    v[0, start_idx] = 0.0

    obs0 = seq_idx[0]
    for k in range(succ_indptr[start_idx], succ_indptr[start_idx + 1]):
        j = succ_indices[k]
        log_p = succ_log_trans[k]
        emit = log_emit[j, obs0]
        if emit > NEG:
            v[1, j] = log_p + emit
            bp[1, j] = start_idx

    for t in range(2, t_len + 1):
        obs = seq_idx[t - 1]
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit <= NEG:
                continue

            best_score = NEG
            best_prev = 0
            for k in range(pred_indptr[j], pred_indptr[j + 1]):
                i = pred_indices[k]
                log_trans = pred_log_trans[k]
                prev_score = v[t - 1, i]
                if prev_score > NEG:
                    s = prev_score + log_trans
                    if s > best_score:  # strict >, ties stay with lowest i
                        best_score = s
                        best_prev = i

            if best_score > NEG:
                v[t, j] = best_score + emit
                bp[t, j] = best_prev

    return v, bp


def _run_forward(seq_idx, n_states, predecessors, successors,
                 log_emit, emitting_mask, start_idx):
    """sparse forward pass with logsumexp. mirrors _run_viterbi structurally."""
    t_len = len(seq_idx)
    alpha = np.full((t_len + 1, n_states), -math.inf, dtype=np.float64)

    alpha[0, start_idx] = 0.0

    obs = int(seq_idx[0])
    for j, log_p in successors[start_idx]:
        emit = log_emit[j, obs]
        if emit > -math.inf:
            alpha[1, j] = log_p + emit

    for t in range(2, t_len + 1):
        obs = int(seq_idx[t - 1])
        for j in range(n_states):
            if not emitting_mask[j]:
                continue
            emit = log_emit[j, obs]
            if emit == -math.inf:
                continue

            preds = predecessors[j]

            # max-shift to keep the exp's from underflowing
            max_val = -math.inf
            for i, log_trans in preds:
                prev = alpha[t - 1, i]
                if prev > -math.inf:
                    s = prev + log_trans
                    if s > max_val:
                        max_val = s

            if max_val > -math.inf:
                s = 0.0
                for i, log_trans in preds:
                    prev = alpha[t - 1, i]
                    if prev > -math.inf:
                        s += math.exp(prev + log_trans - max_val)
                alpha[t, j] = max_val + math.log(s) + emit

    return alpha


class ProfileHMM:
    """one HMM per pattern.

    structure for a pattern of length L:
        START -> M_0 (self-loop) -> M_1 (self-loop) -> ... -> M_{L-1} -> GTAIL (self-loop) -> END
    plus I_i parallel to each M_i (carryover) and D_i for skipping.
    """

    def __init__(self, pattern: str, params: Optional[ProfileHMMParameters] = None):
        self.pattern = pattern.upper()
        self.params = params or ProfileHMMParameters()
        self.L = len(pattern)

        self._build_states()
        self._build_transitions()
        self._build_emissions()
        self._rebuild_sparse()

    def _build_states(self):
        """state list: START, FLANK, then M_i / I_i / D_i (D skipped at i=0),
        then GTAIL, END."""
        self.states: List[Tuple[StateType, int, Optional[str]]] = []
        self.state_to_idx: Dict[Tuple[StateType, int, Optional[str]], int] = {}

        self.states.append((StateType.START, -1, None))
        self.states.append((StateType.FLANK, -1, None))

        for i, base in enumerate(self.pattern):
            self.states.append((StateType.MATCH, i, base))
            self.states.append((StateType.INSERT, i, None))
            if i > 0:
                self.states.append((StateType.DELETE, i, None))

        self.states.append((StateType.GTAIL, self.L, 'G'))
        self.states.append((StateType.END, -1, None))

        for idx, state in enumerate(self.states):
            self.state_to_idx[state] = idx

        self.n_states = len(self.states)
        self.start_idx = self.state_to_idx[(StateType.START, -1, None)]
        self.flank_idx = self.state_to_idx[(StateType.FLANK, -1, None)]
        self.end_idx = self.state_to_idx[(StateType.END, -1, None)]
        self.gtail_idx = self.state_to_idx[(StateType.GTAIL, self.L, 'G')]

    def _get_state_idx(self, state_type: StateType, position: int,
                       base: Optional[str] = None) -> int:
        for state, idx in self.state_to_idx.items():
            if state[0] == state_type and state[1] == position:
                if base is None or state[2] == base:
                    return idx
        return -1

    def _build_transitions(self):
        """fill log_trans from the parameter dataclass.

        we set entries explicitly per state type and trust that the implied
        outgoing mass sums to 1 (it does, for the defaults). don't normalize
        afterwards because the variant code overrides individual self-loops
        and re-normalizes proportionally.
        """
        p = self.params
        self.log_trans = np.full((self.n_states, self.n_states), -np.inf)

        for state, idx in self.state_to_idx.items():
            state_type, pos, _ = state

            if state_type == StateType.START:
                m0_idx = self._get_state_idx(StateType.MATCH, 0, self.pattern[0])
                # split entry mass between the leading flank and M_0. when
                # p_start_to_flank == 0 the flank is unreachable and behaviour
                # matches the pre-flank model (START -> M_0 with prob 1).
                if p.p_start_to_flank > 0:
                    self.log_trans[idx, self.flank_idx] = math.log(p.p_start_to_flank)
                    self.log_trans[idx, m0_idx] = math.log(1.0 - p.p_start_to_flank)
                else:
                    self.log_trans[idx, m0_idx] = 0.0  # log(1)

            elif state_type == StateType.FLANK:
                m0_idx = self._get_state_idx(StateType.MATCH, 0, self.pattern[0])
                self.log_trans[idx, idx] = math.log(p.p_flank_extend)
                self.log_trans[idx, m0_idx] = math.log(1.0 - p.p_flank_extend)

            elif state_type == StateType.MATCH:
                self.log_trans[idx, idx] = math.log(p.p_match_extend)

                if pos < self.L - 1:
                    next_base = self.pattern[pos + 1]
                    m_next_idx = self._get_state_idx(StateType.MATCH, pos + 1, next_base)
                    self.log_trans[idx, m_next_idx] = math.log(p.p_match_to_next)

                    i_idx = self._get_state_idx(StateType.INSERT, pos)
                    self.log_trans[idx, i_idx] = math.log(p.p_match_to_insert)

                    if pos + 1 < self.L - 1:
                        d_idx = self._get_state_idx(StateType.DELETE, pos + 1)
                        if d_idx >= 0:
                            self.log_trans[idx, d_idx] = math.log(p.p_match_to_delete)
                else:
                    # last match -> gtail or insert. split the leftover mass.
                    remaining = 1.0 - p.p_match_extend
                    self.log_trans[idx, self.gtail_idx] = math.log(remaining * p.p_gtail_from_last)
                    i_idx = self._get_state_idx(StateType.INSERT, pos)
                    self.log_trans[idx, i_idx] = math.log(remaining * (1 - p.p_gtail_from_last))

            elif state_type == StateType.INSERT:
                self.log_trans[idx, idx] = math.log(p.p_insert_to_insert)
                if pos < self.L - 1:
                    next_base = self.pattern[pos + 1]
                    m_next_idx = self._get_state_idx(StateType.MATCH, pos + 1, next_base)
                    self.log_trans[idx, m_next_idx] = math.log(p.p_insert_to_match)
                else:
                    self.log_trans[idx, self.gtail_idx] = math.log(p.p_insert_to_match)

            elif state_type == StateType.DELETE:
                m_idx = self._get_state_idx(StateType.MATCH, pos, self.pattern[pos])
                self.log_trans[idx, m_idx] = math.log(p.p_delete_to_match)
                if pos < self.L - 1:
                    d_next_idx = self._get_state_idx(StateType.DELETE, pos + 1)
                    if d_next_idx >= 0:
                        self.log_trans[idx, d_next_idx] = math.log(p.p_delete_to_delete)

            elif state_type == StateType.GTAIL:
                self.log_trans[idx, idx] = math.log(p.p_gtail_extend)
                self.log_trans[idx, self.end_idx] = math.log(1.0 - p.p_gtail_extend)

    def _build_emissions(self):
        """match emits its base; insert biases toward the previous base
        (because that's what tdt carryover looks like in real reads); gtail
        emits G."""
        p = self.params
        self.log_emit = np.full((self.n_states, 4), -np.inf)

        for state, idx in self.state_to_idx.items():
            state_type, pos, base = state

            if state_type == StateType.FLANK:
                # uniform: residual primer can be any base, so the flank stays
                # agnostic and is chosen purely on transition cost.
                self.log_emit[idx, :] = math.log(0.25)

            elif state_type == StateType.MATCH:
                for i, b in enumerate(BASES):
                    if b == base:
                        self.log_emit[idx, i] = math.log(1.0 - p.p_emission_error)
                    else:
                        self.log_emit[idx, i] = math.log(p.p_emission_error / 3)

            elif state_type == StateType.INSERT:
                if pos > 0:
                    prev_base = self.pattern[pos - 1]
                    for i, b in enumerate(BASES):
                        # 0.55 for the carryover base, 0.15 each for the rest.
                        # numbers are eyeballed from a few hundred reads; not
                        # worth fitting more precisely.
                        if b == prev_base:
                            self.log_emit[idx, i] = math.log(0.55)
                        else:
                            self.log_emit[idx, i] = math.log(0.15)
                else:
                    self.log_emit[idx, :] = math.log(0.25)

            elif state_type == StateType.GTAIL:
                for i, b in enumerate(BASES):
                    if b == 'G':
                        self.log_emit[idx, i] = math.log(1.0 - p.p_emission_error)
                    else:
                        self.log_emit[idx, i] = math.log(p.p_emission_error / 3)

    def _rebuild_sparse(self):
        """rebuild sparse adjacency + emitting mask after log_trans is touched.

        variant_decoder calls this after overriding match self-loops; otherwise
        only __init__ touches log_trans and this runs once. also rebuilds the
        CSR-flat copies + sentinel-converted log_emit + bool mask array
        consumed by _run_viterbi_njit. log_emit must be set before we get
        here (we read it for the sentinel copy).
        """
        self.predecessors, self.successors = _build_sparse(self.log_trans)
        self.emitting_mask = [
            s[0] != StateType.START and s[0] != StateType.END for s in self.states
        ]
        (self.pred_indptr, self.pred_indices, self.pred_log_trans,
         self.succ_indptr, self.succ_indices, self.succ_log_trans) = \
            _build_sparse_csr(self.log_trans)
        self.emitting_mask_arr = np.array(self.emitting_mask, dtype=np.bool_)
        self.log_emit_sentinel = np.where(
            np.isneginf(self.log_emit), NEG_INF_SENTINEL, self.log_emit
        ).astype(np.float64)

    def viterbi(self, sequence: str
                ) -> Tuple[List[Tuple[StateType, int, Optional[str]]], float]:
        """MAP state path. return (path, log_score).

        path is a list of (StateType, position, base_or_None) tuples in time
        order, length T (one per observation, no START/END).
        """
        seq = sequence.upper()
        seq_idx = np.array([BASE_TO_IDX[b] for b in seq], dtype=np.int32)
        v, bp = _run_viterbi_njit(
            seq_idx, self.n_states,
            self.pred_indptr, self.pred_indices, self.pred_log_trans,
            self.succ_indptr, self.succ_indices, self.succ_log_trans,
            self.log_emit_sentinel, self.emitting_mask_arr, self.start_idx,
        )
        # restore -inf so path reconstruction below behaves like the list version
        v = np.where(v == NEG_INF_SENTINEL, -np.inf, v)
        t_len = len(seq_idx)

        # termination: best predecessor of END
        best_score = -math.inf
        best_state = 0
        for i, log_p in self.predecessors[self.end_idx]:
            score = v[t_len, i] + log_p
            if score > best_score:
                best_score = score
                best_state = i

        # backtrack
        path: List[Tuple[StateType, int, Optional[str]]] = []
        curr = best_state
        for t in range(t_len, 0, -1):
            path.append(self.states[curr])
            curr = int(bp[t, curr])
        path.reverse()
        return path, best_score

    def score_viterbi(self, sequence: str) -> float:
        """terminal Viterbi (MAP) score only - no backtrack, no path.

        identification runs this against all 17 patterns and only the winner
        needs its path reconstructed, so we skip the backtrack (and the -inf
        restore) for the losers. returns exactly the score viterbi() would.
        """
        seq = sequence.upper()
        seq_idx = np.array([BASE_TO_IDX[b] for b in seq], dtype=np.int32)
        v, _bp = _run_viterbi_njit(
            seq_idx, self.n_states,
            self.pred_indptr, self.pred_indices, self.pred_log_trans,
            self.succ_indptr, self.succ_indices, self.succ_log_trans,
            self.log_emit_sentinel, self.emitting_mask_arr, self.start_idx,
        )
        t_len = len(seq_idx)
        best_score = -math.inf
        for i, log_p in self.predecessors[self.end_idx]:
            vi = v[t_len, i]
            # sentinel (-1e308) marks unreachable states; real log-probs never
            # come close, so this cleanly mirrors the -inf guard in viterbi().
            if vi > NEG_INF_SENTINEL:
                score = vi + log_p
                if score > best_score:
                    best_score = score
        return best_score

    def forward(self, sequence: str) -> np.ndarray:
        """log-alpha matrix, shape (T+1, n_states)."""
        seq = sequence.upper()
        seq_idx = np.array([BASE_TO_IDX[b] for b in seq], dtype=np.int32)
        return _run_forward(
            seq_idx, self.n_states, self.predecessors, self.successors,
            self.log_emit, self.emitting_mask, self.start_idx,
        )

    def score_sequence(self, sequence: str) -> float:
        """log P(seq | model). forward + logsumexp at END."""
        seq = sequence.upper()
        seq_idx = np.array([BASE_TO_IDX[b] for b in seq], dtype=np.int32)
        alpha = _run_forward(
            seq_idx, self.n_states, self.predecessors, self.successors,
            self.log_emit, self.emitting_mask, self.start_idx,
        )
        t_len = len(seq_idx)
        terms = []
        for i, log_p in self.predecessors[self.end_idx]:
            v = alpha[t_len, i] + log_p
            if v > -math.inf:
                terms.append(v)
        return _logsumexp(terms)

    def decode_to_pattern(self, sequence: str) -> str:
        """run viterbi and pull out the unique match-state bases in order.
        cheap reconstruction of the source pattern from a noisy read."""
        path, _ = self.viterbi(sequence)
        decoded = []
        seen = set()
        for state_type, pos, base in path:
            if state_type == StateType.MATCH and pos not in seen:
                decoded.append(base)
                seen.add(pos)
        return ''.join(decoded)

    def score_runs(self, runs: List[Tuple[str, int]]) -> float:
        """run-based scorer. operates on the run-length-encoded read.

        loops over runs, expects them to match the pattern in order. if a
        run is the wrong base but matches the previous pattern position, we
        count it as carryover and apply the laplace length penalty with a
        smaller expected length. unmatched pattern positions are penalized
        once at the end.

        not used in the main pipeline (the base-by-base viterbi is what
        decides the constants) but useful for diagnostic scoring with the
        empirically-validated laplace length model.
        """
        if not runs:
            return -np.inf

        score = 0.0
        pattern_pos = 0
        p = self.params

        for run_base, run_len in runs:
            if pattern_pos >= self.L:
                # past the pattern. either this is gtail or junk.
                if run_base == 'G':
                    score += laplace_log_likelihood(
                        run_len, p.hp_expected_length * 2, p.hp_laplace_scale * 2
                    )
                else:
                    score += math.log(p.p_emission_error)
                continue

            expected_base = self.pattern[pattern_pos]

            if run_base == expected_base:
                score += math.log(1.0 - p.p_emission_error)
                score += laplace_log_likelihood(
                    run_len, p.hp_expected_length, p.hp_laplace_scale
                )
                pattern_pos += 1

            elif pattern_pos > 0 and run_base == self.pattern[pattern_pos - 1]:
                # carryover from previous position
                score += math.log(p.p_match_to_insert)
                score += laplace_log_likelihood(
                    run_len, p.hp_expected_length * 0.5, p.hp_laplace_scale
                )

            else:
                score += math.log(p.p_emission_error / 3)

        unmatched = self.L - pattern_pos
        if unmatched > 0:
            score += unmatched * math.log(p.p_match_to_delete)
        return score


class ProfileHMMDecoder:
    """multi-pattern wrapper. holds one ProfileHMM per name, runs forward
    on each to identify which pattern a read belongs to."""

    def __init__(self, patterns: Dict[str, str],
                 params: Optional[ProfileHMMParameters] = None):
        self.patterns = patterns
        self.params = params or ProfileHMMParameters()
        self.hmms: Dict[str, ProfileHMM] = {
            name: ProfileHMM(pattern, self.params)
            for name, pattern in patterns.items()
        }

    def identify(self, sequence: str, strip_primers: bool = False,
                 forward_primer: str = "", reverse_primer: str = ""
                 ) -> Tuple[str, float, Dict[str, float]]:
        """forward-score the read against every pattern, return the argmax.
        ties go to whichever name dict iteration hits first; in practice
        forward scores are well-separated so this never matters."""
        seq = sequence.upper()
        if strip_primers and forward_primer and reverse_primer:
            if seq.startswith(forward_primer):
                seq = seq[len(forward_primer):]
            if seq.endswith(reverse_primer):
                seq = seq[:-len(reverse_primer)]

        scores = {name: hmm.score_sequence(seq) for name, hmm in self.hmms.items()}
        best_name = max(scores, key=scores.get)
        return best_name, scores[best_name], scores

    def decode(self, sequence: str, strip_primers: bool = False,
               forward_primer: str = "", reverse_primer: str = ""
               ) -> Tuple[str, str, float]:
        """identify + viterbi-decode in one shot."""
        seq = sequence.upper()
        if strip_primers and forward_primer and reverse_primer:
            if seq.startswith(forward_primer):
                seq = seq[len(forward_primer):]
            if seq.endswith(reverse_primer):
                seq = seq[:-len(reverse_primer)]
        name, score, _ = self.identify(seq)
        decoded = self.hmms[name].decode_to_pattern(seq)
        return name, decoded, score


class _VariantHMM:
    """one variant for variant-level alignment.

    same machinery as ProfileHMM but with per-position self-loops fit to a
    specific variant's designed hp lengths, and a viterbi that tallies how
    many bases got assigned to each match state during backtrack. used by
    VariantLibrary for the per-variant analysis (paper section 5).
    """

    def __init__(self, log_trans: np.ndarray, log_emit: np.ndarray,
                 states: List[Tuple[StateType, int, Optional[str]]],
                 start_idx: int, end_idx: int,
                 designed_hp_lengths: List[int]):
        self.n_states = len(states)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.log_emit = np.array(log_emit, dtype=np.float64, copy=True)
        self.states = list(states)
        self.emitting_mask = [
            s[0] != StateType.START and s[0] != StateType.END for s in self.states
        ]
        self.designed_hp_lengths = list(designed_hp_lengths)
        log_trans_arr = np.asarray(log_trans, dtype=np.float64)
        self.predecessors, self.successors = _build_sparse(log_trans_arr)
        (self.pred_indptr, self.pred_indices, self.pred_log_trans,
         self.succ_indptr, self.succ_indices, self.succ_log_trans) = \
            _build_sparse_csr(log_trans_arr)
        self.emitting_mask_arr = np.array(self.emitting_mask, dtype=np.bool_)
        self.log_emit_sentinel = np.where(
            np.isneginf(self.log_emit), NEG_INF_SENTINEL, self.log_emit
        ).astype(np.float64)

    def viterbi_align(self, seq_idx: np.ndarray) -> Tuple[float, List[int]]:
        """returns (score, observed_hp_counts). counts are the number of
        emissions assigned to each match state along the MAP path."""
        v, bp = _run_viterbi_njit(
            seq_idx, self.n_states,
            self.pred_indptr, self.pred_indices, self.pred_log_trans,
            self.succ_indptr, self.succ_indices, self.succ_log_trans,
            self.log_emit_sentinel, self.emitting_mask_arr, self.start_idx,
        )
        v = np.where(v == NEG_INF_SENTINEL, -np.inf, v)
        t_len = len(seq_idx)

        best_score = -math.inf
        best_state = 0
        for i, log_p in self.predecessors[self.end_idx]:
            s = v[t_len, i] + log_p
            if s > best_score:
                best_score = s
                best_state = i

        n_pos = len(self.designed_hp_lengths)
        observed = [0] * n_pos
        curr = best_state
        for t in range(t_len, 0, -1):
            st_type, pos, _ = self.states[curr]
            if st_type == StateType.MATCH and 0 <= pos < n_pos:
                observed[pos] += 1
            curr = int(bp[t, curr])
        return best_score, observed


class VariantLibrary:
    """list of variant HMMs. align_read picks the best variant for a read
    and returns observed/designed hp pairs; align_batch is just a loop."""

    def __init__(self, pattern_length: int):
        self.pattern_length = pattern_length
        self.variants: List[_VariantHMM] = []

    def add_variant(self, log_trans, log_emit,
                    states: List[Tuple[StateType, int, Optional[str]]],
                    start_idx: int, end_idx: int,
                    designed_hp_lengths: List[int]):
        log_trans_arr = np.asarray(log_trans, dtype=np.float64)
        if log_trans_arr.ndim == 1:
            n = len(states)
            log_trans_arr = log_trans_arr.reshape(n, n)

        log_emit_arr = np.asarray(log_emit, dtype=np.float64)
        if log_emit_arr.ndim == 1:
            n = len(states)
            log_emit_arr = log_emit_arr.reshape(n, 4)

        self.variants.append(_VariantHMM(
            log_trans_arr, log_emit_arr, states,
            start_idx, end_idx, designed_hp_lengths,
        ))

    def __len__(self):
        return len(self.variants)

    def len(self) -> int:
        return len(self.variants)

    def align_read(self, seq_idx: np.ndarray) -> Tuple[int, float, List[int]]:
        """returns (best_variant_idx, score, hp_pairs_flat).

        hp_pairs_flat is [designed_0, observed_0, designed_1, observed_1, ...]
        of length 2 * pattern_length. easier to write to a tsv that way.
        """
        seq_idx = np.asarray(seq_idx, dtype=np.int32)
        best_idx = 0
        best_score = -math.inf
        best_observed: List[int] = []
        for i, variant in enumerate(self.variants):
            score, observed = variant.viterbi_align(seq_idx)
            if score > best_score:
                best_score = score
                best_idx = i
                best_observed = observed

        hp_pairs: List[int] = []
        if self.variants:
            designed = self.variants[best_idx].designed_hp_lengths
            for k in range(self.pattern_length):
                hp_pairs.append(int(designed[k]) if k < len(designed) else 0)
                hp_pairs.append(int(best_observed[k]) if k < len(best_observed) else 0)
        return best_idx, best_score, hp_pairs

    def align_batch(self, sequences: List[np.ndarray], read_ids: List[str]
                    ) -> List[Tuple[str, int, float, List[int]]]:
        out = []
        for rid, seq in zip(read_ids, sequences):
            vidx, score, hp = self.align_read(seq)
            out.append((rid, vidx, score, hp))
        return out


def demo():
    """sanity-check viterbi + identify on synthetic CATCAT-style reads."""
    pattern = "CATCAT"
    logger.info(f"target: {pattern}")
    hmm = ProfileHMM(pattern)

    test_sequences = [
        "CCCAAAATTTTCCCAAAATTTTGGGGGG",     # clean expansion
        "CCCCCAAATTTTTTCCCCCAAATTTGGGGGGG",  # variable runs
        "CCAATTCCAATTGGGG",                  # short runs
    ]

    for seq in test_sequences:
        path, score = hmm.viterbi(seq)
        decoded = hmm.decode_to_pattern(seq)
        logger.info(f"  seq {seq} -> {decoded} (score {score:.2f}, match={decoded == pattern})")

    patterns = {
        "pattern_A": "CATCAT",
        "pattern_B": "TACTAC",
        "pattern_C": "ACGTAC",
    }
    decoder = ProfileHMMDecoder(patterns)

    test_seqs = [
        ("CCCAAAATTTTCCCAAAATTTTGGGGGG", "pattern_A"),
        ("TTTTAAAACCCTTTTAAAACCCGGGGG", "pattern_B"),
        ("AAAACCCCGGGGTTTTAAAACCCGGGG", "pattern_C"),
    ]
    for seq, expected in test_seqs:
        name, score, _ = decoder.identify(seq)
        logger.info(
            f"  {seq[:30]}... -> {name} (expected {expected}, score {score:.2f})"
        )


if __name__ == "__main__":
    demo()
