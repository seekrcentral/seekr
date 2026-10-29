from __future__ import annotations
"""
modules/analyzers/base.py

Base function and constants that might be common for analyzers.
"""

import re
import os
import copy
import typing
from collections import defaultdict, deque

import numpy as np
import PyGT

import seekr.modules.structures as structures

LOW_FLUX = 1e-19
FAST_FLUX = 1e6
GAS_CONSTANT = 0.0019872 # in kcal/mol*K
DEFAULT_MAX_CONVERGENCE_WINDOWS = 20
NUM_BOOTSTRAP_SAMPLES = 100
NUM_RESAMPLES = 100 #Very slow.
NUM_MONTE_CARLO_SAMPLES = 100
DEFAULT_IMAGE_DIR = "images_and_plots/"
VT_BACKGROUND_COLOR = "white"

# MMVT Constants
NS_PER_PS = 1e-3
N_ALPHA_BETA_DIR = "N_alpha_beta/"
T_ALPHA_DIR = "T_alpha/"
K_ALPHA_BETA_DIR = "k_alpha_beta/"
PI_ALPHA_DIR = "pi_alpha/"
N_IJ_ALPHA_DIR = "N_ij_alpha/"
R_I_ALPHA_DIR = "R_i_alpha/"
N_IJ_DIR = "N_ij/"
R_I_DIR = "R_i/"

# The threshold beneath which to skip plotting the convergence
MIN_PLOT_NORM = 1e-18

class MissingStatisticsError(Exception):
    """
    Catch an error in analysis stage where statistics are missing for a 
    particular anchor or unpartitioned situation.
    """
    pass

def refine_function_str(old_function_str):
        """
        Convert the energy landscape from a form used by OpenMM
        to a form used by Python.
        """
        new_function_str = re.sub("\^", "**", old_function_str)
        return new_function_str

def check_graph_connected(
        graph: dict[int, list[int]], 
        state_index1: int, 
        state_index2: int,
        ) -> bool:
    """
    A graph is made of anchor indices that have recorded bounces against
    neighboring anchors. If pathways can be found, then sufficient statistics
    likely exist to compute kinetics from the resulting bounces.
    
    Parameters:
    -----------
    graph : dict
        A dictionary whose keys are node incides and whose values are
        lists of connected node indices
    state_index1 : int
        The index of the "source" node.
    state_index2 : int
        The index o the "sink" node.
        
    Returns:
    --------
    connected : bool
        If a valid path can be found between state_index1 and 
        state_index2, then return True. Otherwise return False.
    """
    max_iter = len(graph) + 5
    if state_index1 == state_index2:
        return True
    explored_nodes = [state_index1]
    edge_nodes: list[int] = copy.deepcopy(graph[state_index1])
    counter = 0
    while len(edge_nodes) > 0:
        edge_node = edge_nodes.pop()
        if state_index2 == edge_node:
            return True
        explored_nodes.append(edge_node)
        for possible_edge_node in graph[edge_node]:
            if possible_edge_node not in edge_nodes \
                    and possible_edge_node not in explored_nodes:
                edge_nodes.append(possible_edge_node)
                
        counter += 1
        assert counter < max_iter, "Maximum iterations exceeded."
        
    return False

def rate_matrix_to_transition_matrix(
        Q: np.ndarray
        ) -> np.ndarray:
    """
    Given a rate matrix Q, compute probability transition matrix K.
    """
    K = np.zeros(Q.shape, dtype=np.longdouble)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if i == j:
                K[i,j] = 0.0
            else:
                if Q[i,i] == 0.0:
                    K[i,j] = 0.0
                else:
                    K[i,j] = -Q[i,j] / Q[i,i]
    
    return K

def compute_rate_matrix(
        num_states: int,
        N_ij: typing.DefaultDict[tuple[int, int], int | float],
        R_i: typing.DefaultDict[int, float]
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Q and K from N_ij and R_i.
    """
    Q = np.zeros((num_states, num_states), dtype=np.longdouble)
    for i in range(num_states):
        for j in range(num_states):
            if R_i[i] == 0.0:
                Q[i,j] = 0.0
            else:
                Q[i,j] = N_ij[i,j] / R_i[i]
            assert Q[i,j] >= 0.0, "Q[i,j]: {}".format(Q[i,j])

    for i in range(num_states):
        row_sum = np.sum(Q[i], dtype=np.longdouble)
        if row_sum == 0:
            new_row_sum = np.longdouble(0.0)
            for j in range(num_states):
                Q[i,j] = Q[j,i]
                new_row_sum += Q[j,i]
            
            Q[i,i] = -new_row_sum
        else:
            Q[i,i] = -row_sum
            
    K = rate_matrix_to_transition_matrix(Q)
    return Q, K

def impute_missing_rates(
        Q: np.ndarray,
        p_i: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill in missing off-diagonal entries of the rate matrix Q using
    detailed balance against the stationary distribution p_i.

    For each pair (i, j) where Q[i, j] == 0 but Q[j, i] > 0, and both
    p_i[i] and p_i[j] are positive, set
        Q[i, j] = (p_i[j] / p_i[i]) * Q[j, i].
    This is the unique rate consistent with the LSQ p_i and avoids the
    artificial timescales introduced by HIGH/LOW count substitution.
    Pairs with no observation in either direction, or involving a
    milestone with p_i == 0 (unreachable in the observed graph), are
    left at zero. Diagonals are recomputed so each row sums to zero.
    """
    num_states = Q.shape[0]
    Q_new = copy.deepcopy(Q)
    for i in range(num_states):
        Q_new[i, i] = 0.0
    for i in range(num_states):
        if p_i[i] <= 0.0:
            continue
        for j in range(num_states):
            if i == j:
                continue
            if Q_new[i, j] > 0.0:
                continue
            if p_i[j] <= 0.0:
                continue
            if Q_new[j, i] > 0.0:
                Q_new[i, j] = (p_i[j] / p_i[i]) * Q_new[j, i]

    for i in range(num_states):
        row_sum = np.sum(Q_new[i], dtype=np.longdouble)
        Q_new[i, i] = -row_sum

    K_new = rate_matrix_to_transition_matrix(Q_new)
    return Q_new, K_new

def calculate_thermodynamics(
        num_states: int,
        Q: np.ndarray,
        temperature: float,
        Q_raw: np.ndarray | None = None,
        N_ij: typing.DefaultDict | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the stationary milestone probability distribution p_i and
    the corresponding free energy profile.

    Uses a log-space weighted least-squares approach, exploiting detailed
    balance of the CTMC rate matrix Q:  p_i * Q[i,j] = p_j * Q[j,i],
    so log p_j - log p_i = log Q[i,j] - log Q[j,i]. In 2D/ND milestone
    graphs with cycles, the empirical Q entries from finite sampling do
    not exactly satisfy detailed balance. The least-squares solution
    minimizes total squared inconsistency across all observed pairs
    simultaneously, weighted by the harmonic mean of the crossing counts
    N_ij and N_ji (proportional to 1/variance under Poisson statistics).
    This gives an optimal estimate that is independent of which spanning
    tree is chosen and favors well-sampled edges over poorly-sampled ones.

    Parameters
    ----------
    num_states : int
        Number of milestone states.
    Q : np.ndarray
        Rate matrix with zero-substituted entries for unobserved
        transitions (used only to compute the free energy profile).
    temperature : float
        System temperature (units must match GAS_CONSTANT).
    Q_raw : np.ndarray | None, optional
        Rate matrix *before* zero-substitution of missing entries.
        When provided, only genuinely observed transitions (both
        directions positive) contribute equations to the least-squares
        system. Milestones unreachable from the root via observed edges
        receive probability 0 (infinite free energy).
    N_ij : DefaultDict | None, optional
        Raw milestone crossing counts N_ij[(i, j)] before any
        zero-substitution. When provided, each log-ratio equation is
        weighted by the harmonic mean of N_ij[(i,j)] and N_ij[(j,i)],
        so well-sampled edges dominate the solution. When None, all
        edges are given equal weight (equivalent to unweighted LS).
    """
    
    Q_bfs = Q_raw if Q_raw is not None else Q

    # Find the first state with any observed outgoing rate as the root.
    root = None
    for i in range(num_states):
        if any(Q_bfs[i, j] > 0.0 for j in range(num_states) if j != i):
            root = i
            break

    log_pi = np.full(num_states, -np.inf)

    if root is not None:
        # Collect all pairs (i,j) where both directions are observed.
        edges = []
        for i in range(num_states):
            for j in range(i + 1, num_states):
                k_fwd = Q_bfs[i, j]
                k_bwd = Q_bfs[j, i]
                if k_fwd > 0.0 and k_bwd > 0.0:
                    edges.append((i, j, np.log(k_fwd) - np.log(k_bwd)))

        if len(edges) > 0:
            # Build weighted least-squares system. Weight each equation by
            # the harmonic mean of N_ij and N_ji (≈ 1/var of log-ratio).
            n_edges = len(edges)
            A = np.zeros((n_edges + 1, num_states))
            b_vec = np.zeros(n_edges + 1)
            total_weight = 0.0
            for row, (i, j, delta) in enumerate(edges):
                if N_ij is not None:
                    N_fwd = N_ij.get((i, j), 1)
                    N_bwd = N_ij.get((j, i), 1)
                    denom = N_fwd + N_bwd
                    w = (N_fwd * N_bwd / denom) if denom > 0 else 1.0
                else:
                    w = 1.0
                sqrt_w = np.sqrt(max(w, 1e-12))
                total_weight += w
                A[row, j] = sqrt_w
                A[row, i] = -sqrt_w
                b_vec[row] = delta * sqrt_w
            # Gauge constraint: log_pi[root] = 0.
            gauge_w = np.sqrt(total_weight / n_edges) if n_edges > 0 else 1.0
            A[n_edges, root] = gauge_w
            b_vec[n_edges] = 0.0
            log_pi_sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            log_pi_sol -= log_pi_sol[root]  # enforce gauge

            # BFS to identify milestones reachable from root via observed
            # bidirectional edges; isolated components stay at -inf.
            reachable = {root}
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for other in range(num_states):
                    if other not in reachable:
                        if Q_bfs[node, other] > 0.0 and Q_bfs[other, node] > 0.0:
                            reachable.add(other)
                            queue.append(other)
            for idx in reachable:
                log_pi[idx] = log_pi_sol[idx]
        else:
            # No bidirectional edges: only the root state is known.
            log_pi[root] = 0.0

    # Normalize using log-sum-exp for numerical stability.
    finite_mask = np.isfinite(log_pi)
    if not np.any(finite_mask):
        raise ValueError(
            "No milestone transitions observed: cannot compute p_i.")
    log_total = np.logaddexp.reduce(log_pi[finite_mask])
    p_i = np.where(finite_mask, np.exp(log_pi - log_total), 0.0)

    highest_p_i = np.max(p_i[finite_mask])
    free_energy_profile = np.full(num_states, np.inf)
    for i, p_i_val in enumerate(p_i):
        if p_i_val > 0.0:
            free_energy_profile[i] = -GAS_CONSTANT * temperature * np.log(
                p_i_val / highest_p_i)
    
    return p_i, free_energy_profile

def calculate_kinetics(
        num_milestones: int,
        Q: np.ndarray,
        K: np.ndarray,
        end_states: typing.DefaultDict[str, list[int]],
        p_i: np.ndarray | None = None,
):
    """
    Once the rate matrix Q is computed, determine the timescales 
    and probabilities of transfers between different states. Fill
    out all kinetics quantities.

    If p_i is provided, milestones with p_i[i] == 0 (unreachable in the
    observed graph) are excluded from kinetics: their MFPTs become NaN
    and their absorption probabilities are zero. This relies on Q having
    been filled with detailed-balance rates via impute_missing_rates so
    that observed-but-asymmetric crossings do not poison the diagonal.
    """
    assert len(end_states) > 0, "No end (bound or otherwise) states "\
        "defined in this model. Kinetics calculations will not work."
    row_sum_max = float(np.max(np.abs(Q.sum(axis=1))))
    if row_sum_max > 1.E-8:
        # Soft clamp: renormalize diagonals so each row sums to zero
        # exactly. Imputation should have already enforced this, so a
        # large residual here likely indicates a deeper sampling issue.
        Q = copy.deepcopy(Q)
        for i in range(num_milestones):
            off_diag = np.sum(Q[i], dtype=np.longdouble) - Q[i, i]
            Q[i, i] = -off_diag
        K = rate_matrix_to_transition_matrix(Q)

    if p_i is None:
        reachable = set(range(num_milestones))
    else:
        reachable = {i for i in range(num_milestones) if p_i[i] > 0.0}

    # Build a Q (and K) restricted to the reachable milestones so that
    # PyGT does not divide by the zero escape rates of unreachable rows
    # and propagate NaN into the reachable MFPTs.
    reachable_list = sorted(reachable)
    n_red = len(reachable_list)
    red_idx = {m: i for i, m in enumerate(reachable_list)}
    Q_red = np.zeros((n_red, n_red), dtype=np.float64)
    for i, m in enumerate(reachable_list):
        for j, m2 in enumerate(reachable_list):
            Q_red[i, j] = float(Q[m, m2])
    # Rebuild diagonals against the reachable subgraph so rows sum to zero
    # exactly. Otherwise Q[i,i] (which encodes escape to all milestones in
    # the full graph) overcounts escape relative to Q_red's off-diagonals,
    # and the derived branching matrix K_red is sub-stochastic.
    for i in range(n_red):
        Q_red[i, i] = 0.0
        Q_red[i, i] = -np.sum(Q_red[i, :])
    K_red = rate_matrix_to_transition_matrix(Q_red)

    # Collect all end-state milestone indices across all states
    all_end_milestone_indices = []
    for milestone_list in end_states.values():
        all_end_milestone_indices.extend(milestone_list)

    milestone_src_to_milestone_dest_MFPTs = {}
    committor_function_by_milestone = {}
    end_state_to_end_state_probabilities_by_milestone = {}
    # Compute MFPTs and committor functions (dest-centric)
    for dest_state_name, dest_milestone_list in end_states.items():
        for src_state_name, src_milestone_list in end_states.items():
            if src_state_name == dest_state_name:
                continue

            if n_red > 0:
                B, tau = PyGT.tools.load_CTMC(Q_red.T)
            else:
                B, tau = None, None
            
            for dest_milestone_index in dest_milestone_list:
                for src_milestone_index in src_milestone_list:
                    if dest_milestone_index == src_milestone_index:
                        continue
                    if src_milestone_index not in reachable \
                            or dest_milestone_index not in reachable:
                        # Unreachable in the observed graph: store 0 so
                        # downstream probability-weighted sums (which will
                        # also see p_i == 0 for this milestone) remain
                        # numerically clean rather than producing NaN.
                        milestone_src_to_milestone_dest_MFPTs[
                            (src_milestone_index, dest_milestone_index)] = 0.0
                    else:
                        src_red = red_idx[src_milestone_index]
                        dest_red = red_idx[dest_milestone_index]
                        mfpt_ij, mfpt_ji = PyGT.mfpt.compute_MFPT(
                            src_red, dest_red, B, tau)
                        milestone_src_to_milestone_dest_MFPTs[
                            (src_milestone_index, dest_milestone_index)] = mfpt_ji

                # Committor q[i] = P(reach dest before any src | start at i),
                # computed via graph transformation on an absorbing DTMC.
                # Make dest and all src milestones absorbing (self-loops) in
                # the row-stochastic branching matrix K_red, then for each
                # starting reachable transient i, GT-eliminate the others;
                # q[i] = B_gt[dest, i] / (1 - B_gt[i, i]) on column i. GT is
                # numerically stable at high barriers where matrix_power
                # underflows.
                committor = np.zeros(num_milestones, dtype=np.float64)
                if n_red > 0 and dest_milestone_index in reachable:
                    dest_red = red_idx[dest_milestone_index]
                    src_reds = [red_idx[m] for m in src_milestone_list
                                if m in reachable and m != dest_milestone_index]
                    absorbers_red = {dest_red} | set(src_reds)
                    K_abs = np.array(K_red, dtype=np.float64)
                    for a in absorbers_red:
                        K_abs[a, :] = 0.0
                        K_abs[a, a] = 1.0
                    B_abs, tau_abs = PyGT.tools.load_DTMC(K_abs.T, 1.0)
                    committor[dest_milestone_index] = 1.0
                    for i_red, m_i in enumerate(reachable_list):
                        if i_red in absorbers_red:
                            continue
                        keep = absorbers_red | {i_red}
                        rm_vec = np.array(
                            [k not in keep for k in range(n_red)], dtype=bool)
                        B_gt, _ = PyGT.GT.blockGT(
                            rm_vec, B_abs.copy(), tau_abs.copy())
                        B_gt = np.asarray(B_gt, dtype=np.float64)
                        keep_ordered = sorted(keep)
                        new_idx = {old: k for k, old in enumerate(keep_ordered)}
                        ii = new_idx[i_red]
                        dd = new_idx[dest_red]
                        escape = 1.0 - float(B_gt[ii, ii])
                        if escape > 0.0:
                            committor[m_i] = min(1.0, max(
                                0.0, float(B_gt[dd, ii]) / escape))
                committor_function_by_milestone[dest_milestone_index] = committor

    # Compute absorption probabilities (src-centric) via PyGT graph
    # transformation. For each src_milestone, GT-eliminate every reachable
    # transient milestone other than src and the absorbers. The reduced
    # branching matrix B_gt then satisfies, with columns summing to 1,
    #     P(src → absorbed at a) = B_gt[a, src] / (1 - B_gt[src, src]).
    # GT is numerically stable at arbitrary barrier heights, where the
    # direct (I - K_TT) linear solve underflows for rare-event systems.
    # Milestones shared between two end states (Voronoi boundaries between
    # adjacent end-state anchors) are NOT excluded from absorption, so that
    # flux crossing them is correctly captured as reaching the dest state.
    if n_red > 0:
        B_full, tau_full = PyGT.tools.load_CTMC(Q_red.T)
    for src_state_name, src_milestone_list in end_states.items():
        exclusive_src_milestones = set(src_milestone_list)
        for other_state_name, other_milestone_list in end_states.items():
            if other_state_name != src_state_name:
                exclusive_src_milestones -= set(other_milestone_list)
        for src_milestone_index in src_milestone_list:
            probs_at_absorbers: dict[int, float] = {}
            if src_milestone_index in reachable:
                absorbing = {m for m in all_end_milestone_indices
                             if m != src_milestone_index
                             and m not in exclusive_src_milestones
                             and m in reachable}
                src_red = red_idx[src_milestone_index]
                absorber_red = {red_idx[m]: m for m in absorbing}
                keep = {src_red} | set(absorber_red.keys())
                rm_vec = np.array(
                    [i not in keep for i in range(n_red)], dtype=bool)
                B_gt, _ = PyGT.GT.blockGT(
                    rm_vec, copy.deepcopy(B_full), tau_full.copy())
                B_gt = np.asarray(B_gt, dtype=np.float64)
                keep_ordered = sorted(keep)
                new_idx = {old: k for k, old in enumerate(keep_ordered)}
                src_new = new_idx[src_red]
                total_escape = 1.0 - float(B_gt[src_new, src_new])
                for a_red, m in absorber_red.items():
                    if total_escape > 0.0:
                        prob = float(B_gt[new_idx[a_red], src_new]) \
                            / total_escape
                    else:
                        prob = 0.0
                    probs_at_absorbers[m] = min(1.0, max(0.0, prob))
            for dest_state_name, dest_milestone_list in end_states.items():
                if dest_state_name == src_state_name:
                    continue
                for dest_milestone_index in dest_milestone_list:
                    end_state_to_end_state_probabilities_by_milestone[
                        (src_milestone_index, dest_milestone_index)] = \
                            probs_at_absorbers.get(dest_milestone_index, 0.0)

    return milestone_src_to_milestone_dest_MFPTs, committor_function_by_milestone, \
        end_state_to_end_state_probabilities_by_milestone

def get_end_states_not_bulk(
        model: structures.Seekr_model
    ) -> typing.Dict[str, list[int]]:
    """
    Get a dictionary of end states that are not bulk states.
    """
    end_states_not_bulk = defaultdict(list)
    for anchor1 in model.anchors:
        if anchor1.bulk_anchor:
            continue
        if anchor1.end_anchor:
            for milestone in anchor1.milestones:
                end_states_not_bulk[anchor1.name].append(milestone.index)
    return end_states_not_bulk

def calculate_k_on_kinetics(
        num_milestones: int,
        K: np.ndarray,
        end_states_not_bulk: typing.DefaultDict[str, list[int]],
        transition_counts: dict,
        transition_probabilities: dict,
        k_ons_src: dict,
        outer_milestone_indices: list[int],
    ) -> dict:
    """
    Calculate k_on kinetics if bulk anchor stats present.
    """
    assert len(end_states_not_bulk) > 0, "No end (bound or otherwise) states "\
        "defined in this model. K-on calculations will not work."
    k_ons = defaultdict(float)
    K_hat_all = copy.deepcopy(K)
    for state_name, end_milestones in end_states_not_bulk.items():
        for end_milestone in end_milestones:
            assert 0 <= end_milestone < num_milestones, \
                "End milestone {} is out of bounds.".format(end_milestone)
            K_hat_all[end_milestone, :] = 0.0
            K_hat_all[end_milestone, end_milestone] = 1.0

    if len(transition_counts) > 0:
        source_vec = np.zeros((num_milestones, 1))
        for outer_milestone_index in outer_milestone_indices:
            k_on_value = k_ons_src[outer_milestone_index]
            source_vec[outer_milestone_index] = k_on_value
            transition_probs_milestone = transition_probabilities[outer_milestone_index]
            K_hat_all[outer_milestone_index, :] = 0.0
            for inner_milestone_index, value in transition_probs_milestone.items():
                if inner_milestone_index in ["escaped", "total", "stuck"]:
                    continue
                K_hat_all[outer_milestone_index, inner_milestone_index] = value

        # Absorption probabilities via PyGT graph transformation. Pad with
        # an extra "escape" absorbing state so rows summing to < 1 (outer
        # milestones with excluded escape/stuck/total mass, or unobserved
        # milestones with zero rates) yield a column-stochastic transition
        # matrix that PyGT can ingest. Then eliminate every milestone that
        # is neither an outer source nor an end-state absorber in a single
        # blockGT call; the reduced B_gt gives, for each outer m, the
        # absorption probability at each end as
        #     B_gt[end, m] / (1 - B_gt[m, m]).
        N = num_milestones + 1
        escape_idx = num_milestones
        K_padded = np.zeros((N, N), dtype=np.float64)
        K_padded[:num_milestones, :num_milestones] = np.asarray(
            K_hat_all, dtype=np.float64)
        for i in range(num_milestones):
            s = float(K_padded[i, :num_milestones].sum())
            K_padded[i, escape_idx] = max(0.0, 1.0 - s)
        K_padded[escape_idx, escape_idx] = 1.0
        B_full, tau_full = PyGT.tools.load_DTMC(K_padded.T, 1.0)
        all_end_milestones: list[int] = []
        for ends in end_states_not_bulk.values():
            all_end_milestones.extend(ends)
        keep = set(outer_milestone_indices) | set(all_end_milestones) \
            | {escape_idx}
        rm_vec = np.array([k not in keep for k in range(N)], dtype=bool)
        B_gt, _ = PyGT.GT.blockGT(rm_vec, B_full.copy(), tau_full.copy())
        B_gt = np.asarray(B_gt, dtype=np.float64)
        keep_ordered = sorted(keep)
        new_idx = {old: k for k, old in enumerate(keep_ordered)}
        for state_name, end_milestones in end_states_not_bulk.items():
            for end_milestone in end_milestones:
                ee = new_idx[end_milestone]
                for outer_milestone_index in outer_milestone_indices:
                    oo = new_idx[outer_milestone_index]
                    escape = 1.0 - float(B_gt[oo, oo])
                    if escape <= 0.0:
                        continue
                    prob = min(1.0, max(
                        0.0, float(B_gt[ee, oo]) / escape))
                    k_ons[state_name] += prob \
                        * float(source_vec[outer_milestone_index, 0])

    return k_ons

def pretty_string_value_error(
        value: float | None, 
        error: float | None, 
        error_digits: int = 2, 
        use_unicode: bool = True
        ) -> str:
    """
    Returns a value/error combination of numbers in a scientifically 
    'pretty' format.
    
    Scientific quantities often come as a *value* (the actual 
    quantity) and the *error* (the uncertainty in the value).
        
    Given two floats, value and error, return the two in a 
    'pretty' formatted string: where the value and error are truncated
    at the correct precision.
    
    Parameters
    ----------
    value : float
        The quantity in question
        
    error : float
        The uncertainty of the quantity
        
    error_digits : int, default 2
        How many significant figures the error has. Scientific 
        convention holds that errors have 1 or (at most) 2 significant
        figures. Two digits is chosen here by default.
        
    Returns
    -------
    new_string : str
        A new list of strings sorted numerically
    
    Examples
    --------
    
        >>> pretty_string_value_error(1.23456789e8, 4.5678e5, 
                                      error_digits=2)
        "1.2346 +/- 0.0046 * 10^+08"
    
        >>> pretty_string_value_error(5.6e-2, 2.0e-3, error_digits=1)
        "5.6 +/- 0.2 * 10^-02"
    """
    if value is None:
        return "None"
    if error is None or not np.isfinite(error):
        if use_unicode:
            new_string = "{:.6E} \u00B1 UNKNOWN ERROR MARGIN".format(value)
        else:
            new_string = "{:.6E} +/- UNKNOWN ERROR MARGIN".format(value)
    else:
        if not np.isfinite(value):
            return str(value)
        assert "e" in "{:e}".format(value), "Cannot convert into scientific "\
            "notation: {1}".format(value)
        value_mantissa_str, value_exponent_str = \
            "{:e}".format(value).strip().split('e')
        value_mantissa = float(value_mantissa_str)
        value_exponent = int(value_exponent_str)
        error_mantissa_str, error_exponent_str = \
            "{:e}".format(error).strip().split('e')
        error_mantissa = float(error_mantissa_str)
        error_exponent = int(error_exponent_str)
        padding = value_exponent - error_exponent + error_digits - 1
        if padding < 1: padding = 1
        exp_diff = error_exponent - value_exponent
        string_for_formatting = "{:.%df}" % padding
        new_value_mantissa = string_for_formatting.format(value_mantissa)
        new_error_mantissa = string_for_formatting.format(
            error_mantissa*10**exp_diff)
        if use_unicode:
            new_string = "(%s \u00B1 %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
        else:
            new_string = "(%s +/- %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
    return new_string

def quadriture(
        err1: float | None, 
        err2: float | None
    ) -> float | None:
    """Add two errors in quadriture."""
    if err1 is None or err2 is None:
        return None
    return float(np.sqrt(err1**2 + err2**2))

def make_milestone_value_dict(
        model: structures.Seekr_model
    ) -> dict[int, float]:
        """
        Make a dictionary with milestone indices as keys, and
        milestone values as dictionary values.
        """
        milestone_value_dict = {}
        for i, anchor in enumerate(model.anchors):
            for milestone in anchor.milestones:
                value = milestone.variables["avg_value"]
                if milestone.index not in milestone_value_dict:
                    milestone_value_dict[milestone.index] = value
                    
        return milestone_value_dict

def plot_scalar_conv(
        conv_values: np.ndarray, 
        conv_times: np.ndarray,
        filename: str,
        label: str, 
        title: str, 
        conv_errors: dict | None = None,
        y_axis_logarithmic: bool = True,
        return_plot: bool = True,
        xlabel: str = "time per anchor (ns)",
        ) -> None | str | tuple:
    """
    Plot convergence of off/on rate or other scalar values as a 
    function of simulation time.

    Parameters
    ----------
    conv_values : list
        list of calculated scalar values for each convergence interval

    conv_times : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label : str
        The label to give this plot.
    
    title : str
        The title of this plot.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds.
        
    y_axis_logarithmic : bool, default True
        Whether the y-axis will be plotted on a logarithmic scale.

    Returns
    -------
    fig : matplotlib figure
        matplotlib figure plotting N convergence for each milestone
    ax : object
        matplotlib Axes object
    """
    import matplotlib.pyplot as plt
    #if not np.any(np.isfinite(conv_values)) or np.all(conv_values == 0):
    #    return None, None
    for i, conv_value in enumerate(conv_values):
        if not np.isfinite(conv_value) or conv_value == 0:
            conv_values[i] = np.nan
    
    if not np.any(np.isfinite(conv_values)):
        #print("skipping key:", key, "because values aren't finite")
        if return_plot:
            return None, None
        else:
            return None
    
    fig, ax = plt.subplots()
    ax.plot(conv_times, conv_values, 
            linestyle="-", marker="o", markersize=1)
    if conv_errors is not None:
        ax.fill_between(conv_times, conv_values - conv_errors, 
                            conv_values + conv_errors, alpha=0.2)
    x_lim = ax.get_xlim()
    upper_limit = x_lim[1]
    ax.set_xlim((0.0, upper_limit))
    plt.ylabel("$"+label+"$")
    # TODO: change this to plotting over time
    plt.xlabel(xlabel)
    plt.title(title)
    if y_axis_logarithmic:
        plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    if return_plot:
        return fig, ax
    else:
        plt.savefig(filename)
        plt.close()
        return filename

def plot_2d_conv(
        conv_dict: dict, 
        conv_times: np.ndarray, 
        filename: str,
        title: str, 
        xlabel: str,
        zlabel: str,
        conv_errors: dict | None = None,
        y_axis_logarithmic: bool = True,
        num_convergence_steps: int | None = 8,
        return_plot: bool = True,
        ylabel: str = "time (ns)"
        ) -> None | str | tuple:
    """
    
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    x_axis = sorted(conv_dict.keys())
    if num_convergence_steps is None:
        indices = np.arange(len(conv_times))
        y_axis = conv_times
    else:
        indices = np.linspace(0, len(conv_times) - 1, num_convergence_steps, dtype=int)
        y_axis = conv_times[indices]

    y_arr = np.asarray(y_axis, dtype=float)
    cmap = matplotlib.cm.berlin.reversed()   # or 'cool', 'winter', 'copper', etc.
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(y_arr), vmax=np.nanmax(y_arr))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j, y_val in zip(indices, y_axis):
        x_s = []
        y_s = []
        z_s = []
        z_err_lowers = []
        z_err_uppers = []
        for i, x_val in enumerate(x_axis):
            if y_axis_logarithmic:
                if conv_dict[i][j] > 0:
                    z_s.append(np.log10(conv_dict[i][j]))
                    if conv_errors is not None:
                        #z_errs.append(np.log10(conv_errors[i][j]))
                        lower = conv_dict[i][j] - conv_errors[i][j]
                        if lower <= 0.0:
                            lower = conv_dict[i][j]
                        upper = conv_dict[i][j] + conv_errors[i][j]
                        if upper <= 0.0:
                            upper = conv_dict[i][j]
                        z_err_lowers.append(np.log10(lower))
                        z_err_uppers.append(np.log10(upper))
                    y_s.append(y_val)
                    x_s.append(x_val)
            else:
                z_s.append(conv_dict[i][j])
                if conv_errors is not None:
                    #z_errs.append(conv_errors[i][j])
                    z_err_lowers.append(conv_dict[i][j] - conv_errors[i][j])
                    z_err_uppers.append(conv_dict[i][j] + conv_errors[i][j])
                y_s.append(y_val)
                x_s.append(x_val)

        x_s = np.array(x_s)
        y_s = np.array(y_s)
        z_s = np.array(z_s)
        z_err_lowers = np.array(z_err_lowers)
        z_err_uppers = np.array(z_err_uppers)

        if len(x_s) > 0 and np.all(np.isfinite(z_s)) \
                and np.all(np.isfinite(x_s)) and np.all(np.isfinite(y_s)):
            color = cmap(norm(float(y_val)))
            ax.plot(x_s, y_s, z_s, zdir="z", color=color, alpha=0.75)
            if conv_errors is not None:
                for xi, yi, e_lower, e_upper in zip(x_s, y_s, z_err_lowers, z_err_uppers):
                    ax.plot([xi, xi], [yi, yi], [e_lower, e_upper], 
                            color=color, alpha=0.75)
            if y_axis_logarithmic:
                zticks = ax.get_zticks()
                ztick_labels = [f"{10**tick:.3f}" for tick in zticks]
                ax.set_zticks(zticks) # suppresses warning
                ax.set_zticklabels(ztick_labels)

    upper_limit = ax.get_ylim()[1]
    ax.set_ylim((upper_limit, 0))
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_zlabel(zlabel + (" (log₁₀ scale)" if y_axis_logarithmic else ""), labelpad=14)
    plt.title(title)
    if return_plot:
        return fig, ax
    else:
        plt.savefig(filename)
        plt.close(fig)
        return filename

def plot_dict_conv(
        conv_dict: dict, 
        conv_times: np.ndarray, 
        directory: str,
        label_base: str,
        unit: str, 
        conv_errors: dict | None = None,
        skip_null: bool = True,
        y_axis_logarithmic: bool = True,
        title_suffix: str = "", 
        name_base: str | None = None,
        draw_double: bool = True,
        return_plot: bool = True,
        xlabel: str = "time per anchor (ns)",
        ) -> list:
    """
    Plot convergence of N_ij or R_i or other dictionary-based value 
    as a function of simulation time.

    Parameters
    ----------
    conv_dict : dict
        dict of lists of calculated off rates for each convergence 
        interval.
        
    conv_interval : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label_base : str
        The base of the label to give this plot, though the dictionary
        keys will be appended to the label.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds
    
    skip_null : bool, Default True
        If true, than empty convergence lists will be omitted from
        any plots.
    
    y_axis_logarithmic : bool, True
        Whether the y-axis will be plotted on a logarithmic scale.
    
    Returns
    -------
    fig_list : list
        A list of matplotlib figures plotting convergence for each 
        milestone.
        
    ax_list : list
        A list of matplotlib Axes objects.
        
    title_list : list
        A list of the plots' titles.
    
    name_list : list
        A list of the plots' file names.
    """
    import matplotlib.pyplot as plt
    fig_list = []
    ax_list = []
    title_list = []
    name_list = []
    plot_filenames = []
    if return_plot:
        plot_list = []

    if name_base is None:
        name_base = label_base
        
    for key in conv_dict:
        conv_values = conv_dict[key]
        conv_error = None
        if conv_errors is not None:
            if key in conv_errors:
                conv_error = conv_errors[key]

        if not np.all(np.isfinite(conv_values)):
            continue
        if skip_null:
            if np.linalg.norm(conv_values) < MIN_PLOT_NORM:
                continue
        if unit == "":
            unitstr = ""
        else:
            unitstr = "(" + unit +")"
        if isinstance(key, tuple):
            # If right arrow is preferred
            #label = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}(" + unit +")$"
            #title = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}$" + title_suffix
            
            label = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}$" + title_suffix
            name = name_base + "_" + "_".join(map(str, key))
        elif isinstance(key, int) or isinstance(key, str):
            label = "$" + label_base + "_{" + str(key) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + str(key) + "}$" + title_suffix
            name = name_base + "_" + str(key) + ""
        else:
            raise Exception("key type not implemented: {}".format(type(key)))
        
        fig, ax1 = plt.subplots()
        x_data = conv_times
        ax1.plot(conv_times, conv_values, 
                linestyle='-', marker="o", color="r", markersize = 1)
        if conv_error is not None:
            ax1.fill_between(conv_times, conv_values - conv_error, 
                             conv_values + conv_error, color="r", alpha=0.2)
        ax1.set_ylabel(label, color="r")
        ax1.set_xlabel(xlabel)
        if y_axis_logarithmic:
            plt.yscale("log", nonpositive="mask")
        ax_list.append(ax1)
        if draw_double:
            ax2 = ax1.twinx()
            ax2.set_ylabel(label + "/time(ps)", color="b")
            data_per_time = np.zeros(conv_times.shape)
            data_per_time_err_upper = np.zeros(conv_times.shape)
            data_per_time_err_lower = np.zeros(conv_times.shape)
            for i in range(len(conv_times)):
                if x_data[i] == 0:
                    divisor = 1
                else:
                    divisor = x_data[i]
                data_per_time[i] = conv_values[i] / divisor
                if conv_error is not None:
                    data_per_time_err_lower[i] = (conv_values[i] - conv_error[i]) / divisor
                    data_per_time_err_upper[i] = (conv_values[i] + conv_error[i]) / divisor
            ax2.plot(x_data, data_per_time, 
                linestyle='-', marker="o", color="b", markersize = 1)
            if conv_error is not None:
                ax2.fill_between(x_data, data_per_time_err_lower, 
                                 data_per_time_err_upper, color="b", alpha=0.2)
            if y_axis_logarithmic:
                plt.yscale("log", nonpositive="mask")
            ax_list.append(ax2)

        x_lim = ax1.get_xlim()
        upper_limit = x_lim[1]
        ax1.set_xlim((0.0, upper_limit))

        plt.title(title)
        fig_list.append(fig)
        title_list.append(title)
        name_list.append(name)
        plt.tight_layout()
        if return_plot:
            plot_list.append((fig, ax1))
        else:
            filename = os.path.join(directory, name + ".png")
            plt.savefig(filename)
            plot_filenames.append(filename)
            plt.close(fig)
    if return_plot:
        return plot_list
    else:
        return plot_filenames

def find_anchor_points_2d(
        model: structures.Seekr_model, 
        cv_indices: tuple[int, int] = (0,1)
        ) -> dict:
    anchor_values = {}
    num_variables = len(model.collective_variables)
    for alpha, anchor in enumerate(model.anchors):
        values = []
        for i in range(num_variables):
            cv_index = cv_indices[i]
            var_name = "value_{}".format(cv_index)
            values.append(anchor.variables[var_name])
        anchor_values[alpha] = values
        
    return anchor_values

def find_milestone_points_2d(
        model: structures.Seekr_model, 
        vor: any, 
        boundaries: list[float]
        ) -> dict:
    milestone_points = {}
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
    for i, ridge_point_pair in enumerate(vor.ridge_points):
        anchor1_index = ridge_point_pair[0]
        anchor2_index = ridge_point_pair[1]
        if (anchor1_index >= len(model.anchors)) \
                or (anchor2_index >= len(model.anchors)):
            continue
        
        anchor1 = model.anchors[anchor1_index]
        milestone_index = None
        for milestone in anchor1.milestones:
            if milestone.neighbor_anchor_index == anchor2_index:
                milestone_index = milestone.index
                break
            
        if milestone_index is None:
            continue
        
        ridge_vertex_pair = vor.ridge_vertices[i]
        for ridge_vertex_index in ridge_vertex_pair:
            vertex_inside_boundaries = True
            vertex = vor.vertices[ridge_vertex_index]
            if (vertex[0] < min_x) or (vertex[0] > max_x) \
                    or (vertex[1] < min_y) or (vertex[1] > max_y):
                vertex_inside_boundaries = False
                
            if (ridge_vertex_index == -1) or (not vertex_inside_boundaries):
                ridge_vertex_pair.remove(ridge_vertex_index)
        
        if len(ridge_vertex_pair) == 0:
            continue
        elif len(ridge_vertex_pair) == 2:
            vertex1_index = ridge_vertex_pair[0]
            vertex2_index = ridge_vertex_pair[1]
            vertex1 = vor.vertices[vertex1_index]
            vertex2 = vor.vertices[vertex2_index]
            avg_point_x = 0.5 * (vertex1[0] + vertex2[0])
            avg_point_y = 0.5 * (vertex1[1] + vertex2[1])
            avg_point = [avg_point_x, avg_point_y]
            milestone_points[milestone_index] = avg_point
        else:
            adj_point1 = vor.points[anchor1_index]
            adj_point2 = vor.points[anchor2_index]
            avg_point_x = 0.5 * (adj_point1[0] + adj_point2[0])
            avg_point_y = 0.5 * (adj_point1[1] + adj_point2[1])
            avg_point = [avg_point_x, avg_point_y]
            milestone_points[milestone_index] = avg_point
    
    return milestone_points

def make_boundaries(
        anchor_values_dict: dict
        ) -> list[float]:
    """
    Create boundaries for the anchor values in 2D space.
    """
    anchor_values_list = list(anchor_values_dict.values())
    margin = 0.2
    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    for anchor_value_pair in anchor_values_list:
        val_x = anchor_value_pair[0]
        val_y = anchor_value_pair[1]
        if val_x < min_x:
            min_x = val_x
        if val_x > max_x:
            max_x = val_x
        if val_y < min_y:
            min_y = val_y
        if val_y > max_y:
            max_y = val_y
    
    boundaries = [min_x - margin, max_x + margin, min_y - margin, 
                  max_y + margin]
    return boundaries

def plot_voronoi_tesselation(
        model: structures.Seekr_model, 
        boundaries: list[float], 
        fill_values: np.ndarray | None = None, 
        colorbar_label: str = "Energy (kcal/mol)", 
        anchor_values: dict | None = None
        ) -> tuple:
    """
    Would work for either anchors or milestones
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d
    if anchor_values is None:
        anchor_values = find_anchor_points_2d(model)
        
    points = []
    for alpha, anchor in enumerate(model.anchors):
        assert alpha in anchor_values
        points.append(anchor_values[alpha])
        
    points.append([-100.0, -100.0])
    points.append([-100.0, 100.0])
    points.append([100.0, -100.0])
    points.append([100.0, 100.0])
    
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
    
    ax = plt.gca()
    ax.axis(boundaries)
    ax.set_aspect('equal', adjustable='box')
    
    if fill_values is not None:
        fill_values_normalized = {}
        min_fill_values = min(list(fill_values.values()))
        max_fill_values = max(list(fill_values.values()))
        for key, value in fill_values.items():
            fill_values_normalized[key] = (value - min_fill_values)\
                /(max_fill_values - min_fill_values)
        
    for i in range(len(points)):
        my_point_region = vor.point_region[i]
        region = vor.regions[my_point_region]
        if not -1 in region:
            polygon = [vor.vertices[j] for j in region]
            #if len(polygon) == 0: continue
            if fill_values is not None:
                jet = plt.cm.jet
                colormap = jet(fill_values_normalized[i])
            else:
                colormap = VT_BACKGROUND_COLOR
            plt.fill(*zip(*polygon), color=colormap)
                
    if fill_values is not None:
        norm = matplotlib.colors.Normalize(min_fill_values, max_fill_values)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=jet)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label(colorbar_label)
    
    return fig, ax, vor, points[:-4]

def plot_anchor_voronoi_tesselation(
        model: structures.Seekr_model, 
        boundaries: list[float], 
        anchor_fill_values: dict | None = None,
        colorbar_label: str = "Energy (kcal/mol)", 
        anchor_values: dict | None = None
        ) -> tuple:
    
    fig, ax, vor, points = plot_voronoi_tesselation(
        model, boundaries, fill_values=anchor_fill_values, 
        colorbar_label=colorbar_label, anchor_values=anchor_values)
    
    return fig, ax, points

def plot_anchors_2d(
        model: structures.Seekr_model, 
        anchor_values: dict,
        boundaries: list, 
        title: str,
        x_coordinate_title: str,
        y_coordinate_title: str,
        colorbar_label: str = "Energy (kcal/mol)",
        fill_values: dict | None = None,
        dpi: int = 200,
        plot_basename: str = "vt.png",
        plot_dir: str | None = None, 
        return_plot: bool = True,
        ) -> None | str | tuple:
    """
    
    """
    import matplotlib.pyplot as plt
    fig, ax, points = plot_anchor_voronoi_tesselation(
        model, boundaries, fill_values, colorbar_label=colorbar_label, 
        anchor_values=anchor_values)
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
    if model.using_toy() and fill_values is None:
        toy_settings = model.get_toy_settings()
        potential_energy_expression \
            = refine_function_str(
                toy_settings.potential_energy_expression)
        
        landscape_resolution=100
        max_Z=20
        bounds_x = np.linspace(min_x, max_x, landscape_resolution)
        bounds_y = np.linspace(min_y, max_y, landscape_resolution)
        X,Y = np.meshgrid(bounds_x, bounds_y)
        Z = np.zeros((landscape_resolution, landscape_resolution))
        min_Z = np.min(Z) # 0.0
        for i, x1 in enumerate(bounds_x):
            for j, y1 in enumerate(bounds_y):
                Z[j,i] = eval(potential_energy_expression) - min_Z
                if Z[j,i] > max_Z:
                    Z[j,i] = max_Z
        
        p = ax.pcolor(X, Y, Z, cmap=plt.cm.jet, vmin=Z.min(), vmax=Z.max())
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
    
    ax.set_title(title)
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    
    if return_plot:
        return fig, ax
    else:
        plot_filename = os.path.join(plot_dir, plot_basename)
        plt.savefig(plot_filename, dpi=dpi)
        plt.close()
        return plot_filename

def plot_milestones_2d(
        model: structures.Seekr_model, 
        boundaries: list, 
        title: str,
        x_coordinate_title: str,
        y_coordinate_title: str,
        colorbar_label: str = "Energy (kcal/mol)",
        fill_values: dict | None = None,
        dpi: int = 200,
        plot_basename: str = "vt.png",
        plot_dir: str | None = None, 
        return_plot: bool = True,
        ) -> None | str | tuple:
        #model, plot_dir, iteration, boundaries, 
        #         title, x_coordinate_title, y_coordinate_title, 
        #         omit_iter_label=True, dpi=200, file_base="vt", 
        #         draw_string=False, fill_values=None, edge_dictionary=None,
        #         edge_labels=True):
    NODE_SIZE = 80
    import networkx as nx
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax, vor, anchor_points = plot_voronoi_tesselation(
        model, boundaries, fill_values=None)
    milestone_points = find_milestone_points_2d(model, vor, boundaries)
    ax.set_title(title)
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    
    pos = milestone_points
    G = nx.DiGraph(directed=True)
    milestone_labels = sorted(milestone_points.keys())
    G.add_nodes_from(milestone_labels)
    if fill_values is None:
        node_color = "black"
    else:
        #fill_values_normalized = (fill_values-np.min(fill_values))\
        #    /(np.max(fill_values)-np.min(fill_values))
        if fill_values is not None:
            fill_values_normalized = {}
            min_fill_values = min(list(fill_values.values()))
            max_fill_values = max(list(fill_values.values()))
            for key, value in fill_values.items():
                fill_values_normalized[key] = (value - min_fill_values)\
                    /(max_fill_values - min_fill_values)
        rainbow = plt.cm.rainbow
        node_color = []
        for i in range(model.num_milestones):
            node_color.append(rainbow(fill_values_normalized[i]))

        norm = matplotlib.colors.Normalize(min_fill_values, max_fill_values)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=rainbow)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label(colorbar_label)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=node_color)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4, font_color="black")
    
    if return_plot:
        return fig, ax
    else:
        plot_filename = os.path.join(plot_dir, plot_basename)
        plt.savefig(plot_filename, dpi=dpi)
        plt.close()
        return plot_filename

def _validate_and_assign_results(value, error, units):
    val_or_null = value if np.isfinite(value) else None
    err_or_null = error if np.isfinite(error) else None
    return {
        "value": val_or_null,
        "error": err_or_null,
        "units": units
    }