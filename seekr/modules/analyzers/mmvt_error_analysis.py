"""
modules/analyzers/mmvt_error_analysis.py

Procedures used for error analysis in MMVT.
"""

import copy
from collections import defaultdict

import numpy as np

import seekr.modules.analyzers.base as analyzers_base
import seekr.modules.structures as structures
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics
import seekr.modules.analyzers.mmvt_data_sample as mmvt_data_sample
import seekr.modules.analyzers.mmvt_data_window as mmvt_data_window
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics

def resample_anchor_statistics(
        anchor_stats_alpha_list: list[list[mmvt_anchor_statistics.Mmvt_anchor_statistics]],
        ) -> list[list[mmvt_anchor_statistics.Mmvt_anchor_statistics]]:
    """
    Resample the anchor statistics for each data window.
    """
    new_anchor_stats_alpha_list = []
    for alpha, anchor_stats_windows in enumerate(anchor_stats_alpha_list):
        single_window_step_count = anchor_stats_windows[0].step_count
        single_window_time_count = anchor_stats_windows[0].time_count
        dt = single_window_time_count / single_window_step_count
        new_anchor_stats_list = []
        for window_id, anchor_stats in enumerate(anchor_stats_windows):
            # For each data window, reconstruct the anchor statistics by
            #  resampling the MMVT anchor statistics from the probabilities
            #  predicted by the window's main data sample
            new_anchor_stats = mmvt_anchor_statistics.Mmvt_anchor_statistics(alpha)
            new_anchor_stats.first_encounter_time = 0.0

            # Construct a new sequence of transitions from a Markov matrix made
            # from the transitions within the anchor statistics
            ordered_keys = sorted(anchor_stats.N_alpha_beta.keys())
            if len(ordered_keys) == 0:
                # No keys - meaningless - skip
                continue
            
            key_indices = np.arange(len(ordered_keys))
            # Find sequence length by the number of transitions observed in
            #  the original anchor statistics
            sequence_length = 0
            avg_times_alpha_beta = {}
            for key in ordered_keys:
                sequence_length += anchor_stats.N_alpha_beta[key]
            
            for key in ordered_keys:
                avg_times_alpha_beta[key] = anchor_stats.T_alpha_total / \
                    sequence_length \
                    if sequence_length > 0 else 0.0
            
            # For starting probs
            count_vector = np.zeros(len(ordered_keys))
            # For transition probs
            count_matrix = np.zeros((len(ordered_keys), len(ordered_keys)))
            avg_times_i_j = {}
            
            
            for key_index in key_indices:
                for key_index2 in key_indices:
                    if key_index == key_index2:
                        continue # For now
                    key = ordered_keys[key_index]
                    key2 = ordered_keys[key_index2]
                    N_i_j_alpha = anchor_stats.N_i_j_alpha[(key, key2)]
                    count_matrix[key_index, key_index2] = N_i_j_alpha

                avg_times_i_j[key_index] = anchor_stats.R_i_alpha_total[key] / \
                    count_matrix[key_index,:].sum() \
                    if count_matrix[key_index,:].sum() > 0 else 0.0
                
            # NOTE: this is correct: N_i_j_alpha determines how many transitions
            #  to a different milestone, and N_alpha_beta contains all bounces.
            #  therefore, the value of the main diagonal is found by: 
            #  N_alpha_beta[alpha] - sum_over_j(N_i_j_alpha)
            for key_index in key_indices:
                row_sum = np.sum(count_matrix[:, key_index])
                count_matrix[key_index, key_index] \
                    = anchor_stats.N_alpha_beta[ordered_keys[key_index]] - row_sum
                count_vector[key_index] = anchor_stats.N_alpha_beta[ordered_keys[key_index]]
            
            initial_prob_vector = count_vector / np.sum(count_vector) \
                if np.sum(count_vector) > 0 else count_vector
            markov_trans_matrix = np.zeros(count_matrix.shape)
            for i in range(count_matrix.shape[0]):
                row_sum = np.sum(count_matrix[i,:])
                if row_sum == 0:
                    continue
                for j in range(count_matrix.shape[1]):
                    markov_trans_matrix[i, j] = count_matrix[i, j] / row_sum

            current_state = np.random.choice(len(initial_prob_vector), p=initial_prob_vector)
            sequence = [current_state]
            for _ in range(int(sequence_length)-1):
                probabilities = markov_trans_matrix[current_state]
                if (probabilities < 0).any():
                    print("alpha: ", alpha, " window_id: ", window_id)
                    print("anchor_stats: ", anchor_stats)
                    print("N_alpha_beta: ", anchor_stats.N_alpha_beta)
                    print("N_i_j_alpha: ", anchor_stats.N_i_j_alpha)
                    print("count_matrix: ", count_matrix)
                    print("markov_trans_matrix: ", markov_trans_matrix)
                    raise Exception("Negative probabilities in Markov matrix.")
                if np.sum(probabilities) == 0.0:
                    probabilities = initial_prob_vector
                    print("Warning: zero row in Markov matrix, using initial probabilities.")
                    print("initial_prob_vector: ", initial_prob_vector)

                elif np.sum(probabilities) != 1.0:
                    print("alpha: ", alpha, " window_id: ", window_id)
                    print("sequence_length:", sequence_length)
                    print("count_matrix: ", count_matrix)
                    print("probabilities: ", probabilities)
                    raise Exception("Probabilities do not sum to 1.0")
                next_state = np.random.choice(len(markov_trans_matrix), p=probabilities)
                current_state = next_state
                sequence.append(current_state)
            
            # First: N_alpha_beta, k_alpha_beta
            new_N_alpha_beta = defaultdict(int)
            new_N_i_j_alpha = defaultdict(int)
            new_R_i_alpha_total = defaultdict(float)
            new_T_alpha_total = 0.0
            for state_index in range(1, len(sequence)):
                prev_state = sequence[state_index-1]
                current_state = sequence[state_index]
                current_milestone_index = ordered_keys[current_state]
                previous_milestone_index = ordered_keys[prev_state]
                new_N_alpha_beta[current_milestone_index] += 1
                if prev_state != current_state:
                    new_N_i_j_alpha[(previous_milestone_index, current_milestone_index)] += 1
                    new_R_i_alpha_total[previous_milestone_index] += avg_times_i_j[prev_state]
                
                new_T_alpha_total += avg_times_alpha_beta[previous_milestone_index]

            new_anchor_stats.N_alpha_beta = new_N_alpha_beta
            new_anchor_stats.N_i_j_alpha = new_N_i_j_alpha
            new_anchor_stats.R_i_alpha_total = new_R_i_alpha_total
            new_anchor_stats.T_alpha_total = new_T_alpha_total
            new_anchor_stats.step_count = int(np.ceil(new_T_alpha_total / dt))
            new_anchor_stats.time_count = new_anchor_stats.step_count * dt
            new_anchor_stats.compute_k_alpha_beta()
            new_anchor_stats.last_milestone_alias_touched = ordered_keys[sequence[-1]]
            new_anchor_stats.last_milestone_time_touched = new_anchor_stats.T_alpha_total
            new_anchor_stats_list.append(new_anchor_stats)
            
        new_anchor_stats_alpha_list.append(new_anchor_stats_list)

    return new_anchor_stats_alpha_list

# TODO: move to analyzers_base
def find_nonzero_matrix_entries(
        M: np.ndarray
        ) -> list[tuple[int, int]]:
    """
    For a given matrix M find all the nonzero entries. Return the index
    tuples in a list.
    """
    assert len(M.shape) == 2, "Only 2D matrices allowed."
    nonzero_matrix_indices = []
    n = M.shape[0]
    m = M.shape[1]
    for i in range(n):
        for j in range(m):
            if i == j:
                continue
            if M[i,j] != 0:
                nonzero_matrix_indices.append((i,j))
    return nonzero_matrix_indices

def make_new_Nij_alpha(
        mmvt_Qij_alpha: np.ndarray, 
        mmvt_Ri_alpha: np.ndarray
        ) -> np.ndarray:
    """
    After a new matrix Q has been sampled, the "data" N_ij must be
    adjusted to be consistent with with newly sampled matrix Q and 
    R_i.
    """
    n_milestones = mmvt_Qij_alpha.shape[0]
    new_mmvt_Nij_alpha = np.zeros((n_milestones, n_milestones))
    for i in range(n_milestones):
        for j in range(n_milestones):
            if i == j: continue
            new_mmvt_Nij_alpha[i,j] = mmvt_Qij_alpha[i,j] * mmvt_Ri_alpha[i,0]
            
    return new_mmvt_Nij_alpha

def irreversible_stochastic_matrix_algorithm_sample(
        Q, N, R, nonzero_indices=None):
    """
    Sample an additional matrix Q from the distribution using a Monte-Carlo
    method. Use a modified version of Algorithm 1 from Noe 2008.
    """
    import random
    from scipy.stats import expon
    MAX_ITER = 1e7
    if (not Q.any()) or (not N.any()) or (not R.any()):
        return Q
    # step 0 - Generate uniform random variables: i,j members of {1,...,m}
    m = Q.shape[0]
    if nonzero_indices == None:
        i = random.choice(range(m))
        j = random.choice(range(m))
        counter = 0
        while (Q[i,j] == 0.0) or (Q[i,i] == 0.0) or (N[i,j] <= 1) \
                or (R[i,0] == 0) or (i==j):
            i = random.choice(range(m))
            j = random.choice(range(m))
            counter += 1
            if counter > MAX_ITER:
                raise Exception("Maximum iterations exceeded.")
    else:
        i, j = random.choice(nonzero_indices)
        assert i!=j, "Cannot choose same indices i and j"
    
    # step 1 - Initialize the matrix (already done)
    Qnew = copy.deepcopy(Q)
    # step 2.1 - delta, r between 0 and 1
    # delta must be sampled such that when it's subtracted from q_ij, the 
    #  result doesn't go below zero
    delta = Q[i,j] * (1.0 - expon.rvs())
    random_uniform = random.random()  
    # step 2.2 - copy Q to a new matrix - already done
    
    # step 2.3 - Nonreversible element shift
    # prior probability: P(Q)
    log_prior_probability_old = np.abs(-Qnew[i,j])
    log_prior_probability_new = np.abs(-Qnew[i,j] + delta)
    if (log_prior_probability_old == 0.0) or (log_prior_probability_new == 0.0):
        p_acc = 0.0
    else:
        log_p_Q_old = N[i,j] * np.log(np.abs(Qnew[i,j]))  - Qnew[i,j] * R[i,0]
        log_p_Q_new = N[i,j] * np.log(np.abs(Qnew[i,j] - delta)) \
            - (Qnew[i,j] - delta) * R[i,0]
        p_acc = log_p_Q_new - log_p_Q_old + log_prior_probability_new \
            - log_prior_probability_old
        
    if np.log(random_uniform) <= p_acc: 
        #log(r) can be directly compared to 
        # log-likelihood acceptance, p_acc
        Qnew[i,i] = Qnew[i,i] + delta
        Qnew[i,j] = Qnew[i,j] - delta
        
    return Qnew

def monte_carlo_milestoning_error(
        model: structures.Seekr_model,
        main_data_sample: mmvt_data_sample.Mmvt_data_sample, 
        num: int = 100, 
        stride: int | None = None, 
        skip: int | None = None,
        verbose: bool = False,
        bulk_stats: bulk_anchor_statistics.Bulk_anchor_statistics | None = None
        ) -> None:
    """
    Calculates an error estimate by sampling a distribution of rate 
    matrices assumming a Poisson (gamma) distribution with 
    parameters Nij and Ri using Markov chain Monte Carlo.
        
    Enforces detailed Balance-- using a modified version of 
    Algorithm 4 from Noe 2008 for rate matrices.-- 
    Citation: Noe, F. "Probability Distributions of molecular observables
    computed from Markov models." J. Chem. Phys. 2008, 128, No. 244103.
    Distribution is:  p(Q|N) = p(Q)p(N|Q)/p(N) = 
        p(Q) PI(q_ij**N_ij * exp(-q_ij * Ri))
    """
    MAX_SKIP = 1000
    MAX_STRIDE = 100
    Q_ij = main_data_sample.Q
    Q = copy.deepcopy(Q_ij)
    n_milestones = Q.shape[0] #get size of count matrix
    if skip is None:
        skip = 40*n_milestones
        if skip > MAX_SKIP:
            skip = MAX_SKIP
    if stride is None:
        stride = 4*n_milestones
        if stride > MAX_STRIDE:
            stride = MAX_STRIDE
    k_alpha_beta_matrix, N_alpha_beta_matrix, T_alpha_matrix, mmvt_Nij_alpha, \
        mmvt_Ri_alpha, mmvt_Qij_alpha, T \
        = main_data_sample.make_mcmc_quantities(model)
    MFPTs_list = defaultdict(list)
    MFPTs_error = defaultdict(float)
    state_to_state_energies_list = defaultdict(list)
    state_to_state_energies_error = defaultdict(float)
    k_off_list = []
    k_ons_list = defaultdict(list)
    k_ons_error = defaultdict(float)
    k_off_error = None
    p_i_list = []
    free_energy_profile_list = []
    free_energy_anchors_list = []
    data_sample_list = []
    if verbose: print("collecting ", num, " MCMC samples from ", 
                      num*(stride) + skip, " total moves")  
    new_data_sample = None
    nonzero_indices_k_alpha_beta_matrix = find_nonzero_matrix_entries(
        k_alpha_beta_matrix)
    nonzero_indices_Q_alpha_list = []
    for alpha, anchor in enumerate(model.anchors):
        if "molecular_dynamics" not in anchor.scale_types:
            continue
        nonzero_indices_Q_alpha = find_nonzero_matrix_entries(
            mmvt_Qij_alpha[alpha])
        nonzero_indices_Q_alpha_list.append(nonzero_indices_Q_alpha)
    
    for counter in range(num * (stride) + skip + 1):
        new_data_sample = mmvt_data_sample.Mmvt_data_sample(
            main_data_sample.N_alpha_beta, main_data_sample.k_alpha_beta,
            main_data_sample.N_i_j_alpha, main_data_sample.R_i_alpha,
            main_data_sample.T_alpha
        )
        k_alpha_beta_matrix = irreversible_stochastic_matrix_algorithm_sample(
            k_alpha_beta_matrix, N_alpha_beta_matrix, T_alpha_matrix,
            nonzero_indices_k_alpha_beta_matrix)
        mmvt_Qnew_list = []
        for alpha, anchor in enumerate(model.anchors):
            if "molecular_dynamics" not in anchor.scale_types:
                continue
            mmvt_Qnew_alpha = irreversible_stochastic_matrix_algorithm_sample(
                mmvt_Qij_alpha[alpha], mmvt_Nij_alpha[alpha], 
                mmvt_Ri_alpha[alpha], nonzero_indices_Q_alpha_list[alpha])
            mmvt_Qnew_list.append(mmvt_Qnew_alpha)
            
        if counter > skip and counter % stride == 0:
            if bulk_stats is not None:
                new_bulk_stats = bulk_anchor_statistics\
                    .resample_bulk_anchor_statistics(bulk_stats)
            mmvt_Nij_list = []
            for alpha, anchor in enumerate(model.anchors):
                if anchor.bulk_anchor:
                    continue
                new_mmvt_Nij_alpha = make_new_Nij_alpha(
                    mmvt_Qnew_list[alpha], mmvt_Ri_alpha[alpha])
                mmvt_Nij_list.append(new_mmvt_Nij_alpha)
                
            new_data_sample.fill_k_from_matrices(model, k_alpha_beta_matrix)
            new_data_sample.calculate_pi_alpha(model)
            new_data_sample.fill_N_R_alpha_from_matrices(
                model, mmvt_Nij_list, mmvt_Ri_alpha)
            new_data_sample.fill_out_data_quantities(model)
            #new_data_sample.compute_rate_matrix()
            new_data_sample.Q, new_data_sample.K = analyzers_base.compute_rate_matrix(
                model.num_milestones, new_data_sample.N_ij, new_data_sample.R_i)
            Q_raw, _ = analyzers_base.compute_rate_matrix(
                model.num_milestones, new_data_sample.N_ij_unmodified,
                new_data_sample.R_i_unmodified)
            new_data_sample.N_ij = main_data_sample.N_ij
            new_data_sample.R_i = main_data_sample.R_i

            #if model.using_bd():
            #    new_data_sample.parse_browndye_results(
            #        bd_sample_from_normal=True)
            new_data_sample.p_i, new_data_sample.free_energy_milestones = \
                analyzers_base.calculate_thermodynamics(
                    model.num_milestones, new_data_sample.Q,
                    model.temperature, Q_raw=Q_raw
            )
            end_states, anchor_indices_by_name \
                = mmvt_data_sample.make_end_states_dict_and_anchor_indices_by_name(model)
            new_data_sample.calculate_extra_thermodynamics(model)
            new_data_sample.milestone_to_milestone_MFPTs, \
            new_data_sample.milestone_committor_functions, \
            new_data_sample.milestone_to_milestone_probabilities = \
                analyzers_base.calculate_kinetics(
                    model.num_milestones, new_data_sample.Q, new_data_sample.K, 
                    end_states
                )
            new_data_sample.calculate_extra_kinetics(model, end_states)
            #new_data_sample.fill_out_k_off_and_k_on(anchor_indices_by_name)
            new_data_sample.fill_out_k_off(anchor_indices_by_name)
            if bulk_stats is not None:
                end_states_not_bulk = analyzers_base.get_end_states_not_bulk(model)
                new_data_sample.k_ons = analyzers_base.calculate_k_on_kinetics(
                    model.num_milestones, new_data_sample.K, end_states_not_bulk,
                    new_bulk_stats.transition_counts, 
                    new_bulk_stats.transition_probabilities,
                    new_bulk_stats.k_ons_src,
                    new_bulk_stats.outer_milestone_indices)
            data_sample_list.append(new_data_sample)
    
    return data_sample_list

def monte_carlo_data_windows(
        model: structures.Seekr_model, 
        window_index_list: list[int],
        data_window_list: list[mmvt_data_window.Mmvt_data_window],
        num_error_samples: int
    ) -> None:
    """
    Perform Monte Carlo sampling on the data windows.
    """
    for i, window_index in enumerate(window_index_list):
        data_window = data_window_list[i]
        data_sample_list = monte_carlo_milestoning_error(
            model, data_window.main_data_sample, num_error_samples)
        data_window.data_sample_list = data_sample_list
    
    return
