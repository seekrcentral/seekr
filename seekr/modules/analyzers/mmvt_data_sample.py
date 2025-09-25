"""
modules/analyzers/mmvt_data_sample.py

Given a set of MMVT Anchor Statistics, construct a data sample, which is 
a single snapshot of MMVT milestoning statistics. A data sample can also
be sampled by other means for the purposes of error analysis.
"""

import copy
import typing
from collections import defaultdict

from attrs import define, field, validators
import numpy as np
from scipy import linalg as la

import seekr.modules.structures as structures
import seekr.modules.analyzers.base as analyzers_base

HIGH_K_ALPHA_BETA_FLUX = 1e10
FLUX_MATRIX_K_EXPONENT = 1000
LOW_PROBABILITY = 1e-25

def flux_matrix_to_K(
        M: np.ndarray
        ) -> np.ndarray:
    """
    Given a flux matrix M, compute probability transition matrix K.
    """
    K = np.zeros((M.shape[0]-1, M.shape[0]-1))
    for i in range(M.shape[0]-1):
        for j in range(M.shape[0]-1):
            if i == j:
                K[i,j] = 0.0
            else:
                K[i,j] = -M[i,j] / M[i,i]
                
            assert K[i,j] >= 0.0, "Negative values for K matrix not allowed."
    
    return K
            
def make_end_states_dict_and_anchor_indices_by_name(
        model: structures.Seekr_model,
    ) -> typing.DefaultDict[str, list[int]]:
    """
    Make the end_states dictionary used in analysis for kinetics portion.
    """
    end_states = defaultdict(list)
    anchor_indices_by_name = {}
    for anchor in model.anchors:
        if anchor.end_anchor:
            for milestone in anchor.milestones:
                end_states[anchor.name].append(milestone.index)
        anchor_indices_by_name[anchor.name] = anchor.index

    return end_states, anchor_indices_by_name

@define
class Mmvt_data_sample:
    """
    A data sample for a chunk of MMVT statistics. In contrast
    to a data_window, this will be sampled from a set of statistics
    for the purposes of one data point for error analysis.
    """
    # These parameters are assigned upon instantiation
    N_alpha_beta: dict = field(
        validator=validators.instance_of(dict),)
    k_alpha_beta: dict = field(
        validator=validators.instance_of(dict),)
    N_i_j_alpha: list = field(
        factory=list,
        validator=validators.instance_of(list),)
    R_i_alpha: list = field(
        factory=list,
        validator=validators.instance_of(list),)
    T_alpha: list = field(
        factory=list,
        validator=validators.instance_of(list),)

    # These parameters are computed by subsequent methods
    N_alpha: list = field(
        factory=list,
        validator=validators.instance_of(list),)
    k_alpha: list = field(
        factory=list,
        validator=validators.instance_of(list),)
    pi_alpha: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    N_ij: typing.DefaultDict = field(
        factory=defaultdict,
        validator=validators.instance_of(defaultdict),)
    R_i: typing.DefaultDict = field(
        factory=defaultdict,
        validator=validators.instance_of(defaultdict),)
    T: float = field(
        default=0.0,
        validator=validators.optional(validators.instance_of(float)),)
    N_ij_unmodified: typing.DefaultDict = field(
        factory=defaultdict,
        validator=validators.instance_of(defaultdict),)
    R_i_unmodified: typing.DefaultDict = field(
        factory=defaultdict,
        validator=validators.instance_of(defaultdict),)
    Q: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    K: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    p_i: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_milestones: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_anchors: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    milestone_to_milestone_probabilities: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_to_milestone_MFPTs: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_committor_functions: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_probabilities: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_energies: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_MFPTs: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    
    k_off: float | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(float)),)
    k_ons: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)

    def preprocess_statistics(
            self, 
            model: structures.Seekr_model):
        """
        Fill out quantities such as N_alpha, k_alpha, etc.
        """
        self.N_alpha = []
        self.k_alpha = []
        for alpha, anchor in enumerate(model.anchors):
            if anchor.bulk_anchor:
                continue
            self.N_alpha.append(0)
            for key in self.N_i_j_alpha[alpha]:
                self.N_alpha[alpha] += self.N_i_j_alpha[alpha][key]
            
            k_alpha = 0.0
            for beta, anchor2 in enumerate(model.anchors):
                key = (alpha, beta)
                if key in self.k_alpha_beta:
                    k_alpha += self.k_alpha_beta[key]
            self.k_alpha.append(k_alpha)
            
        return

    def calculate_pi_alpha(
            self,
            model: structures.Seekr_model
            ) -> None:
        """
        Computes the equilibrium probability quantities "pi_alpha" used
        in MMVT theory. The value self.pi_alpha gets set by this 
        function.
        """
        # First, determine if there is a bulk anchor, and if so, use it as
        # the "pivot", if not, make our own pivot.
        # The "pivot" is the entry in the flux_matrix that 
        bulk_indices = []
        found_bulk = False
        for alpha, anchor in enumerate(model.anchors):
            if anchor.bulk_anchor:
                bulk_indices.append(alpha)
                found_bulk = True
            else:
                assert not found_bulk, \
                "The bulk anchors must be the last ones in the model."
        
        if len(bulk_indices) == 0:
            # Then we have a model without a bulk anchor: we need to make our
            # own pivot
            pivot_index = model.num_anchors
            flux_matrix_dimension = model.num_anchors + 1
        else:
            # If bulk anchors exist, we can use them as the pivot
            pivot_index = bulk_indices[0]
            flux_matrix_dimension = model.num_anchors-len(bulk_indices)+1
        
        self.pi_alpha = np.zeros(flux_matrix_dimension)
        flux_matrix = np.zeros((flux_matrix_dimension, flux_matrix_dimension))
        column_sum = np.zeros(flux_matrix_dimension)
        flux_matrix[pivot_index, pivot_index] = 1.0
        for alpha, anchor1 in enumerate(model.anchors):
            flux_matrix[alpha, pivot_index] = 1.0
            if anchor1.bulk_anchor:
                break
            id_alias = anchor1.alias_from_neighbor_id(pivot_index)
            flux_matrix[pivot_index, alpha] = 0.0
            dead_end_anchor = False
            if len(anchor1.milestones) == 1:
                dead_end_anchor = True
            for beta, anchor2 in enumerate(model.anchors):
                if beta == pivot_index:
                    break
                if alpha == beta:
                    pass 
                else:
                    id_alias = anchor1.alias_from_neighbor_id(
                        anchor2.index)
                    if id_alias is None:
                        flux_matrix[alpha, beta] = 0.0
                    else:
                        if dead_end_anchor:
                            # This line was supposed to work for a 1D
                            # Smoluchowski system, but with a 3D
                            # spherical system, the 2.0 needs to be 1.0.
                            #flux_matrix[alpha, beta] = 2.0 *\
                            #     self.k_alpha_beta[(alpha, beta)]
                            flux_matrix[alpha, beta] = 1.0 *\
                                 self.k_alpha_beta[(alpha, beta)]
                        else:
                            flux_matrix[alpha, beta] = \
                                self.k_alpha_beta[(alpha, beta)]
                        column_sum[alpha] += flux_matrix[alpha, beta]
                
            flux_matrix[alpha, alpha] = -column_sum[alpha]
            
        flux_matrix[pivot_index, pivot_index-1] = HIGH_K_ALPHA_BETA_FLUX
        prob_equil = np.zeros((flux_matrix_dimension,1))
        prob_equil[pivot_index] = 1.0
        pi_alpha = la.solve(flux_matrix.T, prob_equil)
        self.pi_alpha = abs(pi_alpha)
        
        # refine pi_alpha
        pi_alpha_slice = np.zeros((flux_matrix_dimension-1, 1))
        for i in range(flux_matrix_dimension-1):
            pi_alpha_slice[i,0] = -self.pi_alpha[i,0] * flux_matrix[i,i]
                    
        K = flux_matrix_to_K(flux_matrix)
        K_inf = np.linalg.matrix_power(K, FLUX_MATRIX_K_EXPONENT)
        stationary_dist = K_inf.T @ pi_alpha_slice
        for i in range(flux_matrix_dimension-1):
            self.pi_alpha[i,0] = -stationary_dist[i,0] / flux_matrix[i,i]
        for alpha, anchor1 in enumerate(model.anchors):
            if anchor1.bulk_anchor:
                break
            # Set low probabilities to anchors with no transitions observed.
            if self.k_alpha[alpha] == 0:
                self.pi_alpha[alpha,0] = LOW_PROBABILITY
        # TODO: should this be indexed by pivot_index and end_state?
        self.pi_alpha[-1,0] = 0.0
        self.pi_alpha = self.pi_alpha / np.sum(self.pi_alpha)
        return
    
    def fill_out_data_quantities(
            self,
            model: structures.Seekr_model
            ) -> None:
        """
        Compute quantities such as N_ij, R_i, and T for eventual 
        construction of rate matrix Q.
        """
        self.N_ij = defaultdict(float)
        self.R_i = defaultdict(float)
        self.T = 0.0
        sum_pi_alpha_over_T_alpha = 0.0
        for alpha, anchor in enumerate(model.anchors):
            if anchor.bulk_anchor:
                break
            if self.T_alpha[alpha] == 0.0:
                continue
            this_anchor_pi_alpha = self.pi_alpha[alpha]
            sum_pi_alpha_over_T_alpha += this_anchor_pi_alpha \
                / self.T_alpha[alpha]
                
        self.T = float(sum_pi_alpha_over_T_alpha ** -1)
        assert self.T >= 0.0, "self.T should be positive: {}".format(self.T)
        for alpha, anchor in enumerate(model.anchors):
            if anchor.bulk_anchor:
                break
            this_anchor_pi_alpha = float(self.pi_alpha[alpha])
            T_alpha = self.T_alpha[alpha]
            assert T_alpha >= 0.0, \
                "T_alpha should be positive: {}".format(T_alpha)
            if T_alpha > 0.0:
                time_fraction = self.T / T_alpha
            else:
                time_fraction = 0.0
            N_i_j_alpha = self.N_i_j_alpha[alpha]
            for key in N_i_j_alpha:
                assert time_fraction >= 0.0, \
                    "time_fraction should be positive: {}".format(time_fraction)
                assert this_anchor_pi_alpha >= 0.0, \
                    "this_anchor_pi_alpha should be positive: {}".format(
                        this_anchor_pi_alpha)
                if N_i_j_alpha[key] > 0.0:
                    self.N_ij[key] += time_fraction * \
                        this_anchor_pi_alpha * N_i_j_alpha[key]
                else:
                    # Bring the entry into existence
                    self.N_ij[key] += 0.0
            
            R_i_alpha = self.R_i_alpha[alpha]
            if len(R_i_alpha) > 0:
                for key in R_i_alpha:
                    assert time_fraction >= 0.0, \
                        "time_fraction should be positive: {}".format(
                            time_fraction)
                    assert this_anchor_pi_alpha >= 0.0, \
                        "this_anchor_pi_alpha should be positive: {}".format(
                            this_anchor_pi_alpha)
                    if R_i_alpha[key] > 0.0:
                        self.R_i[key] += time_fraction * this_anchor_pi_alpha \
                            * R_i_alpha[key]
                    else:
                        # Bring the entry into existence
                        self.R_i[key] += 0.0
            else:
                raise Exception("R_i_alpha should always be set.")

        # Now we need to try and correct the zero entries
        self.N_ij_unmodified = copy.deepcopy(self.N_ij)
        self.R_i_unmodified = copy.deepcopy(self.R_i)
        for key in self.N_ij:
            if self.N_ij[key] == 0.0:
                i = key[0]
                if self.R_i[i] == 0.0:
                    # Then this milestone has no transitions out: give
                    #  it a high N_ij
                    self.N_ij[key] = analyzers_base.HIGH_N_IJ
                else:
                    # Then this milestone has other transitions out: give it
                    #  a low N_IJ
                    self.N_ij[key] = analyzers_base.LOW_N_IJ

        for i in self.R_i:
            if self.R_i[i] == 0.0:
                self.R_i[i] = analyzers_base.DEFAULT_R_I
            
        return
    
    def calculate_extra_thermodynamics(
            self,
            model: structures.Seekr_model
            ):
        """
        Use this data sample's statistics to construct the 
        anchor free energy profile.
        """
        # TODO: use k_alpha_beta to compute the probabilities of
        # going from one anchor to another.
        # TODO: don't name it state-to-state energies - name it anchor-to-anchor energies
        anchor_to_anchor_energies = defaultdict(float)
        end_anchor_indices = []
        self.free_energy_anchors = np.zeros(model.num_anchors)
        highest_pi_alpha = max(self.pi_alpha)
        for alpha in range(model.num_anchors):
            if model.anchors[alpha].bulk_anchor:
                pi_alpha_val = self.pi_alpha[-1, 0]
            else:
                pi_alpha_val = self.pi_alpha[alpha, 0]
            free_energy = -analyzers_base.GAS_CONSTANT*model.temperature*np.log(
                pi_alpha_val / highest_pi_alpha)
            self.free_energy_anchors[alpha] = free_energy

        for i, anchor in enumerate(model.anchors):
            if anchor.end_anchor and not anchor.bulk_anchor:
                end_anchor_indices.append(i)
        
        for src_anchor_index in end_anchor_indices:
            for dest_anchor_index in end_anchor_indices:
                if src_anchor_index == dest_anchor_index:
                    continue
                
                energy_diff = self.free_energy_anchors[dest_anchor_index] \
                    - self.free_energy_anchors[src_anchor_index]
                src_anchor_name = model.anchors[src_anchor_index].name
                dest_anchor_name = model.anchors[dest_anchor_index].name
                anchor_to_anchor_energies[(src_anchor_name, dest_anchor_name)] \
                    = energy_diff

        self.anchor_to_anchor_energies = anchor_to_anchor_energies
        return

    def calculate_extra_kinetics(
            self,
            model: structures.Seekr_model,
            end_states: typing.DefaultDict[str, list[int]]
            ) -> None:
        """
        Convert milestone kinetic quantities to anchor kinetic
        quantities.
        """
        anchor_src_to_anchor_dest_MFPTs = {}
        end_state_to_end_state_probabilities_anchor = {}
        for src_state_name, src_milestone_list in end_states.items():
            marginal_prob_src = 0.0
            this_src_state_normalized_p_i = np.zeros(len(src_milestone_list))
            for i, src_milestone in enumerate(src_milestone_list):
                this_src_state_normalized_p_i[i] = self.p_i[src_milestone]
                marginal_prob_src += self.p_i[src_milestone]
            this_src_state_normalized_p_i /= marginal_prob_src
            probability_check = 0.0
            for dest_state_name, dest_milestone_list in end_states.items():
                if src_state_name == dest_state_name:
                    continue

                anchor_key = (src_state_name, dest_state_name)
                anchor_src_to_anchor_dest_MFPTs[anchor_key] = 0.0
                end_state_to_end_state_probabilities_anchor[anchor_key] = 0.0
                for i, src_milestone in enumerate(src_milestone_list):
                    marginal_prob_dest = 0.0
                    this_dest_state_normalized_prob = np.zeros(len(dest_milestone_list))
                    for j, dest_milestone in enumerate(dest_milestone_list):
                        if src_milestone == dest_milestone:
                            continue
                        prob_src_to_dest = self.milestone_to_milestone_probabilities[\
                            (src_milestone, dest_milestone)]
                        this_dest_state_normalized_prob[j] = prob_src_to_dest
                        marginal_prob_dest += prob_src_to_dest

                    this_dest_state_normalized_prob /= marginal_prob_dest

                    for j, dest_milestone in enumerate(dest_milestone_list):
                        if src_milestone == dest_milestone:
                            continue
                        prob_weight = this_src_state_normalized_p_i[i] \
                            * this_dest_state_normalized_prob[j]
                        this_src_dest_time_weighted_by_prob =  \
                            prob_weight * self.milestone_to_milestone_MFPTs[\
                            (src_milestone, dest_milestone)]
                        end_state_to_end_state_probabilities_anchor[anchor_key] \
                            += prob_weight
                        anchor_src_to_anchor_dest_MFPTs[anchor_key] \
                            += this_src_dest_time_weighted_by_prob

                probability_check += end_state_to_end_state_probabilities_anchor[anchor_key]
            
            assert np.isclose(probability_check, 1.0)
            
        self.anchor_to_anchor_probabilities \
            = end_state_to_end_state_probabilities_anchor
        self.anchor_to_anchor_MFPTs = anchor_src_to_anchor_dest_MFPTs
        return
    
    def fill_out_k_off(
            self,
            anchor_indices_by_name: typing.DefaultDict[str, int]):
        """
        Fill out the k_off and k_on rate constants for the model.
        """
        MFPT_to_bulk_non_normalized = 0.0
        marginal_prob = 0.0
        for key, MFPT in self.anchor_to_anchor_MFPTs.items():
            key1, key2 = key
            if key2 == "bulk":
                alpha = anchor_indices_by_name[key1]
                prob = self.pi_alpha[alpha]
                MFPT_to_bulk_non_normalized += MFPT * prob
                marginal_prob += prob
        if marginal_prob > 0.0:
            MFPT_to_bulk = MFPT_to_bulk_non_normalized / marginal_prob
            self.k_off = 1.0e12 / float(MFPT_to_bulk)
            
        return

    def fill_out_milestoning_quantities(
            self,
            model: structures.Seekr_model
            ) -> None:
        """
        Fill out quantities such as this data sample's Q and K matrices,
        as well as thermodynamics and kinetics quantities.
        """
        self.Q, self.K = analyzers_base.compute_rate_matrix(
            model.num_milestones, self.N_ij, self.R_i
        )
        #if model.using_bd():
        #    new_data_sample.parse_browndye_results(
        #        bd_sample_from_normal=True)
        self.p_i, self.free_energy_milestones = \
            analyzers_base.calculate_thermodynamics(
                model.num_milestones, self.Q, self.K, self.N_ij, self.R_i, 
                model.temperature
        )
        end_states, anchor_indices_by_name \
            = make_end_states_dict_and_anchor_indices_by_name(model)
        self.calculate_extra_thermodynamics(model)
        self.milestone_to_milestone_MFPTs, self.milestone_committor_functions,\
            self.milestone_to_milestone_probabilities = \
                analyzers_base.calculate_kinetics(
                    model.num_milestones, self.Q, self.K, end_states
        )
        self.calculate_extra_kinetics(model, end_states)
        self.fill_out_k_off(anchor_indices_by_name)

    def make_mcmc_quantities(
            self,
            model: structures.Seekr_model
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, 
                       list[np.ndarray], list[np.ndarray], 
                       list[np.ndarray], float]:
        """
        The MCMC matrix sampler requires inputs to be in Numpy matrix
        form. The Data_sample objects do not have their inputs in matrix
        form, so convert and return matrix objects that may be used in
        the MCMC matrix sampler.
        
        Finds k_alpha_beta_matrix, N_alpha_beta_matrix, T_alpha_matrix,
        mmvt_Nij_alpha, mmvt_Ri_alpha, mmvt_Qij_alpha, T
        """
        n_anchors = model.num_anchors
        n_milestones = model.num_milestones
        k_alpha_beta_matrix = np.zeros((n_anchors, n_anchors))
        N_alpha_beta_matrix = np.zeros((n_anchors, n_anchors))
        T_alpha_matrix = np.zeros((n_anchors, 1))
        for alpha in range(n_anchors):
            if not model.anchors[alpha].md:
                continue
            for beta in range(n_anchors):
                if alpha == beta: continue
                if (alpha,beta) in self.k_alpha_beta:
                    k_alpha_beta_matrix[alpha,beta] \
                        = self.k_alpha_beta[(alpha,beta)]
                    k_alpha_beta_matrix[alpha,alpha] \
                        -= self.k_alpha_beta[(alpha,beta)]
                    N_alpha_beta_matrix[alpha,beta] \
                        = self.N_alpha_beta[alpha,beta]
            
            T_alpha_matrix[alpha,0] = self.T_alpha[alpha]

        invT = 0.0
        mmvt_Nij_alpha = []
        mmvt_Ri_alpha = []
        mmvt_Qij_alpha = []
        for alpha in range(n_anchors):
            if not model.anchors[alpha].md:
                continue
            if self.T_alpha[alpha] > 0.0:
                invT += self.pi_alpha[alpha] / self.T_alpha[alpha]
            this_mmvt_Nij_alpha = np.zeros((n_milestones, n_milestones))
            this_mmvt_Ri_alpha = np.zeros((n_milestones, 1))
            this_mmvt_Qij_alpha = np.zeros((n_milestones, n_milestones))
            for i in range(n_milestones):
                for j in range(n_milestones):
                    key = (i,j)
                    if key in self.N_i_j_alpha[alpha]:
                        this_mmvt_Nij_alpha[i,j] = self.N_i_j_alpha[alpha][key]
                
                if i in self.R_i_alpha[alpha]:
                    this_mmvt_Ri_alpha[i,0] = self.R_i_alpha[alpha][i]
            
            for i in range(n_milestones):
                for j in range(n_milestones):
                    if i == j: continue
                    if this_mmvt_Ri_alpha[i,0] > 0.0:
                        this_mmvt_Qij_alpha[i,j] = this_mmvt_Nij_alpha[i,j] \
                            / this_mmvt_Ri_alpha[i,0]
                        this_mmvt_Qij_alpha[i,i] -= this_mmvt_Nij_alpha[i,j] \
                            / this_mmvt_Ri_alpha[i,0]
                        
            mmvt_Nij_alpha.append(this_mmvt_Nij_alpha)
            mmvt_Ri_alpha.append(this_mmvt_Ri_alpha)
            mmvt_Qij_alpha.append(this_mmvt_Qij_alpha)
            
        T = 1.0 / invT 
        
        return k_alpha_beta_matrix, N_alpha_beta_matrix, T_alpha_matrix,\
            mmvt_Nij_alpha, mmvt_Ri_alpha, mmvt_Qij_alpha, T
    
    def fill_k_from_matrices(
            self, 
            model: structures.Seekr_model,
            k_alpha_beta_matrix: np.ndarray
            ) -> None:
        """
        When the k-matrix is sampled by the MCMC procedure, the entries
        need to be converted back into a dictionary form in order to
        use the existing data_sample methods.
        """
        n_anchors = model.num_anchors
        self.k_alpha_beta = defaultdict(float)
        self.k_alpha = []
        for alpha in range(n_anchors):
            if not model.anchors[alpha].md:
                continue
            
            k_alpha = 0.0
            for beta in range(n_anchors):
                if alpha == beta: continue
                if k_alpha_beta_matrix[alpha,beta] > 0.0:
                    self.k_alpha_beta[(alpha,beta)] \
                        = k_alpha_beta_matrix[alpha,beta]
                    k_alpha += self.k_alpha_beta[(alpha,beta)]
                self.k_alpha.append(k_alpha)
            
        return
    
    def fill_N_R_alpha_from_matrices(
            self, 
            model: structures.Seekr_model,
            mmvt_Nij_alpha_list: list[np.ndarray], 
            mmvt_Ri_alpha_list: list[np.ndarray]
            ) -> None:
        """
        When the N matrix and R vector is sampled by the MCMC 
        procedure, the entries need to be converted back into a 
        dictionary form in order to use the existing data_sample 
        methods.
        """
        for alpha, anchor in enumerate(model.anchors):
            if self.N_i_j_alpha is None or self.R_i_alpha is None:
                continue
            if not anchor.md:
                break
            new_T_alpha = 0.0
            N_i_j_alpha_keys = list(self.N_i_j_alpha[alpha].keys())
            R_i_alpha_keys = list(self.R_i_alpha[alpha].keys())
            for key in N_i_j_alpha_keys:
                i, j = key
                self.N_i_j_alpha[alpha][key] = float(
                    mmvt_Nij_alpha_list[alpha][i,j])
            
            for key in R_i_alpha_keys:
                self.R_i_alpha[alpha][key] = float(
                    mmvt_Ri_alpha_list[alpha][key,0])
                new_T_alpha += self.R_i_alpha[alpha][key]
            self.T_alpha[alpha] = new_T_alpha
        
        return