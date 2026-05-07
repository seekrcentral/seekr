"""
modules/analyzers/mmvt_data_window.py

A data window represents a "window" in time, which may contain the data 
accumulated over all simulation time previously. The data window contains 
at least one data sample, and may contain many data samples for error
analysis purposes. The data window is most useful for convergence analysis.
"""

from collections import defaultdict

from attrs import define, field, validators
import numpy as np

import seekr.modules.analyzers.base as analyzers_base
import seekr.modules.structures as structures
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics
import seekr.modules.analyzers.mmvt_data_sample as mmvt_data_sample

@define
class Mmvt_data_window:
    """
    A data window for a chunk of MMVT statistics. In contrast
    to a data_sample, this will represent a single "window" in a 
    convergence analysis, (or the entire set of available data).
    A large set of data_samples will be used to fill out the 
    errors of this data window.
    """
    main_data_sample: mmvt_data_sample.Mmvt_data_sample | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(
            mmvt_data_sample.Mmvt_data_sample)),)
    data_sample_list: list[mmvt_data_sample.Mmvt_data_sample] = field(
        factory=list,
        validator=validators.instance_of(list),)
    time: float | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(float)),)
    pi_alpha: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    pi_alpha_err: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    p_i: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    p_i_err: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_milestones: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_milestones_err: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_anchors: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    free_energy_anchors_err: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),)
    milestone_to_milestone_probabilities: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_to_milestone_probabilities_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_to_milestone_MFPTs: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_to_milestone_MFPTs_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_committor_functions: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    milestone_committor_functions_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_probabilities: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_probabilities_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_energies: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_energies_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_MFPTs: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    anchor_to_anchor_MFPTs_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    k_off: float | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(float)),)
    k_off_err: float | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(float)),)
    k_ons: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    k_ons_err: dict = field(
        factory=dict,
        validator=validators.instance_of(dict),)
    
    def fill_out_attributes_from_main_data_sample(self) -> None:
        """
        Fill out this class's attributes from self.main_data_sample
        """
        if self.main_data_sample is not None:
            self.data_sample_list = [self.main_data_sample]
            self.pi_alpha = self.main_data_sample.pi_alpha
            self.p_i = self.main_data_sample.p_i
            self.free_energy_milestones \
                = self.main_data_sample.free_energy_milestones
            self.free_energy_anchors \
                = self.main_data_sample.free_energy_anchors
            self.milestone_to_milestone_probabilities \
                = self.main_data_sample.milestone_to_milestone_probabilities
            self.milestone_to_milestone_MFPTs \
                = self.main_data_sample.milestone_to_milestone_MFPTs
            self.milestone_committor_functions \
                = self.main_data_sample.milestone_committor_functions
            self.anchor_to_anchor_probabilities \
                = self.main_data_sample.anchor_to_anchor_probabilities
            self.anchor_to_anchor_energies \
                = self.main_data_sample.anchor_to_anchor_energies
            self.anchor_to_anchor_MFPTs \
                = self.main_data_sample.anchor_to_anchor_MFPTs
            self.k_off = self.main_data_sample.k_off
            self.k_ons = self.main_data_sample.k_ons

        return

    def fill_out_errors_from_data_samples(self) -> None:
        if self.pi_alpha is None:
            for key in self.milestone_to_milestone_probabilities:
                self.milestone_to_milestone_probabilities_err[key] = None

            for key in self.milestone_committor_functions:
                self.milestone_committor_functions_err[key] = None

            for key in self.milestone_to_milestone_MFPTs:
                self.milestone_to_milestone_MFPTs_err[key] = None

            for key in self.anchor_to_anchor_probabilities:
                self.anchor_to_anchor_probabilities_err[key] = None

            for key in self.anchor_to_anchor_energies:
                self.anchor_to_anchor_energies_err[key] = None

            for key in self.anchor_to_anchor_MFPTs:
                self.anchor_to_anchor_MFPTs_err[key] = None
            
            for key in self.k_ons:
                self.k_ons_err[key] = None
                
            return
        
        if len(self.data_sample_list) > 0:
            pi_alpha_list = []
            p_i_list = []
            free_energy_milestones_list = []
            free_energy_anchors_list = []
            milestone_to_milestone_probabilities_dict_list = defaultdict(list)
            milestone_committor_functions_dict_list = defaultdict(list)
            milestone_to_milestone_MFPTs_dict_list = defaultdict(list)
            anchor_to_anchor_probabilities_dict_list = defaultdict(list)
            anchor_to_anchor_energies_dict_list = defaultdict(list)
            anchor_to_anchor_MFPTs_dict_list = defaultdict(list)
            k_off_list = []
            k_ons_list = defaultdict(list)
            self.pi_alpha_err = np.zeros(self.pi_alpha.shape)
            self.p_i_err = np.zeros(self.p_i.shape)
            self.free_energy_milestones_err = np.zeros(self.free_energy_milestones.shape)
            self.free_energy_anchors_err = np.zeros(self.free_energy_anchors.shape)
            self.milestone_to_milestone_probabilities_err = {}
            self.milestone_committor_functions_err = {}
            self.milestone_to_milestone_MFPTs_err = {}
            self.anchor_to_anchor_probabilities_err = {}
            self.anchor_to_anchor_energies_err = {}
            self.anchor_to_anchor_MFPTs_err = {}
            self.k_off_err = 0.0
            self.k_ons_err = {}
            for counter, data_sample in enumerate(self.data_sample_list):
                # Error quantities
                if data_sample.pi_alpha is None:
                    continue
                pi_alpha_list.append(data_sample.pi_alpha)
                if data_sample.p_i is None:
                    continue
                p_i_list.append(data_sample.p_i)
                free_energy_milestones_list.append(data_sample.free_energy_milestones)
                free_energy_anchors_list.append(data_sample.free_energy_anchors)
                for key, value in data_sample.milestone_to_milestone_probabilities.items():
                    milestone_to_milestone_probabilities_dict_list[key]\
                        .append(value)
                for key, value in data_sample.milestone_committor_functions.items():
                    milestone_committor_functions_dict_list[key]\
                        .append(value)
                for key, value in data_sample.milestone_to_milestone_MFPTs.items():
                    milestone_to_milestone_MFPTs_dict_list[key]\
                        .append(value)
                for key, value in data_sample.anchor_to_anchor_probabilities.items():
                    anchor_to_anchor_probabilities_dict_list[key]\
                        .append(value)
                for key, value in data_sample.anchor_to_anchor_energies.items():
                    anchor_to_anchor_energies_dict_list[key]\
                        .append(value)
                for key, value in data_sample.anchor_to_anchor_MFPTs.items():
                    anchor_to_anchor_MFPTs_dict_list[key]\
                        .append(value)
                if data_sample.k_off is not None:
                    k_off_list.append(data_sample.k_off)
                for key, value in data_sample.k_ons.items():
                    if value is not None:
                        k_ons_list[key].append(value)

            for i in range(self.pi_alpha.shape[0]):
                pi_alpha_val_list = []
                for j in range(len(pi_alpha_list)):
                    pi_alpha_val_list.append(pi_alpha_list[j][i])
                self.pi_alpha_err[i] = np.std(pi_alpha_val_list)

            for i in range(self.p_i.shape[0]):
                p_i_val_list = []
                for j in range(len(p_i_list)):
                    p_i_val_list.append(p_i_list[j][i])
                self.p_i_err[i] = np.std(p_i_val_list)

            for i in range(self.free_energy_milestones.shape[0]):
                free_energy_milestones_val_list = []
                for j in range(len(free_energy_milestones_list)):
                    free_energy_milestones_val_list.append(free_energy_milestones_list[j][i])
                self.free_energy_milestones_err[i] = np.std(free_energy_milestones_val_list)

            for i in range(self.free_energy_anchors.shape[0]):
                free_energy_anchors_val_list = []
                for j in range(len(free_energy_anchors_list)):
                    free_energy_anchors_val_list.append(free_energy_anchors_list[j][i])
                self.free_energy_anchors_err[i] = np.std(free_energy_anchors_val_list)

            for key in self.milestone_to_milestone_probabilities:
                error = np.std(milestone_to_milestone_probabilities_dict_list[key])
                self.milestone_to_milestone_probabilities_err[key] = error

            for key in self.milestone_committor_functions:
                error = np.std(milestone_committor_functions_dict_list[key])
                self.milestone_committor_functions_err[key] = error

            for key in self.milestone_to_milestone_MFPTs:
                error = np.std(milestone_to_milestone_MFPTs_dict_list[key])
                self.milestone_to_milestone_MFPTs_err[key] = error

            for key in self.anchor_to_anchor_probabilities:
                error = np.std(anchor_to_anchor_probabilities_dict_list[key])
                self.anchor_to_anchor_probabilities_err[key] = error

            for key in self.anchor_to_anchor_energies:
                error = np.std(anchor_to_anchor_energies_dict_list[key])
                self.anchor_to_anchor_energies_err[key] = error
            
            for key in self.anchor_to_anchor_MFPTs:
                error = np.std(anchor_to_anchor_MFPTs_dict_list[key])
                self.anchor_to_anchor_MFPTs_err[key] = error

            if len(k_off_list) > 0:
                self.k_off_err = np.std(k_off_list)
            
            for key in self.k_ons:
                if len(k_ons_list[key]) > 0:
                    error = np.std(k_ons_list[key])
                    self.k_ons_err[key] = error

        return

def check_extraction(
        model: structures.Seekr_model,
        anchor_stats_alpha_list: list[list[mmvt_anchor_statistics.Mmvt_anchor_statistics]],
        window_index: int = -1,
        abort_on_error: bool = True
        ) -> bool:
    """
    Check the extracted data to ensure that sufficient transitions
    have accrued to fill out the MMVT model. If abort_on_error is False,
    the result returns False if any errors are found, but does not raise
    an error.
    """
    error_warning_base = "Anchor(s) {0} are missing some "\
        "statistics. Consider running simulations of anchor(s) {0} "\
        "for longer time scales or readjust anchor locations to "\
        "make transitions more frequent. You may skip this check "\
        "with the --skip_checks (-s) option.\n"
    path_not_connected_string = "Insufficient statistics were found "\
        "to create a path between end {0} \"{1}\" and \"{2}\".\n"
    error_warning_string = ""
    end_anchor_set = set()
    end_milestone_sets = []
    anchors_missing_alpha_beta_stats = set()
    anchors_missing_i_j_statistics = set()
    connections_anchor_to_anchor = defaultdict(list)
    connections_milestone_to_milestone = defaultdict(list)
    # A dictionary whose keys are milestone pairs and values are anchor indices
    for alpha, anchor in enumerate(model.anchors):
        end_milestone_set = set()
        if anchor.bulk_anchor:
            for milestone in anchor.milestones:
                connections_anchor_to_anchor[anchor.index].append(
                    milestone.neighbor_anchor_index)
                end_milestone_set.add(milestone.index)
                for milestone2 in anchor.milestones:
                    if milestone.alias_index == milestone2.alias_index:
                        continue
                    # NOTE: This will cause the program to assume that BD connections
                    #  between milestones will be sufficiently sampled. While this
                    #  may not necessarily be true, BD checks are handled elsewhere.
                    # TODO: BD sampling checks
                    connections_milestone_to_milestone[milestone.index].append(
                        milestone2.index)
            bulk_anchor_index = anchor.index
            end_anchor_set.add(bulk_anchor_index)
            continue

        if anchor.end_anchor:
            end_anchor_set.add(anchor.index)
            for milestone in anchor.milestones:
                end_milestone_set.add(milestone.index)

        if len(end_milestone_set) > 0:
            end_milestone_sets.append(end_milestone_set)

        if anchor_stats_alpha_list[alpha] is not None:
            existing_alias_alpha_beta = []
            existing_alias_i_j = []
            if len(anchor_stats_alpha_list[alpha]) == 0:
                anchors_missing_alpha_beta_stats.add(anchor.index)
                continue
            if window_index >= len(anchor_stats_alpha_list[alpha]):
                window_index_new = len(anchor_stats_alpha_list[alpha]) - 1
            else:
                window_index_new = window_index
            anchor_stats_window: mmvt_anchor_statistics.Mmvt_anchor_statistics \
                = anchor_stats_alpha_list[alpha][window_index_new]
            for key in anchor_stats_window.k_alpha_beta:
                if anchor_stats_window.k_alpha_beta[key] > 0.0:
                    existing_alias_alpha_beta.append(key)
                assert anchor_stats_window.k_alpha_beta[key] >= 0.0, \
                    "negative k_alpha_beta values not allowed."
            
            for key in anchor_stats_window.N_i_j_alpha:
                if anchor_stats_window.N_i_j_alpha[key] > 0:
                    existing_alias_i_j.append(key)

            for milestone in anchor.milestones:
                if milestone.alias_index in existing_alias_alpha_beta:
                    connections_anchor_to_anchor[anchor.index].append(
                        milestone.neighbor_anchor_index)
                else:
                    anchors_missing_alpha_beta_stats.add(anchor.index)

                for milestone2 in anchor.milestones:
                    if milestone.alias_index == milestone2.alias_index:
                        continue
                    if (milestone.alias_index, milestone2.alias_index) \
                            in existing_alias_i_j:
                        connections_milestone_to_milestone[milestone.index].append(
                            milestone2.index)
                    else:
                        anchors_missing_i_j_statistics.add(anchor.index)
                        
    missing_anchor_path = False
    for end_anchor_index in end_anchor_set:
        for end_anchor_index2 in end_anchor_set:
            if end_anchor_index == end_anchor_index2:
                continue
            states_connected = analyzers_base.check_graph_connected(
                connections_anchor_to_anchor, end_anchor_index, end_anchor_index2)
            if not states_connected:
                error_str = path_not_connected_string.format(
                    "anchors",
                    model.anchors[end_anchor_index].name,
                    model.anchors[end_anchor_index2].name)
                error_warning_string += error_str
                missing_anchor_path = True
    missing_milestone_path = False
    for i, end_milestone_set in enumerate(end_milestone_sets):
        for j, end_milestone_set2 in enumerate(end_milestone_sets):
            if i == j:
                continue
            some_states_connected = False
            for end_milestone_index in end_milestone_set:
                for end_milestone_index2 in end_milestone_set2:
                    states_connected = analyzers_base.check_graph_connected(
                        connections_milestone_to_milestone, end_milestone_index, 
                        end_milestone_index2)
                    if states_connected:
                        some_states_connected = True
            if not some_states_connected:
                error_str = path_not_connected_string.format(
                    "milestones", end_milestone_set, end_milestone_set2)
                error_warning_string += error_str
                missing_milestone_path = True

    if missing_anchor_path or missing_milestone_path:
        anchors_missing_statistics = anchors_missing_alpha_beta_stats \
            | anchors_missing_i_j_statistics
        error_warning_string += error_warning_base.format(
            anchors_missing_statistics)
    
    checks_passed = True
    if error_warning_string != "":
        if abort_on_error:
            raise analyzers_base.MissingStatisticsError(
                error_warning_string)
        else:
            checks_passed = False

    return checks_passed

def assign_data_windows_sample(
        model: structures.Seekr_model, 
        anchor_stats_alpha_list: list[list[mmvt_anchor_statistics.Mmvt_anchor_statistics]],
        bulk_stats: bulk_anchor_statistics.Bulk_anchor_statistics | None,
        window_index_list: list[int],
        data_window_list: list[Mmvt_data_window],
        assign_to_main: bool):
    """
    For each of the statistics windows, create the corresponding data
    sample.
    """
    end_states_not_bulk = analyzers_base.get_end_states_not_bulk(model)
    for i, window_index in enumerate(window_index_list):
        N_alpha_beta = defaultdict(int)
        k_alpha_beta = defaultdict(float)
        N_i_j_alpha = []
        R_i_alpha_total = []
        T_alpha_total = []
        window_time = 0.0
        for alpha, anchor1 in enumerate(model.anchors):
            if anchor1.bulk_anchor:
                continue
            #if window_index >= len(anchor_stats_alpha_list[alpha]):
            #    window_index = len(anchor_stats_alpha_list[alpha]) - 1
            if i >= len(anchor_stats_alpha_list[alpha]):
                i_modified = len(anchor_stats_alpha_list[alpha]) - 1
            else:
                i_modified = i
            anchor_stats_window: mmvt_anchor_statistics.Mmvt_anchor_statistics \
                = anchor_stats_alpha_list[alpha][i_modified]
            if anchor_stats_window.time_count > window_time:
                window_time = anchor_stats_window.time_count
            anchor_N_alpha_beta = anchor_stats_window.N_alpha_beta
            anchor_k_alpha_beta = anchor_stats_window.k_alpha_beta
            for beta, anchor2 in enumerate(model.anchors):
                if anchor2.bulk_anchor:
                    continue
                if alpha == beta:
                    continue
                id_alias = anchor1.alias_from_neighbor_id(anchor2.index)
                if id_alias is None:
                    continue
                if len(anchor_N_alpha_beta) == 0:
                    # Then no transitions were observed in this anchor
                    default_flux = analyzers_base.FAST_FLUX
                else:
                    default_flux = analyzers_base.LOW_FLUX
                
                if id_alias in anchor_N_alpha_beta:
                    N_alpha_beta[(alpha, beta)] = anchor_N_alpha_beta[id_alias]
                    k_alpha_beta[(alpha, beta)] = anchor_k_alpha_beta[id_alias]
                else:
                    N_alpha_beta[(alpha, beta)] = 0
                    k_alpha_beta[(alpha, beta)] = default_flux #0.0
                
            anchor_N_i_j_alpha = anchor_stats_window.N_i_j_alpha
            N_i_j_alpha_element = defaultdict(int)
            R_i_alpha_element = defaultdict(float)
            
            # Fill out empty milestone statistics
            for milestone1 in anchor1.milestones:
                R_i_alpha_element[milestone1.index] = 0.0
                for milestone2 in anchor1.milestones:
                    if milestone1.index == milestone2.index:
                        continue
                    new_key = (milestone1.index, milestone2.index)
                    N_i_j_alpha_element[new_key] = 0
            
            for key in anchor_N_i_j_alpha:
                (alias_id_i, alias_id_j) = key
                id_i = anchor1.id_from_alias(alias_id_i)
                id_j = anchor1.id_from_alias(alias_id_j)
                new_key = (id_i, id_j)
                N_i_j_alpha_element[new_key] = anchor_N_i_j_alpha[key]
            N_i_j_alpha.append(N_i_j_alpha_element)
            
            anchor_R_i_alpha = anchor_stats_window.R_i_alpha_total
            
            for key in anchor_R_i_alpha:
                alias_id_i = key
                id_i = anchor1.id_from_alias(alias_id_i)
                assert id_i is not None, \
                    "Anchor {} missing milestone alias {}."\
                        .format(anchor1.index, alias_id_i)
                R_i_alpha_element[id_i] = anchor_R_i_alpha[key]
            
            R_i_alpha_total.append(R_i_alpha_element)
            anchor_T_alpha = anchor_stats_window.T_alpha_total
            T_alpha_total.append(anchor_T_alpha)
        abort_on_check_error = False
        if assign_to_main and i == len(window_index_list) - 1:
            abort_on_check_error = True

        check_passed = check_extraction(
            model, anchor_stats_alpha_list, window_index=i,
            abort_on_error=abort_on_check_error)
        
        data_sample = mmvt_data_sample.Mmvt_data_sample(
            N_alpha_beta, k_alpha_beta, N_i_j_alpha, 
            R_i_alpha_total, T_alpha_total)
        
        if i == len(window_index_list) - 1:
            print("data_sample.k_alpha_beta:", data_sample.k_alpha_beta)
            print("data_sample.N_alpha_beta:", data_sample.N_alpha_beta)
            exit()
        
        """
        if bulk_stats is not None:
            data_sample.k_ons_src = bulk_stats.k_ons_src
            data_sample.k_on_errors_src = bulk_stats.k_on_errors_src
            data_sample.transition_counts = bulk_stats.transition_counts
            data_sample.transition_probabilities = bulk_stats.transition_probabilities
            data_sample.outer_milestone_indices = bulk_stats.outer_milestone_indices
        """

        if check_passed:
            data_sample.preprocess_statistics(model)
            data_sample.calculate_pi_alpha(model)
            data_sample.fill_out_data_quantities(model)
            data_sample.fill_out_milestoning_quantities(model)
            if bulk_stats is not None:
                data_sample.k_ons = analyzers_base.calculate_k_on_kinetics(
                    model.num_milestones, data_sample.K, end_states_not_bulk,
                    bulk_stats.transition_counts, 
                    bulk_stats.transition_probabilities,
                    bulk_stats.k_ons_src,
                    bulk_stats.outer_milestone_indices)
        
        data_window = data_window_list[i]
        if assign_to_main:
            data_window.time = window_time
            data_window.main_data_sample = data_sample
            data_window.fill_out_attributes_from_main_data_sample()
        else:
            data_window.data_sample_list.append(data_sample)

    return
