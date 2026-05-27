from __future__ import annotations
"""
modules/analyzers/mmvt_anchor_statistics.py

Contain all the transition statistics for an anchor within
an MMVT calculation.
"""

import os
from collections import defaultdict

from attrs import define, field, validators
import numpy as np

import seekr.modules.structures as structures

@define
class Mmvt_anchor_statistics:
    """
    Contain all the transition statistics for an anchor within
    an MMVT calculation.
    """
    alpha: int = field(validator=validators.instance_of(int))
    step_count: int = field(
        default=0,
        validator=validators.instance_of(int),)
    time_count: float = field(
        default=0.0,
        validator=validators.instance_of(float),)
    first_encounter_time: float = field(
        default=0.0,
        validator=validators.instance_of(float),)
    last_milestone_alias_touched: int = field(
        default=0,
        validator=validators.instance_of(int),)
    last_milestone_time_touched: float = field(
        default=0.0,
        validator=validators.instance_of(float),)
    N_alpha_beta: dict | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(dict)),)
    k_alpha_beta: dict | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(dict)),)
    N_i_j_alpha: dict | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(dict)),)
    R_i_alpha_total: dict | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(dict)),)
    T_alpha_total: float | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(float)),)
    
    def compute_k_alpha_beta(self) -> None:
        """
        Compute the k_alpha_beta values from N_alpha_beta.
        """
        if self.N_alpha_beta is not None:
            self.k_alpha_beta = defaultdict(float)
            for key in self.N_alpha_beta:
                if self.T_alpha_total == 0:
                    self.k_alpha_beta[key] = 0.0
                else:
                    self.k_alpha_beta[key] = self.N_alpha_beta[key] / self.T_alpha_total
        else:
            self.k_alpha_beta = None
        return

    def add_mmvt_anchor_statistics(
            self, 
            other: "Mmvt_anchor_statistics") -> None:
        """
        Add the statistics from another Mmvt_anchor_statistics object.
        """
        self.step_count += other.step_count
        self.time_count += other.time_count
        for key in other.N_alpha_beta:
            self.N_alpha_beta[key] += other.N_alpha_beta[key]
        for key in other.N_i_j_alpha:
            self.N_i_j_alpha[key] += other.N_i_j_alpha[key]
        for key in other.R_i_alpha_total:
            self.R_i_alpha_total[key] += other.R_i_alpha_total[key]
        self.T_alpha_total += other.T_alpha_total
        return

    def get_statistics_message(
            self,
            model: structures.Seekr_model,
            alpha: int,
            use_json: bool = True
            ) -> str | dict:
        """
        
        """
        message = None
        counts_anchor_to_anchor = {}
        counts_milestone_to_milestone = {}
        times_milestone_to_milestone = {}
        anchor = model.anchors[alpha]
        existing_alias_alpha_beta = []
        existing_alias_i_j = []
        if self.k_alpha_beta is None or self.N_i_j_alpha is None:
            if use_json:
                return {}
            else:
                return ""
        
        for key in self.k_alpha_beta:
            if self.k_alpha_beta[key] > 0.0:
                existing_alias_alpha_beta.append(key)

        for key in self.N_i_j_alpha:
            if self.N_i_j_alpha[key] > 0:
                existing_alias_i_j.append(key)

        for milestone in anchor.milestones:
            key = (anchor.index, milestone.neighbor_anchor_index)
            key_str = str(key)
            if milestone.alias_index in existing_alias_alpha_beta:
                counts_anchor_to_anchor[key_str] = self.N_alpha_beta[milestone.alias_index]
            else:
                counts_anchor_to_anchor[key_str] = 0
            times_milestone_to_milestone[milestone.index] = self.R_i_alpha_total.get(milestone.alias_index, 0.0)
            for milestone2 in anchor.milestones:
                if milestone.alias_index == milestone2.alias_index:
                    continue
                key2 = (milestone.index, milestone2.index)
                key2_str = str(key2)
                if (milestone.alias_index, milestone2.alias_index) \
                        in existing_alias_i_j:
                    counts_milestone_to_milestone[key2_str] = \
                        self.N_i_j_alpha[(milestone.alias_index, milestone2.alias_index)]
                else:
                    counts_milestone_to_milestone[key2_str] = 0

        if use_json:
            # Use JSON-style message
            message = {
                "anchor_transition_counts": counts_anchor_to_anchor,
                "milestone_transition_counts": counts_milestone_to_milestone,
            }
        else:
            message = ""
            message += "    Anchor transition counts:\n"
            for key in counts_anchor_to_anchor:
                message += f"        {key}: {counts_anchor_to_anchor[key]}\n"
            if len(counts_anchor_to_anchor) == 0:
                message += "        None\n"
            message += "    Milestone transition counts:\n"
            for key in counts_milestone_to_milestone:
                message += f"        {key}: {counts_milestone_to_milestone[key]}\n"
            if len(counts_milestone_to_milestone) == 0:
                message += "        None\n"
            message += "    Milestone transition times (ps):\n"
            for key in times_milestone_to_milestone:
                message += f"        {key}: {times_milestone_to_milestone[key]:<.3f}\n"
            if len(times_milestone_to_milestone) == 0:
                message += "        None\n"
            message += f"    Total time (ps): {self.T_alpha_total:<.3f}\n"

        return message

# TODO: this is just read_MD_output_file - there needs to be one for
# multiple BD states! (Need multiple BD states to identify committor
# function.)
def read_output_file(
        output_filename: str,
        alpha: int = 0,
        min_step: float = 0,
        max_step: float = float('inf')
        ) -> list["Mmvt_anchor_statistics"]:
    """
    Read the output file for a single anchor and extract statistics.
    
    Parameters
    ----------
    output_filename : str
        The path to the output file to read.
    
    Returns
    -------
    list[Mmvt_anchor_statistics]
        A list of Mmvt_anchor_statistics objects containing the extracted data.
    """
    anchor_statistics_windows_list = []
    assert os.path.exists(output_filename)
    anchor_statistics = Mmvt_anchor_statistics(alpha=alpha)
    anchor_statistics.N_i_j_alpha = defaultdict(int)
    anchor_statistics.R_i_alpha_total = defaultdict(float)
    anchor_statistics.N_alpha_beta = defaultdict(int)
    anchor_statistics.T_alpha_total = 0.0
    time_between_statistics = None
    last_time = None
    with open(output_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                if anchor_statistics.step_count > 0:
                    anchor_statistics.compute_k_alpha_beta()
                    anchor_statistics_windows_list.append(anchor_statistics)
                anchor_statistics = Mmvt_anchor_statistics(alpha=alpha)
                anchor_statistics.N_i_j_alpha = defaultdict(int)
                anchor_statistics.R_i_alpha_total = defaultdict(float)
                anchor_statistics.N_alpha_beta = defaultdict(int)
                anchor_statistics.T_alpha_total = 0.0
            elif line.startswith("step_count"):
                step_count = int(line.split()[1])
                if step_count < min_step or step_count > max_step:
                    break
                anchor_statistics.step_count = step_count
            elif line.startswith("time_count"):
                time_count = float(line.split()[1])
                if last_time is None:
                    last_time = time_count
                else:
                    time_between_statistics = time_count - last_time
                    if not np.isclose(time_between_statistics, (time_count - last_time)):
                        print("time_between_statistics:", time_between_statistics)
                        print("time_count:", time_count)
                        print("last_time:", last_time)
                        print("time_count - last_time:", time_count - last_time)
                        raise Exception("Time between statistics does not match")
                    assert time_count > last_time, "Time count should be non-decreasing." \
                    + f" Filename: {output_filename}, time_count: {time_count}, last_time: {last_time}"
                anchor_statistics.time_count = time_count
                last_time = time_count
            elif line.startswith("first_encounter_time"):
                first_encounter_time = float(line.split()[1])
                anchor_statistics.first_encounter_time = first_encounter_time
            elif line.startswith("last_milestone_alias_touched"):
                last_milestone_alias_touched = int(line.split()[1])
                anchor_statistics.last_milestone_alias_touched = last_milestone_alias_touched
            elif line.startswith("last_milestone_time_touched"):
                last_milestone_time_touched = float(line.split()[1])
                anchor_statistics.last_milestone_time_touched = last_milestone_time_touched
                # See NOTE below for milestone_#_time
                anchor_statistics.T_alpha_total = last_milestone_time_touched \
                    - anchor_statistics.first_encounter_time
            elif line.startswith("milestone"):
                line_split_by_spaces = line.split()
                key_string = line_split_by_spaces[0].strip(":")
                value_string = line_split_by_spaces[1]
                key_string_parts = key_string.split("_")
                if key_string_parts[-1] == "count":
                    src_index = int(key_string_parts[1])
                    dest_index = int(key_string_parts[2])
                    anchor_statistics.N_alpha_beta[dest_index] += int(value_string)
                    if src_index != dest_index:
                        anchor_statistics.N_i_j_alpha[(src_index, dest_index)] += int(value_string)

                elif key_string_parts[-1] == "time":
                    src_index = int(key_string_parts[1])
                    anchor_statistics.R_i_alpha_total[src_index] += float(value_string)
                    # NOTE: T_alpha_total should actually not be assigned here,
                    # but rather the difference between first_encounter_time and 
                    # last_milestone_time_touched. This is because, the counter for 
                    # milestone_#_time might only be updated when a different milestone
                    # is touched than where originated - not the right count for k_alpha_beta...
                    #anchor_statistics.T_alpha_total += float(value_string)
                    
                else:
                    raise ValueError(
                        f"Unexpected key in output file: {key_string}")
    return anchor_statistics_windows_list, time_between_statistics

def read_output_file_list(
        output_file_list: list[str], 
        anchor: structures.Anchor, 
        min_step: float, 
        max_step: float) -> None:
    """
    Read the output files for this anchor and extract statistics.
    """
    # NOTE: in this function, we want to generate a list of Mmvt_anchor_statistics
    # going through time, so that we can generate convergence plots, as well
    # as use them for some error analysis. However, the different swarm statistics
    # for a given restart can be combined into a single Mmvt_anchor_statistics
    # object. However, we will need to concatenate the lists of Mmvt_anchor_statistics
    # objects from each of the restarts. The file sorter currently places all the 
    # swarm files together, so that's good news.
    # TODO: test to make sure that more than one swarm gets added to statistics 
    # properly, and that convergence will be correct across multiple restarts - 
    # which means that there aren't any duplicate statistics.
    anchor_stats_alpha_list = []
    on_restart = 0
    mmvt_statistics_list_this_restart = None
    time_between_statistics = None
    #assert len(output_file_list) > 0, f"No output files found for anchor: {anchor.index}."
    if len(output_file_list) == 0:
        return anchor_stats_alpha_list, time_between_statistics
    
    for output_filename in output_file_list:
        this_swarm_id = int(output_filename.split("_")[-1].split(".")[0])
        if this_swarm_id == 0:
            if mmvt_statistics_list_this_restart is not None:
                anchor_stats_alpha_list.extend(mmvt_statistics_list_this_restart)
            # If we are on a new swarm, we need to increment the restart counter.
            on_restart += 1
            # generate a new mmvt_statistics list
            mmvt_statistics_list_this_restart, time_between_statistics\
                 = read_output_file(output_filename, alpha=anchor.index, 
                                    min_step=min_step, max_step=max_step)
        else:
            # Then it's a new swarm, and we can add the statistics
            new_swarm_mmvt_statistics_list, new_swarm_time_between_statistics \
                = read_output_file(output_filename, alpha=anchor.index, 
                                   min_step=min_step, max_step=max_step)
            assert np.isclose(new_swarm_time_between_statistics, time_between_statistics), \
                "Time between statistics does not match"
            for mmvt_statistics, new_swarm_mmvt_statistics in zip(
                    mmvt_statistics_list_this_restart, new_swarm_mmvt_statistics_list):
                # We can combine the statistics from the two swarms.
                mmvt_statistics.add_mmvt_anchor_statistics(new_swarm_mmvt_statistics)

    anchor_stats_alpha_list.extend(mmvt_statistics_list_this_restart)
    #if anchor.index == 2:
    #    exit()
    return anchor_stats_alpha_list, time_between_statistics

def subtract_two_anchor_statistics(
        anchor_statistics_1: "Mmvt_anchor_statistics",
        anchor_statistics_2: "Mmvt_anchor_statistics"
        ) -> "Mmvt_anchor_statistics":
    """
    Perform statistics subtraction anchor_statistics_2 - anchor_statistics_1.
    """
    difference_anchor_statistics = Mmvt_anchor_statistics(anchor_statistics_1.alpha)
    assert anchor_statistics_1.alpha == anchor_statistics_2.alpha
    difference_anchor_statistics.step_count \
        = anchor_statistics_2.step_count - anchor_statistics_1.step_count
    difference_anchor_statistics.time_count \
        = anchor_statistics_2.time_count - anchor_statistics_1.time_count
    difference_anchor_statistics.first_encounter_time \
        = anchor_statistics_1.first_encounter_time
    difference_anchor_statistics.last_milestone_alias_touched \
        = anchor_statistics_2.last_milestone_alias_touched
    difference_anchor_statistics.last_milestone_time_touched \
        = anchor_statistics_2.last_milestone_time_touched
    difference_anchor_statistics.N_alpha_beta = defaultdict(int)
    for key in anchor_statistics_2.N_alpha_beta:
        if key in anchor_statistics_1.N_alpha_beta:
            difference_anchor_statistics.N_alpha_beta[key] \
                = anchor_statistics_2.N_alpha_beta[key] \
                    - anchor_statistics_1.N_alpha_beta[key]
        else:
            difference_anchor_statistics.N_alpha_beta[key] \
                = anchor_statistics_2.N_alpha_beta[key]
    
    T_alpha_total = anchor_statistics_2.T_alpha_total \
        - anchor_statistics_1.T_alpha_total
    difference_anchor_statistics.T_alpha_total = T_alpha_total
    difference_anchor_statistics.k_alpha_beta = defaultdict(float)
    for key in difference_anchor_statistics.N_alpha_beta:
        if T_alpha_total > 0:
            difference_anchor_statistics.k_alpha_beta[key] \
                = difference_anchor_statistics.N_alpha_beta[key] \
                / T_alpha_total
        else:
            difference_anchor_statistics.k_alpha_beta[key] = 0.0

    difference_anchor_statistics.N_i_j_alpha = defaultdict(int)
    for key in anchor_statistics_2.N_i_j_alpha:
        if key in anchor_statistics_1.N_i_j_alpha:
            difference_anchor_statistics.N_i_j_alpha[key] \
                = anchor_statistics_2.N_i_j_alpha[key] \
                    - anchor_statistics_1.N_i_j_alpha[key]
        else:
            difference_anchor_statistics.N_i_j_alpha[key] \
                = anchor_statistics_2.N_i_j_alpha[key]

    difference_anchor_statistics.R_i_alpha_total = defaultdict(float)
    for key in anchor_statistics_2.R_i_alpha_total:
        if key in anchor_statistics_1.R_i_alpha_total:
            difference_anchor_statistics.R_i_alpha_total[key] \
                = anchor_statistics_2.R_i_alpha_total[key] \
                    - anchor_statistics_1.R_i_alpha_total[key]
        else:
            difference_anchor_statistics.R_i_alpha_total[key] \
                = anchor_statistics_2.R_i_alpha_total[key]

    return difference_anchor_statistics

def add_two_anchor_statistics(
        anchor_statistics_1: "Mmvt_anchor_statistics",
        anchor_statistics_2: "Mmvt_anchor_statistics"
        ) -> "Mmvt_anchor_statistics":
    """
    Perform statistics addition anchor_statistics_1 + anchor_statistics_2.
    """
    sum_anchor_statistics = Mmvt_anchor_statistics(anchor_statistics_1.alpha)
    assert anchor_statistics_1.alpha == anchor_statistics_2.alpha
    sum_anchor_statistics.step_count \
        = anchor_statistics_1.step_count + anchor_statistics_2.step_count
    sum_anchor_statistics.time_count \
        = anchor_statistics_1.time_count + anchor_statistics_2.time_count
    sum_anchor_statistics.first_encounter_time \
        = anchor_statistics_1.first_encounter_time
    sum_anchor_statistics.last_milestone_alias_touched \
        = anchor_statistics_2.last_milestone_alias_touched
    sum_anchor_statistics.last_milestone_time_touched \
        = anchor_statistics_2.last_milestone_time_touched
    sum_anchor_statistics.N_alpha_beta = defaultdict(int)
    for key in anchor_statistics_2.N_alpha_beta:
        if key in anchor_statistics_1.N_alpha_beta:
            sum_anchor_statistics.N_alpha_beta[key] \
                = anchor_statistics_1.N_alpha_beta[key] \
                    + anchor_statistics_2.N_alpha_beta[key]
        else:
            sum_anchor_statistics.N_alpha_beta[key] \
                = anchor_statistics_2.N_alpha_beta[key]
    
    T_alpha_total = anchor_statistics_2.T_alpha_total \
        + anchor_statistics_1.T_alpha_total
    sum_anchor_statistics.T_alpha_total = T_alpha_total
    sum_anchor_statistics.k_alpha_beta = defaultdict(float)
    for key in sum_anchor_statistics.N_alpha_beta:
        if T_alpha_total > 0:
            sum_anchor_statistics.k_alpha_beta[key] \
                = sum_anchor_statistics.N_alpha_beta[key] \
                / T_alpha_total
        else:
            sum_anchor_statistics.k_alpha_beta[key] = 0.0

    sum_anchor_statistics.N_i_j_alpha = defaultdict(int)
    for key in anchor_statistics_2.N_i_j_alpha:
        if key in anchor_statistics_1.N_i_j_alpha:
            sum_anchor_statistics.N_i_j_alpha[key] \
                = anchor_statistics_2.N_i_j_alpha[key] \
                    + anchor_statistics_1.N_i_j_alpha[key]
        else:
            sum_anchor_statistics.N_i_j_alpha[key] \
                = anchor_statistics_2.N_i_j_alpha[key]

    sum_anchor_statistics.R_i_alpha_total = defaultdict(float)
    for key in anchor_statistics_2.R_i_alpha_total:
        if key in anchor_statistics_1.R_i_alpha_total:
            sum_anchor_statistics.R_i_alpha_total[key] \
                = anchor_statistics_2.R_i_alpha_total[key] \
                    + anchor_statistics_1.R_i_alpha_total[key]
        else:
            sum_anchor_statistics.R_i_alpha_total[key] \
                = anchor_statistics_2.R_i_alpha_total[key]

    return sum_anchor_statistics

def make_piecewise_anchor_statistics(
        anchor_statistics_windows_list: list[list[Mmvt_anchor_statistics]]
        ) -> list[list[Mmvt_anchor_statistics]]:
    """
    Take the anchor statistics windows list, which is cumulative, and
    based on the differences between subsequent windows, construct that
    window's contributions to the statistics.
    """
    all_anchors_stats_contrib_list = []
    for alpha in range(len(anchor_statistics_windows_list)):
        anchor_statistics_contributions_list = []
        if anchor_statistics_windows_list[alpha] is None:
            all_anchors_stats_contrib_list.append(None)
            continue
        for i in range(0, len(anchor_statistics_windows_list[alpha])):
            if i == 0:
                contribution = anchor_statistics_windows_list[alpha][i]
            else:
                prev_window = anchor_statistics_windows_list[alpha][i - 1]
                curr_window = anchor_statistics_windows_list[alpha][i]
                contribution = subtract_two_anchor_statistics(prev_window, curr_window)
            anchor_statistics_contributions_list.append(contribution)

        all_anchors_stats_contrib_list.append(anchor_statistics_contributions_list)

    return all_anchors_stats_contrib_list

def make_cumulative_anchor_statistics(
        anchor_statistics_piecewise_list: list[list[Mmvt_anchor_statistics]]
        ) -> list[list[Mmvt_anchor_statistics]]:
    """
    Take the anchor statistics piecewise list, and
    make it cumulative.
    """
    cumulative_anchor_statistics = []
    for alpha in range(len(anchor_statistics_piecewise_list)):
        cumulative_windows = []
        if anchor_statistics_piecewise_list[alpha] is None:
            cumulative_windows.append(None)
            continue

        for i in range(len(anchor_statistics_piecewise_list[alpha])):
            anchor_stat = anchor_statistics_piecewise_list[alpha][i]
            problem = False
            for key in anchor_stat.k_alpha_beta:
                # NOTE: save this for if we want to restore it someday
                #assert anchor_stat.k_alpha_beta[key] >= 0.0, \
                #        "negative k_alpha_beta values not allowed."
                if anchor_stat.k_alpha_beta[key] < 0.0:
                    problem = True

            if problem:
                continue
            if i == 0:
                cumulative_windows.append(anchor_stat)
            else:
                cumulative_windows.append(
                    add_two_anchor_statistics(
                        cumulative_windows[i - 1],
                        anchor_stat
                    )
                )
            
        cumulative_anchor_statistics.append(cumulative_windows)
    
    return cumulative_anchor_statistics

def pare_down_anchor_statistics(
        anchor_stats_alpha_list: list[list[Mmvt_anchor_statistics]],
        window_index_list: list[int]
) -> None:
    """
    Pare down the anchor statistics to only include the specified windows.
    Ensure that at least one member is kept for each non-None, non-empty alpha:
    - If the selected indices yield no members (because the list is short),
      keep the last member of the original list.
    """
    for alpha in range(len(anchor_stats_alpha_list)):
        if anchor_stats_alpha_list[alpha] is not None:
            orig = anchor_stats_alpha_list[alpha]
            if len(orig) == 0:
                # leave as empty list
                anchor_stats_alpha_list[alpha] = []
                continue
            selected = [orig[i] for i in window_index_list if 0 <= i < len(orig)]
            if not selected:
                # If nothing was selected (lists too short), keep the last member
                anchor_stats_alpha_list[alpha] = [orig[-1]]
            else:
                anchor_stats_alpha_list[alpha] = selected
    return