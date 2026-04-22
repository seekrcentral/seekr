"""
modules/engines/browndye/analyzer.py

Browndye analysis methods to extract meaningful statistics from outputs.
"""

import os
import glob
from collections import defaultdict

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics

def parse_browndye_results(
        model: structures.Seekr_model,
        bulk_anchor: structures.Anchor,
        set_name: str,
        bd_sample_from_normal=False
        ) -> bulk_anchor_statistics.Bulk_anchor_statistics:
    """
    Parse the Browndye results files to extract statistics.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine.type == "browndye"
    bd_directory = os.path.join(
        model.directory, "anchor_sets", set_name, 
        bd_engine.b_surface_directory)
    browndye_bin_dir = bd_engine.binary_directory
    b_surface_output_file_glob = os.path.join(
        bd_directory, bd_engine.results_glob)
    output_file_list = glob.glob(b_surface_output_file_glob)
    output_file_list = base.order_files_numerically(output_file_list)
    bd_transition_counts = {}
    bd_transition_probabilities = {}
    outer_milestone_indices = []
    if len(output_file_list) > 0:
        compute_rate_const_full_path = os.path.join(
            browndye_bin_dir, "compute_rate_constant")
        k_ons_src, k_on_errors_src, reaction_probabilities, \
            reaction_probability_errors, transition_counts = \
            browndye_base.run_compute_rate_constant(
                compute_rate_const_full_path, output_file_list, 
                sample_error_from_normal=bd_sample_from_normal)
        
        bd_transition_counts["b_surface"] = transition_counts
        bd_transition_probabilities["b_surface"] \
            = reaction_probabilities
        b_surface_k_ons_src = k_ons_src
        b_surface_k_on_errors_src = k_on_errors_src
        for milestone in bulk_anchor.milestones:
            neighbor_anchor = model.anchors[milestone.neighbor_anchor_index]
            transition_counts_anchor = defaultdict(int)
            transition_probabilities_anchor = defaultdict(float)
            spherical_milestones_in_neighbor_anchor, this_anchors_bd_cv_index \
                = engines_preparer.get_spherical_milestones_and_cv_index_in_neighbor_anchor(
                    model, neighbor_anchor)
            inner_milestone_index = spherical_milestones_in_neighbor_anchor[1].index
            outer_milestone_index = spherical_milestones_in_neighbor_anchor[0].index
            outer_milestone_indices.append(outer_milestone_index)
            assert inner_milestone_index in transition_counts
            assert outer_milestone_index in transition_counts
            transition_counts_anchor[inner_milestone_index] \
                = transition_counts[inner_milestone_index]
            transition_counts_anchor["escaped"] \
                = transition_counts[outer_milestone_index] \
                - transition_counts[inner_milestone_index]
            transition_counts_anchor["total"] \
                = transition_counts[outer_milestone_index]
            if transition_counts_anchor["escaped"] == 0:
                transition_probabilities_anchor[
                    inner_milestone_index] = 1.0
                transition_probabilities_anchor["escaped"] \
                    = 0
            else:
                transition_probabilities_anchor[
                    inner_milestone_index] \
                    = transition_counts_anchor[
                        inner_milestone_index] \
                    / transition_counts_anchor["total"]
                transition_probabilities_anchor["escaped"] \
                    = transition_counts_anchor["escaped"] \
                    / transition_counts_anchor["total"]
            
            bd_transition_counts[outer_milestone_index] \
                = transition_counts_anchor
            bd_transition_probabilities[outer_milestone_index] \
                    = transition_probabilities_anchor
            
        bulk_anchor_stats = bulk_anchor_statistics.Bulk_anchor_statistics(
            transition_counts=bd_transition_counts,
            transition_probabilities=bd_transition_probabilities,
            k_ons_src=b_surface_k_ons_src,
            k_on_errors_src=b_surface_k_on_errors_src,
            outer_milestone_indices=outer_milestone_indices
        )
        return bulk_anchor_stats
    else:
        return None