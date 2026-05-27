from __future__ import annotations
"""
modules/engines/sda/analyzer.py

SDA analysis methods to extract meaningful statistics from outputs.
"""

import os
import glob
from collections import defaultdict

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.sda.base as sda_base
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics

def parse_sda_results(
        model: structures.Seekr_model,
        bulk_anchor: structures.Anchor,
        set_name: str,
        sample_error_from_normal: bool = False
        ) -> bulk_anchor_statistics.Bulk_anchor_statistics:
    """
    Parse the SDA results files to extract statistics.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine.type == "sda"
    bd_directory = os.path.join(
        model.directory, "anchor_sets", set_name, 
        bd_engine.b_surface_directory)
    b_surface_output_file_glob = os.path.join(
        bd_directory, bd_engine.results_glob)
    output_file_list = glob.glob(b_surface_output_file_glob)
    output_file_list = base.order_files_numerically(output_file_list)
    bd_transition_counts = {}
    bd_transition_probabilities = {}
    outer_milestone_indices = []

    # Compute milestone indices by radii
    milestone_indices_by_radius = {}
    for neighbor_anchor_index in bulk_anchor.neighbor_anchor_indices:
        neighbor_anchor = model.anchors[neighbor_anchor_index]
        spherical_milestones_in_neighbor_anchor, this_anchors_bd_cv_index \
            = engines_preparer\
                .get_spherical_milestones_and_cv_index_in_neighbor_anchor(
                    model, neighbor_anchor)
        for milestone in spherical_milestones_in_neighbor_anchor:
            radius = milestone.variables['avg_value']
            milestone_indices_by_radius[radius] = milestone.index

    if len(output_file_list) > 0:
        if len(output_file_list) == 1:
            compute_rate_const_full_path = os.path.join(
                bd_engine.aux_directory, "Bootstrap_multiCPU.py")
        else:
            compute_rate_const_full_path = os.path.join(
                bd_engine.bin_directory, "nos2rates")
        k_ons_src, k_on_errors_src, reaction_probabilities, \
            reaction_probability_errors, transition_counts = \
            sda_base.run_compute_rate_constant(
                output_file_list,
                milestone_indices_by_radius,
                sample_error_from_normal=sample_error_from_normal)
        
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