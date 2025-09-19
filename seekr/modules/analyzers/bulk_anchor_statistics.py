"""
modules/analyzers/bulk_anchor_statistics.py

An object for containing bulk anchor statistics for k-on calculations.
"""

import random
from attrs import define, field, Factory

# TODO: move this lonely class to modules/analyzers/base.py??

@define
class Bulk_anchor_statistics:
    transition_counts: dict = field(default=Factory(dict))
    transition_probabilities: dict = field(default=Factory(dict))
    k_ons_src: dict = field(default=Factory(dict))
    k_on_errors_src: dict = field(default=Factory(dict))
    outer_milestone_indices: list = field(default=Factory(list))

    def check(self) -> bool:
        error_warning_base = "Bulk anchor is missing adequate "\
            "statistics. Consider running simulations of bulk anchor "\
            "for additional time. Transition counts from bulk anchor "\
            "to milestone(s) {0} were found to be zero. You may skip this "\
            "check with the --skip_checks (-s) option.\n"
        no_stats_milestones = []
        for count_key in self.transition_counts["b_surface"]:
            if count_key in ["escaped", "total", "stuck"]:
                continue
            if self.transition_counts["b_surface"][count_key] < 1:
                no_stats_milestones.append(count_key)
                
        if len(no_stats_milestones) > 0:
            print(error_warning_base.format(", ".join(no_stats_milestones)))
            return False
        return True
    
def resample_bulk_anchor_statistics(
        bulk_stats: Bulk_anchor_statistics,
    ) -> Bulk_anchor_statistics:
    resampled_k_ons_src = {}
    resampled_k_on_errors_src = {}
    for key in bulk_stats.k_ons_src:
        if key not in bulk_stats.k_on_errors_src:
            continue
        mean = bulk_stats.k_ons_src[key]
        stddev = bulk_stats.k_on_errors_src[key]
        if mean == 0 or stddev == 0 or stddev is None:
            resampled_k_ons_src[key] = mean
            resampled_k_on_errors_src[key] = stddev
            continue
        resampled_value = max(0, random.gauss(mean, stddev))
        resampled_k_ons_src[key] = resampled_value
        resampled_k_on_errors_src[key] = stddev
    
    return Bulk_anchor_statistics(
        transition_counts=bulk_stats.transition_counts,
        transition_probabilities=bulk_stats.transition_probabilities,
        k_ons_src=resampled_k_ons_src,
        k_on_errors_src=resampled_k_on_errors_src,
        outer_milestone_indices=bulk_stats.outer_milestone_indices
    )