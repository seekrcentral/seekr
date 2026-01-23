"""
modules/scales/molecular_dynamics/mmvt_analyzer.py

Molecular dynamics-specific MMVT analysis methods.
"""

import os
import glob
import typing

import numpy as np

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics

def make_anchor_stats_alpha_md(
        model: structures.Seekr_model,
        anchor: structures.Anchor,
        stage_index: int,
        min_time_ps: float,
        max_time_ps: float,
        time_between_statistics: float | None,
        ) -> typing.Tuple[mmvt_anchor_statistics.Mmvt_anchor_statistics,
                          float]:
    """
    Create and populate an mmvt_anchor_statistics object for an anchor
    simulated with molecular dynamics.
    """
    # TODO: add a way to obtain the output file glob from the sampling method?
    #  because this feels hacky.
    output_file_glob = os.path.join(
        model.directory, anchor.directory, 
        anchor.get_stage_parameters_by_index(stage_index)["directory"],
        "*.out")
    output_file_list = glob.glob(output_file_glob)
    output_file_list = base.order_files_numerically(
        output_file_list)
    md_timestep = model.get_timestep_by_type("molecular_dynamics")
    min_step = int(min_time_ps / md_timestep)
    max_step = int(max_time_ps / md_timestep)
    anchor_stats_alpha, time_between_statistics_this_anchor \
            = mmvt_anchor_statistics.read_output_file_list(
            output_file_list, anchor, min_step, max_step)
    if time_between_statistics is None:
        time_between_statistics = time_between_statistics_this_anchor
    if time_between_statistics is not None:
        assert np.isclose(time_between_statistics_this_anchor, 
                            time_between_statistics), \
                            "Time between statistics does not match"
    return anchor_stats_alpha, time_between_statistics