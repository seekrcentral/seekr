"""
modules/scales/mmvt_analyzer.py

MMVT analysis methods that are scale-specific.
"""

import typing

import seekr.modules.structures as structures
import seekr.modules.scales.molecular_dynamics.mmvt_analyzer as mmvt_md_analyzer
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics
import seekr.modules.engines.browndye.analyzer as browndye_analyzer
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_statistics

def make_anchor_stats_alpha(
        model: structures.Seekr_model,
        anchor: structures.Anchor,
        stage_index: int,
        min_time_ps: float,
        max_time_ps: float,
        time_between_statistics: float | None,
        ) -> typing.Tuple[mmvt_anchor_statistics.Mmvt_anchor_statistics | None,
                          float | None]:
    """
    Decide how to populate mmvt_anchor_statistics object based on sampling
    method.
    """
    if "molecular_dynamics" in anchor.scale_types:
        stats, time_between_stats = mmvt_md_analyzer\
            .make_anchor_stats_alpha_md(model, anchor, stage_index,
                                          min_time_ps, max_time_ps,
                                          time_between_statistics)
    else:
        stats, time_between_stats = None, None

    return stats, time_between_stats

def make_bulk_anchor_stats(
        model: structures.Seekr_model,
        anchor: structures.Anchor,
        ) -> bulk_anchor_statistics.Bulk_anchor_statistics | None:
    """
    Decide how to populate bulk anchor statistics based on sampling method.
    """
    if "brownian_dynamics" in anchor.scale_types:
        return browndye_analyzer.parse_browndye_results(model, anchor, False)
    else:
        pass
    return