from __future__ import annotations
"""
test_analyze.py

Smoke tests for seekr.analyze — verify that analyze() returns without
exception after all non-BD stages have been run.
"""

import glob
import os
from collections import defaultdict
from unittest.mock import patch

import pytest

import seekr.modules.analyzers.base as analyzer_base
import seekr.analyze as analyze
import seekr.modules.analyzers.bulk_anchor_statistics as bulk_anchor_stats_mod
import seekr.modules.engines.preparer as engines_preparer

# ---------------------------------------------------------------------------
# Helpers for writing synthetic MMVT output files
# ---------------------------------------------------------------------------

_SYNTHETIC_COUNTS = 10
_SYNTHETIC_TIME_PER_ALIAS = 50.0
_SYNTHETIC_STEP_COUNT = 10000


def _find_mmvt_stage(model):
    """Return the first MMVT MD stage and its 1-based index, or (None, None)."""
    for i, stage in enumerate(model.stages):
        if (stage.scale_type == "molecular_dynamics"
                and stage.sampling_method.type == "mmvt"):
            return stage, i + 1
    return None, None


def _write_synthetic_out_files(model):
    """
    Write one synthetic MMVT statistics window to disk for every MD anchor
    in the model.  The window contains enough counts for check_extraction to
    confirm connectivity between all end-anchor states.
    """
    stage, stage_index = _find_mmvt_stage(model)
    if stage is None:
        return
    set_name = stage.set_name

    for anchor in model.anchors:
        if anchor.bulk_anchor:
            continue
        if "molecular_dynamics" not in anchor.scale_types:
            continue
        stage_params_dir = anchor.get_stage_parameters_by_index(
            stage_index)["directory"]
        out_dir = os.path.join(
            model.directory, "anchor_sets", set_name,
            anchor.directory, stage_params_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Remove any stale .out files from a previous run so that only the
        # synthetic statistics are present when the analyzer globs *.out.
        for stale in glob.glob(os.path.join(out_dir, "*.out")):
            os.remove(stale)

        aliases = [m.alias_index for m in anchor.milestones]
        total_time = _SYNTHETIC_TIME_PER_ALIAS * len(aliases)

        lines = [
            "# window start\n",
            f"step_count {_SYNTHETIC_STEP_COUNT}\n",
            f"time_count {total_time}\n",
            "first_encounter_time 0.0\n",
            f"last_milestone_alias_touched {aliases[0]}\n",
            f"last_milestone_time_touched {_SYNTHETIC_TIME_PER_ALIAS}\n",
        ]
        # Ensure N_alpha_beta[b] > 0 for every alias b, and
        # N_i_j_alpha[(a, b)] > 0 for every pair a != b.
        for a in aliases:
            for b in aliases:
                lines.append(f"milestone_{a}_{b}_count: {_SYNTHETIC_COUNTS}\n")
        for a in aliases:
            lines.append(f"milestone_{a}_time: {_SYNTHETIC_TIME_PER_ALIAS}\n")
        lines.append("# window end\n")

        out_file = os.path.join(out_dir, "milestoning_restart1_swarm_0.out")
        with open(out_file, "w") as f:
            f.writelines(lines)


def _make_synthetic_bulk_stats(model):
    """
    Build a Bulk_anchor_statistics from model topology alone —
    no BD simulation or Browndye/SDA binary required.
    The returned object passes bulk_stats.check() and supplies the
    fields consumed by calculate_k_on_kinetics.
    """
    bulk_anchor = next((a for a in model.anchors if a.bulk_anchor), None)
    if bulk_anchor is None:
        return None

    outer_milestone_indices = []
    transition_counts = {}
    transition_probabilities = {}
    k_ons_src = defaultdict(float)
    k_on_errors_src = defaultdict(float)

    for milestone in bulk_anchor.milestones:
        neighbor = model.anchors[milestone.neighbor_anchor_index]
        spherical, _ = (
            engines_preparer
            .get_spherical_milestones_and_cv_index_in_neighbor_anchor(
                model, neighbor))
        # sorted descending by avg_value: spherical[0] = outer, [1] = inner
        outer_m, inner_m = spherical[0], spherical[1]
        outer_idx = outer_m.index
        inner_idx = inner_m.index
        outer_milestone_indices.append(outer_idx)

        transition_counts[outer_idx] = {
            inner_idx: 900, "escaped": 100, "stuck": 0, "total": 1000}
        transition_probabilities[outer_idx] = {
            inner_idx: 0.9, "escaped": 0.1}
        k_ons_src[outer_idx] = 1.0e8
        k_on_errors_src[outer_idx] = 1.0e7

    # b_surface summary required by bulk_stats.check()
    b_surface = {}
    for outer_idx in outer_milestone_indices:
        for k, v in transition_counts[outer_idx].items():
            b_surface[k] = b_surface.get(k, 0) + v
    transition_counts["b_surface"] = b_surface

    return bulk_anchor_stats_mod.Bulk_anchor_statistics(
        transition_counts=transition_counts,
        transition_probabilities=transition_probabilities,
        k_ons_src=k_ons_src,
        k_on_errors_src=k_on_errors_src,
        outer_milestone_indices=outer_milestone_indices)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analyze_returns_result(seekr_run):
    """analyze() returns a non-None result after simulations."""
    model, _, scenario_name = seekr_run
    # Only analyze models that completed the MMVT stage
    mmvt_stages = [s for s in model.stages if s.name == "MMVT"]
    if not mmvt_stages:
        return  # no MMVT stage in this scenario; skip silently
    if ("hg_metad_dist" in scenario_name) or ("hg_smd_dist" in scenario_name)\
            or ("hg_encounters_bd" in scenario_name) \
            or ("hg_trajectories_bd" in scenario_name):
        result = analyze.analyze(model, skip_checks=True,
                                 only_extract_data=True)
    else:
        result = analyze.analyze(model, skip_checks=True)
    assert result is not None, (
        f"[{scenario_name}] analyze() returned None"
    )


def test_analyze_full_pipeline(seekr_prepared):
    """
    Full MMVT analysis pipeline driven by synthetic statistics — no
    simulation is required.  Exercises check_extraction,
    fill_out_and_process_data_samples, and error_analysis for every
    scenario type, including those with BD bulk anchors.
    """
    model, _, scenario_name = seekr_prepared

    stage, stage_index = _find_mmvt_stage(model)
    if stage is None:
        return  # no MMVT stage in this scenario

    # Populate model.anchors with the set-specific anchor objects so that
    # anchor.directory and stage_parameters align with the on-disk paths.
    stage.load_sets_into_model(model)

    _write_synthetic_out_files(model)

    if model.using_scale("brownian_dynamics"):
        synthetic_bulk = _make_synthetic_bulk_stats(model)
        with patch(
                "seekr.modules.scales.mmvt_analyzer.make_bulk_anchor_stats",
                return_value=synthetic_bulk):
            result = analyze.analyze(model, skip_checks=False)
    else:
        result = analyze.analyze(model, skip_checks=False)

    assert result is not None, \
        f"[{scenario_name}] analyze() returned None"
    assert result.main_data_window is not None, \
        f"[{scenario_name}] main_data_window is None"
    
    result.print_results()
    result.make_image_directory(analyzer_base.DEFAULT_IMAGE_DIR)
    result.save_plots(analyzer_base.DEFAULT_IMAGE_DIR)


# ---------------------------------------------------------------------------
# Error analysis method tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _toy_metad_1cv_analysis_state(tmp_path_factory):
    """Module-scoped fixture: prepare toy_metad_1cv, write synthetic
    out-files, return model ready for analyze() calls."""
    import seekr.tests.create_seekr_input as create_seekr_input
    import seekr.prepare as prepare
    root_dir = str(tmp_path_factory.mktemp("toy_metad_1cv_err"))
    seekr_input = create_seekr_input.create_toy_input(root_dir)
    old_dir = os.getcwd()
    try:
        model, file_path = prepare.prepare(seekr_input, force_overwrite=False)
        model.directory = os.path.abspath(os.path.dirname(file_path))
    finally:
        os.chdir(old_dir)
    stage, _ = _find_mmvt_stage(model)
    assert stage is not None, "toy_metad_1cv must have an MMVT stage"
    stage.load_sets_into_model(model)
    _write_synthetic_out_files(model)
    return model


def test_error_analysis_resample(_toy_metad_1cv_analysis_state):
    """error_analysis_method='resample' runs without exception."""
    model = _toy_metad_1cv_analysis_state
    result = analyze.analyze(model, skip_checks=False,
                             error_analysis_method="resample")
    assert result is not None
    assert result.main_data_window is not None


def test_error_analysis_montecarlo(_toy_metad_1cv_analysis_state):
    """error_analysis_method='montecarlo' runs without exception."""
    model = _toy_metad_1cv_analysis_state
    result = analyze.analyze(model, skip_checks=False,
                             error_analysis_method="montecarlo")
    assert result is not None
    assert result.main_data_window is not None

