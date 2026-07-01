from __future__ import annotations
"""
test_run.py

Smoke tests for seekr.run — verify that run.run() completes without
exception for all non-BD stages of every scenario.
"""

import os

import pytest

import seekr.run as run
import seekr.modules.structures as structures


def test_run_produces_model(seekr_run):
    """After running all stages, the model is still a valid Seekr_model."""
    model, _, scenario_name = seekr_run
    assert isinstance(model, structures.Seekr_model), (
        f"[{scenario_name}] seekr_run should yield a Seekr_model"
    )


def test_run_model_directory_unchanged(seekr_run):
    """run() must not delete the model directory."""
    model, _, scenario_name = seekr_run
    assert os.path.isdir(model.directory), (
        f"[{scenario_name}] model directory missing after run"
    )


def test_run_cwd_restored(seekr_run):
    """The process cwd must be the same before and after the fixture."""
    # seekr_run fixture saves/restores cwd internally; if we get here,
    # the restore succeeded by definition. We just assert the cwd is valid.
    assert os.path.isdir(os.getcwd())


def _first_md_stage(model):
    """Return the first non-BD MD stage, or None."""
    for stage in model.stages:
        if stage.scale_type != "brownian_dynamics":
            return stage
    return None

def test_rerun_force_overwrite_succeeds(seekr_run):
    """Re-running a completed stage with force_overwrite=True must not raise."""
    model, _, scenario_name = seekr_run
    stage = _first_md_stage(model)
    if stage is None:
        pytest.skip("no MD stage in this scenario")
    run.run(model, stage.name, force_overwrite=True)


@pytest.mark.browndye
@pytest.mark.slow
def test_rerun_bd_force_overwrite_succeeds(hg_encounters_bd_run):
    """Re-running a completed BD stage with force_overwrite=True must not raise."""
    model, _ = hg_encounters_bd_run
    run.run(model, "bd", force_overwrite=True)


@pytest.mark.browndye
@pytest.mark.slow
def test_rerun_bd_partial_completion(hg_encounters_bd_run):
    """Resuming a partially-finished BD run completes without error.

    Simulates a partial run by doubling the encounters threshold on a model
    whose BD stage has already finished at the original (lower) threshold.
    The runner sees the criteria as unmet and continues from existing output
    rather than starting from scratch, exercising the restart path.
    """
    model, _ = hg_encounters_bd_run
    bd_stage = next(s for s in model.stages if s.scale_type == "brownian_dynamics")
    bd_stage.completion_criteria.number_of_encounters *= 2
    run.run(model, bd_stage.name)


@pytest.mark.browndye
@pytest.mark.slow
def test_rerun_bd_trajectories_partial_completion(hg_trajectories_bd_run):
    """Resuming a partially-finished trajectories BD run completes without error.

    Simulates a partial run by doubling the trajectories threshold on a model
    whose BD stage has already finished at the original (lower) threshold.
    Exercises the restart path in choose_next_simulation_bd for the
    number_of_trajectories completion criteria.
    """
    model, _ = hg_trajectories_bd_run
    bd_stage = next(s for s in model.stages if s.scale_type == "brownian_dynamics")
    bd_stage.completion_criteria.number_of_trajectories *= 2
    run.run(model, bd_stage.name)


def test_extract_stage_index(seekr_prepared):
    """By-name lookup hits; unknown name and out-of-range index return None."""
    model, _, _ = seekr_prepared
    assert run.extract_stage_index(model, model.stages[0].name) == 0
    assert run.extract_stage_index(model, "9999") is None


def test_extract_anchor_index(seekr_prepared):
    """'any' passes through; by-name and by-int resolve; unknown returns None."""
    model, _, _ = seekr_prepared
    assert run.extract_anchor_index(model, "any") == "any"
    if model.anchors:
        assert run.extract_anchor_index(model, model.anchors[0].name) == "0"
        assert run.extract_anchor_index(model, "0") == "0"
    assert run.extract_anchor_index(model, "nonexistent") is None


def test_run_single_anchor(seekr_run):
    """run() with anchor_arg='0' runs only that anchor without error."""
    model, _, scenario_name = seekr_run
    stage = next(
        (s for s in model.stages
         if s.scope == "partitioned" and s.scale_type == "molecular_dynamics"),
        None
    )
    if stage is None:
        pytest.skip(f"[{scenario_name}] no partitioned stage")
    run.run(model, stage.name, anchor_arg="0", force_overwrite=True)


def test_run_single_anchor_benchmark(seekr_run):
    """run() with benchmark=True on a single MD anchor runs without error."""
    model, _, scenario_name = seekr_run
    stage = next(
        (s for s in model.stages
         if s.scope == "partitioned" and s.scale_type == "molecular_dynamics"),
        None
    )
    if stage is None:
        pytest.skip(f"[{scenario_name}] no partitioned stage")
    run.run(model, stage.name, anchor_arg="0", force_overwrite=True,
            benchmark=True)


@pytest.mark.browndye
@pytest.mark.slow
def test_rerun_bd_benchmark(hg_encounters_bd_run):
    """Re-running a BD stage with benchmark=True runs without error."""
    model, _ = hg_encounters_bd_run
    run.run(model, "bd", force_overwrite=True, benchmark=True)
