from __future__ import annotations
"""
conftest.py

Shared pytest fixtures.

Parametrized pipeline fixtures
-------------------------------
  seekr_prepared  -- runs prepare.prepare() for each scenario
  seekr_run       -- runs all non-BD stages for each scenario

Legacy single-scenario fixtures (used by test_modules/)
--------------------------------------------------------
  host_guest_mmvt_model_input_persistent
  host_guest_mmvt_model_input
  host_guest_mmvt_model_persistent
  host_guest_mmvt_model
"""

import os
import copy

import pytest

import seekr.prepare as prepare
import seekr.run as run
import seekr.status as status
import seekr.tests.create_seekr_input as create_seekr_input

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict = {
    # toy scenarios
    "toy_metad_1cv":          dict(toy=True),
    "toy_smd_1cv":            dict(toy=True, use_smd=True),
    "toy_metad_2cvs":         dict(toy=True, use_2cvs=True),
    "toy_defineAnchorsLater": dict(toy=True, define_anchors_later=True),
    # host-guest scenarios
    "hg_metad_dist":          dict(),
    "hg_smd_dist":            dict(use_smd=True),
    "hg_metad_rmsd":          dict(use_rmsd_cv=True),
    "hg_metad_dist_bd":       dict(bd=True),
    "hg_metad_dist_sda":      dict(bd=True, use_sda=True),
    "hg_encounters_bd":        dict(bd=True, use_encounters_criteria=True),
    "hg_trajectories_bd":      dict(bd=True, n_trajectories=100),
}

SCENARIO_PARAMS = [
    pytest.param("toy_metad_1cv"),
    pytest.param("toy_smd_1cv"),
    pytest.param("toy_metad_2cvs"),
    pytest.param("toy_defineAnchorsLater"),
    pytest.param("hg_metad_dist"),
    pytest.param("hg_smd_dist",       marks=pytest.mark.slow),
    pytest.param("hg_metad_rmsd",     marks=[pytest.mark.slow, pytest.mark.rmsd]),
    pytest.param("hg_metad_dist_bd",  marks=[pytest.mark.slow, pytest.mark.browndye]),
    pytest.param("hg_metad_dist_sda", marks=[pytest.mark.slow, pytest.mark.sda]),
    pytest.param("hg_encounters_bd",  marks=[pytest.mark.slow, pytest.mark.browndye]),
    pytest.param("hg_trajectories_bd", marks=[pytest.mark.slow, pytest.mark.browndye]),
]

# ---------------------------------------------------------------------------
# Custom CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False,
        help="Run the full test suite including slow scenarios "
             "(default: only fast scenarios)."
    )
    parser.addoption(
        "--rmsd", action="store_true", default=False,
        help="Run tests involving RMSD calculations "
             "(default: skipped)."
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--full"):
        skip_slow = pytest.mark.skip(reason="slow scenario; use --full to run")
        for item in items:
            if "slow" in [m.name for m in item.iter_markers()]:
                item.add_marker(skip_slow)
    if not config.getoption("--rmsd"):
        skip_rmsd = pytest.mark.skip(reason="RMSD scenario; use --rmsd to run")
        for item in items:
            if "rmsd" in [m.name for m in item.iter_markers()]:
                item.add_marker(skip_rmsd)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_seekr_input(scenario_name: str, root_dir: str):
    cfg = SCENARIO_CONFIGS[scenario_name].copy()
    if cfg.pop("toy", False):
        return create_seekr_input.create_toy_input(root_dir, **cfg)
    return create_seekr_input.create_host_guest_input(root_dir, **cfg)


def _prepare_model(seekr_input):
    old_dir = os.getcwd()
    try:
        model, model_xml_path = prepare.prepare(
            seekr_input, force_overwrite=False)
        model.directory = os.path.abspath(os.path.dirname(model_xml_path))
    finally:
        os.chdir(old_dir)
    return model, model_xml_path


# ---------------------------------------------------------------------------
# Parametrized pipeline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=SCENARIO_PARAMS)
def seekr_prepared(request, tmp_path_factory):
    """Prepare a SEEKR model for a given scenario. Returns (model, xml_path, name)."""
    scenario_name = request.param
    root_dir = str(tmp_path_factory.mktemp(scenario_name))
    seekr_input = _create_seekr_input(scenario_name, root_dir)
    model, model_xml_path = _prepare_model(seekr_input)
    return model, model_xml_path, scenario_name


@pytest.fixture(scope="session")
def seekr_run(seekr_prepared):
    """
    Run all non-brownian-dynamics stages sequentially.
    Calls status() after each stage as a smoke test.
    Returns the same (model, xml_path, name) tuple.
    """
    model, model_xml_path, scenario_name = seekr_prepared
    old_dir = os.getcwd()
    try:
        for stage in model.stages:
            #if stage.scale_type == "brownian_dynamics":
            #    continue
            run.run(model, stage.name)
            status.status(model, "progress", print_json=False)
    finally:
        os.chdir(old_dir)
    return model, model_xml_path, scenario_name


# ---------------------------------------------------------------------------
# Legacy single-scenario fixtures (backward-compatible)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def host_guest_mmvt_model_input_persistent(tmp_path_factory):
    """Session-scoped host-guest model input (with BD, amber FF)."""
    rootdir = str(tmp_path_factory.mktemp("hostguest_mmvt"))
    return create_seekr_input.create_host_guest_input(rootdir, bd=True)


@pytest.fixture()
def host_guest_mmvt_model_input(host_guest_mmvt_model_input_persistent):
    """Per-test deep copy of the model input."""
    return copy.deepcopy(host_guest_mmvt_model_input_persistent)


@pytest.fixture(scope="session")
def host_guest_mmvt_model_persistent(host_guest_mmvt_model_input_persistent):
    """Session-scoped prepared host-guest model."""
    model, _ = _prepare_model(host_guest_mmvt_model_input_persistent)
    return model


@pytest.fixture
def host_guest_mmvt_model(host_guest_mmvt_model_persistent):
    """Per-test deep copy of the prepared host-guest model."""
    return copy.deepcopy(host_guest_mmvt_model_persistent)


@pytest.fixture(scope="session")
def hg_encounters_bd_run(tmp_path_factory):
    """
    Prepare and run the BD stage of the hg_encounters_bd scenario.
    Uses Number_of_encounters_completion_criteria with 10 encounters.
    Requires Browndye to be installed; skipped without --full.
    Returns (model, model_xml_path).
    """
    root_dir = str(tmp_path_factory.mktemp("hg_encounters_bd"))
    seekr_input = _create_seekr_input("hg_encounters_bd", root_dir)
    model, model_xml_path = _prepare_model(seekr_input)
    old_dir = os.getcwd()
    try:
        run.run(model, "bd")
    finally:
        os.chdir(old_dir)
    return model, model_xml_path


@pytest.fixture(scope="session")
def hg_trajectories_bd_run(tmp_path_factory):
    """
    Prepare and run the BD stage of the hg_trajectories_bd scenario.
    Uses Number_of_trajectories_completion_criteria with 100 trajectories.
    Requires Browndye to be installed; skipped without --full.
    Returns (model, model_xml_path).
    """
    root_dir = str(tmp_path_factory.mktemp("hg_trajectories_bd"))
    seekr_input = _create_seekr_input("hg_trajectories_bd", root_dir)
    model, model_xml_path = _prepare_model(seekr_input)
    old_dir = os.getcwd()
    try:
        run.run(model, "bd")
    finally:
        os.chdir(old_dir)
    return model, model_xml_path