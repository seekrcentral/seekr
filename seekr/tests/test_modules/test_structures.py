"""
test_structures.py

Unit tests for seekr/modules/structures.py methods with incomplete coverage.

Coverage targets
-----------------
Milestone  : get_cv_indices
Anchor     : make_var_string, get_stage_output_directory,
             change_positions_filename, has_files, get_cv_indices
Seekr_model: get_toy_settings, save, save_next_model, save_cvs/load_cvs,
             save_selections/load_selections, save_anchor_set/load_anchor_set
module     : load_model
"""

import pytest

import seekr.modules.structures as structures


# ---------------------------------------------------------------------------
# Milestone
# ---------------------------------------------------------------------------

def test_milestone_get_cv_indices():
    """Milestone.get_cv_indices returns the cv index from 'me_' variables."""
    m = structures.Milestone(variables={"me_0": 1.0, "neighbor_0": 2.0})
    assert m.get_cv_indices() == [0]


def test_milestone_get_cv_indices_multiple():
    """Milestone.get_cv_indices handles multiple cv dimensions."""
    m = structures.Milestone(variables={
        "me_0": 0.5, "neighbor_0": 1.0,
        "me_1": 0.3, "neighbor_1": 0.6,
    })
    assert sorted(m.get_cv_indices()) == [0, 1]


# ---------------------------------------------------------------------------
# Anchor  (host_guest_mmvt_model is a per-test deep copy — safe to mutate)
# ---------------------------------------------------------------------------

def test_anchor_make_var_string(host_guest_mmvt_model):
    """Anchor.make_var_string returns a non-empty string for a prepared anchor."""
    anchor = host_guest_mmvt_model.anchors[0]
    result = anchor.make_var_string()
    assert isinstance(result, str)
    assert len(result) > 0


def test_anchor_get_stage_output_directory(host_guest_mmvt_model):
    """Anchor.get_stage_output_directory returns a str for every stage."""
    anchor = host_guest_mmvt_model.anchors[0]
    for stage_name in anchor.stage_parameters:
        result = anchor.get_stage_output_directory(stage_name)
        assert isinstance(result, str)


def test_anchor_change_positions_filename(host_guest_mmvt_model):
    """Anchor.change_positions_filename updates the 'positions' key."""
    anchor = host_guest_mmvt_model.anchors[0]
    valid_params = [
        v for v in anchor.stage_parameters.values()
        if v is not None and "index" in v
    ]
    assert valid_params, "Expected at least one populated stage_parameters entry."
    stage_index = min(v["index"] for v in valid_params)
    # change_positions_filename(new_name, stage_number) internally adds 1
    anchor.change_positions_filename("new_positions.pdb", stage_index - 1)
    assert anchor.get_stage_parameters_by_index(stage_index)["positions"] \
        == "new_positions.pdb"


def test_anchor_has_files_false(host_guest_mmvt_model):
    """Anchor.has_files returns False for a prepared-only (never run) model."""
    model = host_guest_mmvt_model
    anchor = model.anchors[0]
    assert anchor.has_files(model, model.directory) is False


def test_anchor_get_cv_indices(host_guest_mmvt_model):
    """Anchor.get_cv_indices returns a list of ints from 'value_' variables."""
    anchor = host_guest_mmvt_model.anchors[0]
    result = anchor.get_cv_indices()
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


# ---------------------------------------------------------------------------
# Seekr_model.get_toy_settings  (parametrized over all scenarios)
# ---------------------------------------------------------------------------

def test_get_toy_settings(seekr_prepared):
    """get_toy_settings works for toy models; raises AssertionError otherwise."""
    model, _, scenario_name = seekr_prepared
    if model.using_toy():
        result = model.get_toy_settings()
        assert result is not None, \
            f"[{scenario_name}] get_toy_settings() returned None"
    else:
        with pytest.raises(AssertionError):
            model.get_toy_settings()


# ---------------------------------------------------------------------------
# Seekr_model save / load  (per-test deep copy + tmp_path — no side effects)
# ---------------------------------------------------------------------------

def test_model_save_and_load_model(host_guest_mmvt_model, tmp_path):
    """Round-trip save → load_model restores key model fields."""
    model = host_guest_mmvt_model
    model_path = str(tmp_path / "model.json")
    model.save(model_path)
    loaded = structures.load_model(model_path)
    assert loaded.temperature == pytest.approx(model.temperature)


def test_model_save_next_model(host_guest_mmvt_model, tmp_path):
    """save_next_model renames the existing model.json and writes a fresh one."""
    model = host_guest_mmvt_model
    model_path = tmp_path / "model.json"
    # Write the initial model.json that save_next_model will rename
    model.save(str(model_path))
    model.directory = str(tmp_path)
    model.save_next_model("model_*.json", "model_{0}.json")
    assert model_path.exists(), "Expected a new model.json after save_next_model"
    assert (tmp_path / "model_0.json").exists(), \
        "Expected the old model to be renamed to model_0.json"


def test_model_save_load_cvs(host_guest_mmvt_model, tmp_path):
    """save_cvs + load_cvs round-trip restores collective variables."""
    model = host_guest_mmvt_model
    original_count = len(model.collective_variables)
    assert original_count > 0, "Fixture must have at least one CV."
    model.directory = str(tmp_path)
    model.save_cvs(str(tmp_path), "main")
    model.collective_variables = []
    model.load_cvs("main")
    assert len(model.collective_variables) == original_count


def test_model_save_load_selections(host_guest_mmvt_model, tmp_path):
    """save_selections + load_selections round-trip restores engine selections."""
    model = host_guest_mmvt_model
    model.directory = str(tmp_path)
    model.save_selections(str(tmp_path), "main")
    # Record original counts then clear
    original_counts = {}
    for engine in model.engines:
        if hasattr(engine.unpartitioned_settings, "selections"):
            original_counts[engine.scale_type] = \
                len(engine.unpartitioned_settings.selections)
            engine.unpartitioned_settings.selections = []
    model.load_selections("main")
    for engine in model.engines:
        if engine.scale_type in original_counts:
            assert len(engine.unpartitioned_settings.selections) \
                == original_counts[engine.scale_type]


def test_model_save_load_anchor_set(host_guest_mmvt_model, tmp_path):
    """save_anchor_set + load_anchor_set round-trip restores anchors."""
    model = host_guest_mmvt_model
    original_count = len(model.anchors)
    assert original_count > 0, "Fixture must have at least one anchor."
    model.directory = str(tmp_path)
    model.save_anchor_set(str(tmp_path), "main")
    model.anchors = []
    model.num_anchors = 0
    model.num_milestones = 0
    model.load_anchor_set("main")
    assert len(model.anchors) == original_count
    assert model.num_anchors == original_count