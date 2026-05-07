"""
test_modify_model.py

Tests for the modify_model() function in seekr/modules/preparer.py.
"""

import os
import pytest
import seekr.prepare as prepare
import seekr.tests.create_seekr_input as create_seekr_input
import seekr.modules.anchor_schemes.preparer as anchor_scheme_preparer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_prepare(seekr_input, force_overwrite=False):
    model, model_xml_path = prepare.prepare(
        seekr_input, force_overwrite=force_overwrite)
    return model, model_xml_path


def _first_prepare(tmp_dir):
    seekr_input = create_seekr_input.create_toy_input(str(tmp_dir))
    model, model_xml_path = _do_prepare(seekr_input)
    return seekr_input, model, model_xml_path


def _anchor_dir_exists(tmp_dir, anchor_dir_name):
    """Check if anchor directory exists at its correct on-disk location."""
    path = os.path.join(str(tmp_dir), "anchor_sets", "main", anchor_dir_name)
    return os.path.isdir(path)


def _plant_mmvt_file(tmp_dir, anchor_dir_name):
    """Create a fake MMVT output file so has_files() returns True for that anchor."""
    stage_dir = os.path.join(
        str(tmp_dir), "anchor_sets", "main", anchor_dir_name, "stage_4_MMVT")
    os.makedirs(stage_dir, exist_ok=True)
    fake_file = os.path.join(stage_dir, "mmvt.swarm_0.txt")
    open(fake_file, "w").close()
    return fake_file


def _plant_unpartitioned_file(tmp_dir, stage_dir_name, filename):
    """Create a fake output file in an unpartitioned stage directory."""
    stage_dir = os.path.join(str(tmp_dir), "unpartitioned_scope", stage_dir_name)
    os.makedirs(stage_dir, exist_ok=True)
    fake_file = os.path.join(stage_dir, filename)
    open(fake_file, "w").close()
    return fake_file


def _add_middle_anchor(seekr_input, value=-0.4):
    """Insert a new Voronoi cell at index 2 (between -0.5 and -0.3)."""
    new_vc = anchor_scheme_preparer.Voronoi_cell()
    new_vc.value = [value]
    new_vc.positions = [[[0.0, value, 0.0]]]
    new_vc.scale_types = ["molecular_dynamics"]
    seekr_input.partitions[0].anchor_scheme.voronoi_cells.insert(2, new_vc)


# ---------------------------------------------------------------------------
# Group A: Simple overwrites (no anchor/stage changes)
# ---------------------------------------------------------------------------

def test_identical_reprepare(tmp_path):
    """Re-preparing with identical input leaves anchor directories untouched."""
    seekr_input, model1, _ = _first_prepare(str(tmp_path))
    model2, _ = _do_prepare(seekr_input)
    assert len(model2.anchors) == len(model1.anchors)
    for i in range(len(model2.anchors)):
        assert model2.anchors[i].directory == model1.anchors[i].directory


def test_temperature_change_no_files(tmp_path):
    """Changing temperature with no run files succeeds without force_overwrite."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    seekr_input.temperature = 310.0
    model2, _ = _do_prepare(seekr_input)
    assert model2.temperature == pytest.approx(310.0)


# ---------------------------------------------------------------------------
# Group B: Stage rename
# ---------------------------------------------------------------------------

def test_stage_renamed_no_files(tmp_path):
    """Renaming a stage with no run files succeeds without force_overwrite."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    seekr_input.stages[0].name = "equilibration_renamed"
    model2, _ = _do_prepare(seekr_input)
    assert any(s.name == "equilibration_renamed" for s in model2.stages)


def test_stage_renamed_with_files_raises(tmp_path):
    """Renaming a stage with existing run files raises without force_overwrite."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _plant_unpartitioned_file(
        tmp_path, "stage_1_equilibration", "conventional.swarm_0.txt")
    seekr_input.stages[0].name = "equilibration_renamed"
    with pytest.raises(Exception, match="overwrite"):
        _do_prepare(seekr_input, force_overwrite=False)


def test_stage_renamed_with_files_force(tmp_path):
    """Renaming a stage with force_overwrite=True proceeds without raising."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _plant_unpartitioned_file(
        tmp_path, "stage_1_equilibration", "conventional.swarm_0.txt")
    seekr_input.stages[0].name = "equilibration_renamed"
    model2, _ = _do_prepare(seekr_input, force_overwrite=True)
    assert any(s.name == "equilibration_renamed" for s in model2.stages)


# ---------------------------------------------------------------------------
# Group C: Anchor deletion
# ---------------------------------------------------------------------------

def test_delete_anchor_no_files(tmp_path):
    """Deleting an anchor with no run files succeeds without force_overwrite."""
    seekr_input, model1, _ = _first_prepare(str(tmp_path))
    seekr_input.partitions[0].anchor_scheme.voronoi_cells.pop(1)
    model2, _ = _do_prepare(seekr_input)
    assert len(model2.anchors) == len(model1.anchors) - 1


def test_delete_anchor_with_files_raises(tmp_path):
    """Deleting an anchor with run files raises without force_overwrite."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _plant_mmvt_file(tmp_path, "anchor_0001")
    seekr_input.partitions[0].anchor_scheme.voronoi_cells.pop(1)
    with pytest.raises(Exception, match="overwrite"):
        _do_prepare(seekr_input, force_overwrite=False)


def test_delete_anchor_with_files_force(tmp_path):
    """Deleting an anchor with force_overwrite=True removes it and proceeds."""
    seekr_input, model1, _ = _first_prepare(str(tmp_path))
    fake_file = _plant_mmvt_file(tmp_path, "anchor_0001")
    seekr_input.partitions[0].anchor_scheme.voronoi_cells.pop(1)
    model2, _ = _do_prepare(seekr_input, force_overwrite=True)
    assert len(model2.anchors) == len(model1.anchors) - 1
    assert not os.path.exists(fake_file)


# ---------------------------------------------------------------------------
# Group D: Anchor insertion (existing directories get renamed)
# ---------------------------------------------------------------------------

def test_add_anchor_renames_existing_dirs(tmp_path):
    """Adding an anchor shifts existing directories; anchor at -0.3 moves to anchor_0003."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _add_middle_anchor(seekr_input, value=-0.4)
    model2, _ = _do_prepare(seekr_input)
    assert len(model2.anchors) == 9
    anchor_at_minus03 = next(
        a for a in model2.anchors
        if abs(a.variables.get("value_0", 99) - (-0.3)) < 1e-9)
    assert anchor_at_minus03.directory == "anchor_0003"
    assert _anchor_dir_exists(tmp_path, "anchor_0003")


# ---------------------------------------------------------------------------
# Group E: Milestone changes (inserting a neighbour alters adjacent milestones)
# ---------------------------------------------------------------------------

def test_changed_milestones_no_files(tmp_path):
    """Adding an anchor (changes neighbours' milestones) with no files succeeds."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _add_middle_anchor(seekr_input, value=-0.4)
    model2, _ = _do_prepare(seekr_input)
    assert len(model2.anchors) == 9


def test_changed_milestones_with_files_raises(tmp_path):
    """Changed milestone positions with run files raise without force_overwrite."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    _plant_mmvt_file(tmp_path, "anchor_0002")
    _add_middle_anchor(seekr_input, value=-0.4)
    with pytest.raises(Exception, match="overwrite"):
        _do_prepare(seekr_input, force_overwrite=False)


def test_changed_milestones_with_files_force(tmp_path):
    """force_overwrite=True removes stale files and accepts milestone changes."""
    seekr_input, _, _ = _first_prepare(str(tmp_path))
    fake_file = _plant_mmvt_file(tmp_path, "anchor_0002")
    _add_middle_anchor(seekr_input, value=-0.4)
    model2, _ = _do_prepare(seekr_input, force_overwrite=True)
    assert not os.path.exists(fake_file)
    assert len(model2.anchors) == 9
