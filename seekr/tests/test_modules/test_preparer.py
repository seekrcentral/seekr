"""
test_preparer.py

Test the modules/preparer.py module.
"""

import os
import glob
import json

import cattrs
import pytest

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.preparer as preparer
import seekr.prepare as prepare
import seekr.tests.create_seekr_input as create_seekr_input

TEST_MODULE_DIRECTORY = os.path.dirname(__file__)

def test_assign_all_state_points_hostguest(host_guest_mmvt_model):
    bound_state = base.State_point()
    bound_state.name = "bound_test"
    bound_state.value = [0.06]
    bulk_state = base.State_point()
    bulk_state.name = "bulk_test"
    bulk_state.value = [1.36]
    preparer.assign_all_state_points([bound_state, bulk_state], 
                                          host_guest_mmvt_model)
    assert host_guest_mmvt_model.anchors[0].name == "bound_test"
    assert host_guest_mmvt_model.anchors[13].name == "bulk_test"
    return


def test_read_input(tmp_path):
    """read_input() should deserialize a JSON file back into a Seekr_input."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path))
    converter = cattrs.Converter()
    json_dict = converter.unstructure(seekr_input)
    input_path = str(tmp_path / "test_input.json")
    with open(input_path, "w") as f:
        json.dump(json_dict, f)
    result = preparer.read_input(input_path)
    assert isinstance(result, preparer.Seekr_input)
    assert result.temperature == pytest.approx(298.15)
    assert result.random_seed == 111
    assert result.plugins == []
    assert len(result.stages) == 4


def test_cleanse_outputs_stage(seekr_run):
    """cleanse_outputs_stage() deletes output files in the target stage dir."""
    model, _, _ = seekr_run
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    unpart = md_engine.unpartitioned_settings
    stage = next(
        (s for s in model.stages
         if getattr(s, "scope", None) == "unpartitioned" and s.type == "md_stage"),
        None,
    )
    if stage is None:
        pytest.skip("no unpartitioned md stage in this scenario")
    full_dir = os.path.join(
        model.directory, unpart.directory, stage.get_directory_name())
    os.makedirs(full_dir, exist_ok=True)
    dummy = os.path.join(full_dir, "dummy_swarm_0.dcd")
    open(dummy, "w").close()
    assert os.path.exists(dummy)
    preparer.cleanse_outputs_stage(model, unpart, stage.index - 1, 0)
    assert not os.path.exists(dummy)


def test_cleanse_outputs_all(seekr_run):
    """cleanse_outputs_all() deletes outputs across every stage."""
    model, _, _ = seekr_run
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    unpart = md_engine.unpartitioned_settings
    unpart_stages = [
        s for s in model.stages
        if getattr(s, "scope", None) == "unpartitioned" and s.type == "md_stage"
    ]
    if not unpart_stages:
        pytest.skip("no unpartitioned md stages in this scenario")
    dummies = []
    for stage in unpart_stages:
        full_dir = os.path.join(
            model.directory, unpart.directory, stage.get_directory_name())
        os.makedirs(full_dir, exist_ok=True)
        dummy = os.path.join(full_dir, "dummy_swarm_0.dcd")
        open(dummy, "w").close()
        dummies.append(dummy)
    preparer.cleanse_outputs_all(model, unpart, swarm_id=0)
    for dummy in dummies:
        assert not os.path.exists(dummy)

""" # TODO: update for latest code version
def test_move_add_delete_input_anchors(tmp_path, host_guest_mmvt_model_input):
    ""
    Test prepare_base.py's ability to move, add, and delete anchors
    in the model.
    ""
    tmp_bd_settings = host_guest_mmvt_model_input.bd_settings
    host_guest_mmvt_model_input.bd_settings = None
    host_guest_mmvt_model_input.directory = os.path.join(
        tmp_path, "host_guest_mmvt_2")
    host_guest_mmvt_model_input.partitions[0].anchors[0].value = [0.06]
    os.chdir(TEST_MODULE_DIRECTORY)
    model2, model2_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input, force_overwrite=True)
    model_dir = os.path.dirname(model2_xml_path)
    model2.directory = os.path.abspath(model_dir)
    assert model2.anchors[0].variables["value_0"] == 0.06
    assert model2.anchors[0].milestones[0].variables["me_0"] == 0.06
    assert model2.anchors[0].milestones[0].variables["neighbor_0"] == 0.15
    assert model2.anchors[0].milestones[0].variables["avg_value"] == 0.105
    assert len(model2.anchors) == 14
    assert model2.num_anchors == 14
    assert len(glob.glob(os.path.join(model2.directory, "anchor_*"))) == 13
    
    # Add anchor
    new_input_anchor = data_structures.Anchor_input()
    new_input_anchor.value = [0.1]
    host_guest_mmvt_model_input.partitions[0].anchors.insert(
        1, new_input_anchor)
    host_guest_mmvt_model_input.directory = os.path.join(
        tmp_path, "host_guest_mmvt_3")
    os.chdir(TEST_MODULE_DIRECTORY)
    model3, model3_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input, force_overwrite=False)
    model_dir = os.path.dirname(model3_xml_path)
    model3.directory = os.path.abspath(model_dir)
    assert model3.anchors[1].variables["value_0"] == 0.1
    assert model3.anchors[1].milestones[0].variables["me_0"] == 0.1
    assert model3.anchors[1].milestones[0].variables["neighbor_0"] == 0.06
    assert model3.anchors[1].milestones[0].variables["avg_value"] == 0.08
    assert model3.anchors[1].milestones[1].variables["me_0"] == 0.1
    assert model3.anchors[1].milestones[1].variables["neighbor_0"] == 0.15
    assert model3.anchors[1].milestones[1].variables["avg_value"] == 0.125
    assert len(model3.anchors) == 15
    assert model3.num_anchors == 15
    assert len(glob.glob(os.path.join(model3.directory, "anchor_*"))) == 14
    
    # Delete anchor
    host_guest_mmvt_model_input.partitions[0].anchors.pop(1)
    host_guest_mmvt_model_input.bd_settings = tmp_bd_settings
    host_guest_mmvt_model_input.directory = os.path.join(
        tmp_path, "host_guest_mmvt_4")
    os.chdir(TEST_MODULE_DIRECTORY)
    model4, model4_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input, force_overwrite=False)
    model_dir = os.path.dirname(model4_xml_path)
    model4.directory = os.path.abspath(model_dir)
    assert model2.anchors[0].milestones[0].variables["me_0"] == 0.06
    assert model2.anchors[0].milestones[0].variables["neighbor_0"] == 0.15
    assert model4.anchors[0].milestones[0].variables['avg_value'] == 0.105
    assert model2.anchors[1].milestones[0].variables["me_0"] == 0.15
    assert model2.anchors[1].milestones[0].variables["neighbor_0"] == 0.06
    assert model4.anchors[1].milestones[0].variables['avg_value'] == 0.105
    assert model2.anchors[1].milestones[1].variables["me_0"] == 0.15
    assert model2.anchors[1].milestones[1].variables["neighbor_0"] == 0.25
    assert model4.anchors[1].milestones[1].variables['avg_value'] == 0.2
    assert len(model4.anchors) == 14
    assert model4.num_anchors == 14
    assert len(glob.glob(os.path.join(model4.directory, "anchor_*"))) == 13
    
    # adjust BD milestone location
    host_guest_mmvt_model_input.partitions[0].anchors[-2].value = [1.75]
    host_guest_mmvt_model_input.partitions[0].anchors[-1].value = [1.95]
    host_guest_mmvt_model_input.directory = os.path.join(
        tmp_path, "host_guest_mmvt_5")
    os.chdir(TEST_MODULE_DIRECTORY)
    model5, model5_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input, force_overwrite=False)
    model_dir = os.path.dirname(model5_xml_path)
    model5.directory = os.path.abspath(model_dir)
    assert model5.anchors[-1].variables["value_0"] == 1.95
    assert model5.anchors[-2].variables["value_0"] == 1.75
    
    # delete BD calculation
    host_guest_mmvt_model_input.bd_settings = None
    host_guest_mmvt_model_input.directory = os.path.join(
        tmp_path, "host_guest_mmvt_6")
    os.chdir(TEST_MODULE_DIRECTORY)
    model6, model6_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input, force_overwrite=True)
    model_dir = os.path.dirname(model6_xml_path)
    model6.directory = os.path.abspath(model_dir)
    
    return
"""

# TODO: copy over test_modify_model from seekr2's test_common_prepare.py
#  once the check module has been adequately build.