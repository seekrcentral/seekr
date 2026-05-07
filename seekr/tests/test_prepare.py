"""
test_prepare.py

Smoke tests for seekr.prepare — verify that prepare.prepare() succeeds
without exception for every scenario in SCENARIO_CONFIGS.
"""

import os
import json

import pytest
import cattrs
from cattrs.gen import make_dict_unstructure_fn

import seekr.prepare as prepare
import seekr.modules.structures as structures
import seekr.modules.preparer as preparer
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.base as scales_base
import seekr.modules.collective_variables.base as cv_base
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.anchor_schemes.base as anchor_schemes_base
import seekr.modules.sampling_methods.preparer as sampling_preparer
import seekr.tests.create_seekr_input as create_seekr_input


def _save_seekr_input(seekr_input, path):
    """
    Write a Seekr_input to JSON, adding 'type' discriminators for all
    polymorphic fields so that preparer.read_input() can round-trip it.
    Uses make_dict_unstructure_fn to avoid recursion when the hook fires
    for a subclass that itself inherits the registered base type.
    """
    converter = cattrs.Converter()

    def unstruct_add_type(obj):
        d = make_dict_unstructure_fn(type(obj), converter)(obj)
        d["type"] = obj.type
        return d

    for base_type in (
        engines_structures.Base_engine_settings_input,
        scales_base.Base_input_stage,
        reporters_preparer.Base_reporter_input,
        restraints_preparer.Base_restraint_input,
        cv_base.Base_CV_input,
        anchor_schemes_base.Base_anchor_scheme,
        sampling_preparer.Base_md_sampling_method_input,
    ):
        converter.register_unstructure_hook(base_type, unstruct_add_type)

    with open(path, "w") as f:
        json.dump(converter.unstructure(seekr_input), f, indent=4)



def test_prepare_creates_model(seekr_prepared):
    """prepare() returns a valid Seekr_model and a model.json path."""
    model, model_xml_path, scenario_name = seekr_prepared
    assert isinstance(model, structures.Seekr_model), (
        f"[{scenario_name}] prepare() should return a Seekr_model, "
        f"got {type(model)}"
    )
    assert os.path.isfile(model_xml_path), (
        f"[{scenario_name}] model JSON not found at {model_xml_path}"
    )


def test_prepare_model_directory_exists(seekr_prepared):
    """The model directory created by prepare() exists on disk."""
    model, _, scenario_name = seekr_prepared
    assert os.path.isdir(model.directory), (
        f"[{scenario_name}] model.directory '{model.directory}' does not exist"
    )


def test_prepare_has_stages(seekr_prepared):
    """The prepared model contains at least one stage."""
    model, _, scenario_name = seekr_prepared
    assert len(model.stages) > 0, (
        f"[{scenario_name}] model.stages is empty"
    )


def test_prepare_has_anchors(seekr_prepared):
    """
    The prepared model has anchors, except for 'defineAnchorsLater' scenarios
    where anchors are intentionally added by a later logistic stage.
    """
    model, _, scenario_name = seekr_prepared
    if "defineAnchorsLater" in scenario_name or "define_anchors_later" in scenario_name:
        pass
    else:
        assert len(model.anchors) > 0, (
            f"[{scenario_name}] model.anchors is empty after prepare"
        )


def test_seekr_input_json_round_trip(tmp_path):
    """Seekr_input survives a JSON round-trip through read_input()."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path / "model"))
    json_path = str(tmp_path / "input.json")
    _save_seekr_input(seekr_input, json_path)

    loaded = preparer.read_input(json_path)

    assert isinstance(loaded, preparer.Seekr_input)
    assert loaded.temperature == pytest.approx(seekr_input.temperature)
    assert len(loaded.stages) == len(seekr_input.stages)
    assert len(loaded.partitions) == len(seekr_input.partitions)


def test_read_input_then_prepare(tmp_path):
    """JSON → read_input() → prepare() succeeds end-to-end."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path / "model"))
    json_path = str(tmp_path / "input.json")
    _save_seekr_input(seekr_input, json_path)

    loaded_input = preparer.read_input(json_path)
    old_dir = os.getcwd()
    try:
        model, model_path = prepare.prepare(loaded_input)
    finally:
        os.chdir(old_dir)

    assert isinstance(model, structures.Seekr_model)
    assert os.path.isfile(model_path)
