"""
test_prepare.py

Smoke tests for seekr.prepare — verify that prepare.prepare() succeeds
without exception for every scenario in SCENARIO_CONFIGS.
"""

import os

import seekr.modules.structures as structures


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
