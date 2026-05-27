from __future__ import annotations
"""
test_check.py

Smoke tests for seekr.modules.check — verify that check_pre_simulation_all()
and check_post_simulation_all() complete without exception for every scenario.
"""

import pytest

import seekr.modules.check as check


def test_check_pre_simulation_all(seekr_prepared):
    """check_pre_simulation_all() must not raise for any prepared scenario."""
    model, _, scenario_name = seekr_prepared
    try:
        check.check_pre_simulation_all(model)
    except Exception as exc:
        pytest.fail(
            f"[{scenario_name}] check_pre_simulation_all() raised: {exc}"
        )

def test_check_post_simulation_all(seekr_run):
    """check_post_simulation_all() must not raise after all stages have run."""
    model, _, scenario_name = seekr_run
    try:
        check.check_post_simulation_all(model)
    except Exception as exc:
        pytest.fail(
            f"[{scenario_name}] check_post_simulation_all() raised: {exc}"
        )
