from __future__ import annotations
"""
test_status.py

Smoke tests for seekr.status — verify that status() returns without exception
after all non-BD stages have been run.
"""

import seekr.status as status


def test_status_progress(seekr_run):
    """status() with instruction='progress' returns without exception."""
    model, _, scenario_name = seekr_run
    result = status.status(model, "progress", print_json=True)
    assert result is not None, (
        f"[{scenario_name}] status('progress') returned None"
    )
    progress = result["progress"]
    assert isinstance(progress, dict)
    for stage_id, stage_blob in progress.items():
        assert isinstance(stage_blob["finished"], bool), stage_id
        anchors = stage_blob["anchors"]
        assert isinstance(anchors, dict), stage_id
        assert "attained" not in stage_blob
        for anchor_id, anchor_blob in anchors.items():
            assert isinstance(anchor_blob["finished"], bool), (stage_id, anchor_id)
            swarms = anchor_blob["swarms"]
            assert isinstance(swarms, dict)
            assert "attained" not in anchor_blob
            for swarm_id, swarm_blob in swarms.items():
                assert isinstance(swarm_id, str), (stage_id, anchor_id, swarm_id)
                assert swarm_id.isdigit()
                assert isinstance(swarm_blob["finished"], bool)
                assert "attained" not in swarm_blob


def test_status_statistics(seekr_run):
    """status() with instruction='statistics' returns without exception."""
    model, _, scenario_name = seekr_run
    result = status.status(model, "statistics", print_json=True)
    assert result is not None, (
        f"[{scenario_name}] status('statistics') returned None"
    )

def test_status_statistics_no_json(seekr_run):
    """status() with instruction='statistics' returns without exception."""
    model, _, scenario_name = seekr_run
    result = status.status(model, "statistics", print_json=False)
    assert result is not None, (
        f"[{scenario_name}] status('statistics') returned None"
    )


def test_status_all(seekr_run):
    """status() with instruction='all' returns without exception."""
    model, _, scenario_name = seekr_run
    result = status.status(model, "all", print_json=True)
    assert result is not None, (
        f"[{scenario_name}] status('all') returned None"
    )
