from __future__ import annotations
"""
test_base.py

Unit tests for seekr/modules/engines/base.py (lock-file mechanism).
"""

import os
import pytest

import seekr.modules.engines.base as engines_base


# ---------------------------------------------------------------------------
# make_lock_string
# ---------------------------------------------------------------------------

def test_make_lock_string_format():
    """Lock string is '<pid>,<timestamp>' with parseable numeric parts."""
    s = engines_base.make_lock_string()
    parts = s.split(",")
    assert len(parts) == 2
    assert int(parts[0]) == os.getpid()
    float(parts[1])  # raises ValueError if not a valid float


# ---------------------------------------------------------------------------
# create_lock
# ---------------------------------------------------------------------------

def test_create_lock_writes_file(tmp_path):
    """create_lock creates the lock file on disk."""
    lock_path = str(tmp_path / "LOCK")
    engines_base.create_lock(lock_path)
    assert os.path.exists(lock_path)


def test_create_lock_returns_lock_string(tmp_path):
    """Returned string matches the file contents."""
    lock_path = str(tmp_path / "LOCK")
    lock_string = engines_base.create_lock(lock_path)
    with open(lock_path) as f:
        on_disk = f.read()
    assert lock_string == on_disk


def test_create_lock_overwrites_existing(tmp_path, capsys):
    """create_lock with verbose=True warns but still overwrites a stale lock."""
    lock_path = str(tmp_path / "LOCK")
    with open(lock_path, "w") as f:
        f.write("stale,0.000")
    lock_string = engines_base.create_lock(lock_path, verbose=True)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    with open(lock_path) as f:
        on_disk = f.read()
    assert on_disk == lock_string
    assert on_disk != "stale,0.000"


# ---------------------------------------------------------------------------
# update_lock
# ---------------------------------------------------------------------------

def test_update_lock_updates_content(tmp_path):
    """update_lock writes a new lock string to disk and returns it."""
    lock_path = str(tmp_path / "LOCK")
    original = engines_base.create_lock(lock_path)
    new_string = engines_base.update_lock(lock_path, original)
    with open(lock_path) as f:
        on_disk = f.read()
    assert on_disk == new_string
    # new_string is a valid "<pid>,<timestamp>" format
    parts = new_string.split(",")
    assert len(parts) == 2
    float(parts[1])


def test_update_lock_raises_if_file_missing(tmp_path):
    """update_lock raises if the lock file has been deleted."""
    lock_path = str(tmp_path / "LOCK")
    with pytest.raises(Exception, match="lock file was deleted"):
        engines_base.update_lock(lock_path, "12345,1.000")


def test_update_lock_raises_on_race_condition(tmp_path):
    """update_lock raises and writes ABORT when another process modified the file."""
    lock_path = str(tmp_path / "LOCK")
    original = engines_base.create_lock(lock_path)
    # Simulate a second process overwriting the lock
    with open(lock_path, "w") as f:
        f.write("99999,9999.999")
    with pytest.raises(Exception, match="lock file modification"):
        engines_base.update_lock(lock_path, original)
    with open(lock_path) as f:
        content = f.read()
    assert content == "ABORT"
