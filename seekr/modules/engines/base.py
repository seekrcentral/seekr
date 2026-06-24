from __future__ import annotations
"""
modules/engines/base.py

Base functions common to all engines. Must be placed here to avoid
circular imports.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)

INFORMATION_CHECKPOINT_FILENAME_NOEXT = "checkpoint_information"
INFORMATION_CHECKPOINT_FILENAME_EXT = ".json"
RESTART_CHECKPOINT_FILENAME_NOEXT = "backup"
RESTART_CHECKPOINT_FILENAME_EXT = ".checkpoint"

def make_lock_string() -> str:
    """
    Make a string that can be put into the LOCK file to detect race
    conditions.
    """
    pid = os.getpid()
    timestamp = time.time()
    lock_string = "{},{:.3f}".format(pid, timestamp)
    return lock_string

def create_lock(
        lock_filename: str, 
        verbose: bool = False
        ) -> str:
    """
    Create a lock file to prevent running if a SEEKR anchor is currently being
    run by another process.
    """
    directory = os.path.dirname(lock_filename)
    if os.path.exists(lock_filename) and verbose:
        print("Warning: a lock file was detected in directory "\
                f"{directory}. This may happen because the a previous "\
                "process failed suddenly, or because a process is currently "\
                "running on this anchor.")
    lock_string = make_lock_string()
    with open(lock_filename, "w") as f:
        f.write(lock_string)
    return lock_string

def update_lock(
        lock_filename: str, 
        lock_string: str
        ) -> str:
    """
    Check the lock file for changes, and update the lock file.
    """
    directory = os.path.dirname(lock_filename)
    if not os.path.exists(lock_filename):
        raise Exception(f"The lock file was deleted for directory {directory}.")
    with open(lock_filename, "r") as f:
        old_lock_string = f.readline()
    if old_lock_string != lock_string:
        with open(lock_filename, "w") as f:
            # kill the other process
            f.write("ABORT")
        abort_str = "A lock file modification was detected for directory "\
                        f"{directory}. Another SEEKR process was "\
                        "running on the same anchor at the same time. Both "\
                        "processes will be killed. You must force this "\
                        "anchor to restart from the beginning to obtain "\
                        "correct results."
        logger.critical(abort_str)
        raise Exception(abort_str)
    # otherwise all is well
    lock_string = make_lock_string()
    with open(lock_filename, "w") as f:
        f.write(lock_string)
    return lock_string