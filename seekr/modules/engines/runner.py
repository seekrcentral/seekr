"""
modules/engines/runner.py

Functions usable by any type of engine for running.
"""

import os
import time

import seekr.modules.structures as structures
import seekr.modules.engines.openmm.runner as runner_openmm
import seekr.modules.engines.browndye.runner as runner_browndye

# TODO: fill out types
def make_lock_string():
    """
    Make a string that can be put into the LOCK file to detect race
    conditions.
    """
    pid = os.getpid()
    timestamp = time.time()
    lock_string = "{},{:.3f}".format(pid, timestamp)
    return lock_string

# TODO: fill out types
def create_lock(lock_filename, verbose=False):
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
    return lock_filename, lock_string

# TODO: fill out types
def update_lock(lock_filename, lock_string):
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
        raise Exception("A lock file modification was detected for directory "\
                        f"{directory}. Another SEEKR process was "\
                        "running on the same anchor at the same time. Both "\
                        "processes will be killed. You must force this "\
                        "anchor to restart from the beginning to obtain "\
                        "correct results.")
    # otherwise all is well
    lock_string = make_lock_string()
    with open(lock_filename, "w") as f:
        f.write(lock_string)
    return lock_string

def run_md_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        anchor_index: int, 
        restart: bool, 
        total_simulation_length: int, 
        swarm_index: int | None = None, 
        device_index: str | None = None,
        force_overwrite: bool = False
        ) -> None:
    """
    Determine which md engine to run and run it
    """
    assert model.md_engine is not None
    found_engine = False
    if model.md_engine.type == "openmm":
        runner_openmm.run_simulation(
            model, stage_index, anchor_index, 
            restart, total_simulation_length, 
            swarm_index, device_index, 
            force_overwrite)
        found_engine = True
    else:
        # TODO: search plugins for other MD engines
        pass

    if not found_engine:
        raise Exception(f"MD engine not found: {model.md_engine.type}.")

    return
    
def run_bd_simulation(
        model: structures.Seekr_model, 
        n_trajectories: int, 
        restart: bool, 
        force_overwrite: bool = False
    ) -> None:
    """
    Determine which bd engine to run and run it
    """
    assert model.bd_engine is not None
    found_engine = False
    if model.bd_engine.type == "browndye":
        runner_browndye.run_simulation(
            model, n_trajectories, restart, force_overwrite)
        found_engine = True
    elif model.bd_engine.type == "sda":
        #TODO: fill out
        #engines_sda.run_simulation(
        #    model, stage_index, restart, force_overwrite)
        found_engine = True
    else:
        # TODO: search plugins for other MD engines
        pass

    if not found_engine:
        raise Exception(f"BD engine not found: {model.bd_engine.type}.")

def cleanse_outputs_bd(
        bulk_anchor, 
        model):
    """
    Remove BD simulation output files.
    """
    assert model.bd_engine is not None
    found_engine = False
    if model.bd_engine.type == "browndye":
        directory = os.path.join(model.directory, bulk_anchor.directory)
        runner_browndye.cleanse_bd_outputs(directory, model.bd_engine.results_glob)
        found_engine = True
    elif model.bd_engine.type == "sda":
        #TODO: fill out
        #engines_sda.run_simulation(
        #    model, stage_index, restart, force_overwrite)
        found_engine = True
    else:
        # TODO: search plugins for other MD engines
        pass

    if not found_engine:
        raise Exception(f"BD engine not found: {model.bd_engine.type}.")