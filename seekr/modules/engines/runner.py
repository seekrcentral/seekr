"""
modules/engines/runner.py

Functions usable by any type of engine for running.
"""

import os
import time

import seekr.modules.structures as structures
import seekr.modules.engines.openmm.runner as runner_openmm
import seekr.modules.engines.browndye.runner as runner_browndye

INFORMATION_CHECKPOINT_FILENAME_NOEXT = "checkpoint_information"
INFORMATION_CHECKPOINT_FILENAME_EXT = ".json"
RESTART_CHECKPOINT_FILENAME_NOEXT = "backup"
RESTART_CHECKPOINT_FILENAME_EXT = ".checkpoint"

def run_md_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        anchor_index: int, 
        restart: bool, 
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
            restart, swarm_index, device_index, 
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
    
def get_checkpoint_information_basename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
                + INFORMATION_CHECKPOINT_FILENAME_EXT
    else:
        checkpoint_information_filename \
            = INFORMATION_CHECKPOINT_FILENAME_NOEXT \
            + f".swarm_{swarm_index}" \
            + INFORMATION_CHECKPOINT_FILENAME_EXT
    return checkpoint_information_filename

def get_restart_checkpoint_basename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + RESTART_CHECKPOINT_FILENAME_EXT
    else:
        restart_checkpoint_filename = RESTART_CHECKPOINT_FILENAME_NOEXT\
            + str(swarm_index) + RESTART_CHECKPOINT_FILENAME_EXT
    return restart_checkpoint_filename
