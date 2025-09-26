"""
modules/engines/runner.py

Functions usable by any type of engine for running.
"""

import os
import json

import cattrs

import seekr.modules.engines.base as engines_base
import seekr.modules.engines.sda.base as sda_base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.browndye.runner as runner_browndye
import seekr.modules.engines.sda.runner as runner_sda

# NOTE: want this module to only be imported selectively

def load_checkpoint_information(
        filename: str
        ) -> engines_structures.Checkpoint_information:
    """
    Load a model object from a file.
    """
    with open(filename, "r") as file:
        json_string: str = json.load(file)
    
    # Use a cattrs Converter to define how to structure the dictionary input
    converter: cattrs.Converter = cattrs.Converter()
    checkpoint_information: engines_structures.Checkpoint_information \
        = converter.structure(json_string, engines_structures.Checkpoint_information)
    return checkpoint_information

def run_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        restart: bool, 
        anchor_index: int | None = None, 
        swarm_index: int | None = None, 
        device_index: str | None = None,
        n_trajectories: int = 0,
        force_overwrite: bool = False
        ) -> None:
    """
    Determine which md engine to run and run it
    """
    stage = model.stages[stage_index]
    engine = model.get_engine_by_scale_type(stage.scale_type)
    if engine.type == "openmm":
        import seekr.modules.engines.openmm.runner as runner_openmm
        runner_openmm.run_simulation(
            model, stage_index, anchor_index, 
            restart, swarm_index, device_index, 
            force_overwrite)
    elif engine.type == "browndye":
        runner_browndye.run_simulation(
            model, n_trajectories, restart, force_overwrite)
    elif engine.type == "sda":
        runner_sda.run_simulation(
            model, n_trajectories, restart, force_overwrite)
    else:
        # TODO: search plugins for other MD engines
        pass

    return
    
def cleanse_outputs_bd(
        bulk_anchor, 
        model):
    """
    Remove BD simulation output files.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    found_engine = False
    if bd_engine.type == "browndye":
        directory = os.path.join(model.directory, bulk_anchor.directory)
        runner_browndye.cleanse_bd_outputs(directory, bd_engine.results_glob, 
                                           check_mode=False)
        found_engine = True
    elif bd_engine.type == "sda":
        directory = os.path.join(model.directory, bulk_anchor.directory)
        runner_sda.cleanse_bd_outputs(directory, sda_base.SDA_RESULTS_GLOB, 
                                      check_mode=False)
        found_engine = True
    else:
        # TODO: search plugins for other MD engines
        pass

    if not found_engine:
        raise Exception(f"BD engine not found: {bd_engine.type}.")
    
def get_checkpoint_information_basename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        swarm_index = 0
    checkpoint_information_filename \
            = engines_base.INFORMATION_CHECKPOINT_FILENAME_NOEXT \
            + f".swarm_{swarm_index}" \
            + engines_base.INFORMATION_CHECKPOINT_FILENAME_EXT
    return checkpoint_information_filename

def get_restart_checkpoint_basename(
        swarm_index: int | None
        ) -> str:
    if swarm_index is None:
        swarm_index = 0
    restart_checkpoint_filename = engines_base.RESTART_CHECKPOINT_FILENAME_NOEXT\
            + str(swarm_index) + engines_base.RESTART_CHECKPOINT_FILENAME_EXT
    return restart_checkpoint_filename
