"""
modules/scales/runner.py

Differentiate between different scales and call the appropriate
runner functions for each scale.
"""

import os
import typing

import seekr.modules.structures as structures
import seekr.modules.scales.structures as scales_structures
import seekr.modules.scales.molecular_dynamics.runner as md_runner
import seekr.modules.scales.brownian_dynamics.runner as bd_runner

def choose_next_simulation(
        model: structures.Seekr_model,
        stage: scales_structures.Stage,
        choose_params: dict,
        force_overwrite: bool,
        ) -> list:
    """
    Choose the next simulation to run based on the stage scale, and return
    a list of instructions to run in this round.
    """
    if stage.scale_type == "molecular_dynamics":
        anchor_index_instruction = choose_params["anchor_index_instruction"]
        device_index = choose_params["device_index"]
        run_instructions = md_runner.choose_next_simulation_md(
            model, stage, anchor_index_instruction, force_overwrite, device_index)
    elif stage.scale_type == "brownian_dynamics":
        run_instructions = bd_runner.choose_next_simulation_bd(model, stage, force_overwrite)
    else:
        raise Exception(f"Stage type not recognized: {stage.type}.")
    return run_instructions
    
def get_calculation_directory(
        engine: structures.Engine,
        root_directory: str):
    """
    Return the calculation directory for a given BD stage index.
    """
    if engine.scale_type == "brownian_dynamics":
        full_directory = bd_runner.get_calculation_directory_bd(
            engine, root_directory)
    elif engine.scale_type == "molecular_dynamics":
        full_directory = md_runner.get_calculation_directory_md(
            engine, root_directory)
    return full_directory
    
def do_run_instruction(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: list,
        device_index: str | None
    ) -> None:
    """
    Execute the run instructions for a given scale type.
    """
    stage = model.stages[stage_index]
    if stage.scale_type == "molecular_dynamics":
        md_runner.do_run_instruction_md(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction, device_index=device_index)
    elif stage.scale_type == "brownian_dynamics":
        bd_runner.do_run_instruction_bd(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction)
    return