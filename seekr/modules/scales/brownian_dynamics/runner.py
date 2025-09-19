"""
modules/scales/brownian_dynamics/runner.py

Perform runner functions specific to brownian dynamics stages.
"""

import os
import typing
import logging
from collections import namedtuple

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.scales.structures as scales_structures
import seekr.modules.engines.runner as engines_runner

Run_instruction_bd = namedtuple(
    "Run_instruction_bd",
    [
        "trajs_to_go_minimum",
        "trajs_so_far",
        "anchor_index",
        "restart",
        "num_transitions"
    ])

def get_calculation_directory_bd(
        bd_engine: structures.BD_engine,
        root_directory: str):
    """
    Return the calculation directory for a given BD stage index.
    """
    full_directory = os.path.join(root_directory, bd_engine.b_surface_directory)
    return full_directory

def choose_next_simulation_bd(
        model: structures.Seekr_model,
        stage_bd: scales_structures.Stage,
        force_overwrite: bool,
        ) -> list:
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    # Device_index is currently irrelevant because no BD engine
    #  has been implemented that uses a GPU.
    bulk_indices = model.get_bulk_indices()
    assert len(bulk_indices) > 0
    bulk_anchors = []
    for bulk_index in bulk_indices:
        bulk_anchors.append(model.anchors[bulk_index])

    if stage_bd.completion_criteria.type == "number_of_trajectories":
        num_trajs_needed = stage_bd.completion_criteria.number_of_trajectories
        min_b_surface_encounters = None
        bd_sim_chunk_size = 1
    else:
        raise NotImplementedError("Completion criteria not yet implemented: "\
                                    f"{stage_bd.completion_criteria.type}.")
        # NOTE: min_b_surface_encounters and bd_sim_chunk_size will need
        #  to be set based on the completion criteria.

    if force_overwrite:
        for bulk_anchor in bulk_anchors:
            engines_runner.cleanse_outputs_bd(bulk_anchor, model)

    run_instructions_unsorted: list[typing.NamedTuple] = []
    bd_transition_counts = bd_engine.get_bd_transition_counts(model.directory)
    
    for bulk_anchor in bulk_anchors:
        # Find the lower and upper milestone indices
        bulk_anchor_neighbor_index = bulk_anchor.neighbor_anchor_indices[0]
        bulk_anchor_neighbor = model.anchors[bulk_anchor_neighbor_index]
        assert len(bulk_anchor_neighbor.milestones) == 2, \
            "Bulk anchor neighbor must have exactly two milestones."
        lower_milestone_index = bulk_anchor_neighbor.milestones[0].index
        upper_milestone_index = bulk_anchor_neighbor.milestones[1].index
        bd_transition_counts = bd_engine.modify_bd_transition_counts(
            bd_transition_counts, lower_milestone_index, upper_milestone_index)
        
    if "b_surface" not in bd_transition_counts:
        run_instruction = Run_instruction_bd(
            num_trajs_needed, 0, bulk_index, False, 0)
        run_instructions_unsorted.append(run_instruction)
    else:
        # see how many simulations need to be run in b_surface
        b_surface_sims_so_far = bd_transition_counts["b_surface"]["total"]
        milestone_b_surface_counts = {}
        for key in bd_transition_counts["b_surface"]:
            if isinstance(key, int):
                milestone_b_surface_counts[key] \
                    = bd_transition_counts["b_surface"][key]
        
        total_b_surface_encounters = min(
            milestone_b_surface_counts.values())
        steps_to_go_minimum = num_trajs_needed - b_surface_sims_so_far
        
        if steps_to_go_minimum > 0:
            run_instruction = Run_instruction_bd(
                steps_to_go_minimum, b_surface_sims_so_far, bulk_index, True, 
                total_b_surface_encounters)
            run_instructions_unsorted.append(run_instruction)
            
        elif min_b_surface_encounters is not None:
            if total_b_surface_encounters < min_b_surface_encounters:
                total_bd_simulation_length \
                    = (b_surface_sims_so_far // bd_sim_chunk_size + 1) \
                    * bd_sim_chunk_size
                run_instruction = Run_instruction_bd(
                    total_bd_simulation_length-b_surface_sims_so_far, b_surface_sims_so_far, 
                    bulk_index, True, total_b_surface_encounters)
                run_instructions_unsorted.append(run_instruction)
                
    return run_instructions_unsorted

def do_run_instruction_bd(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: Run_instruction_bd
    ) -> None:
    """
    Execute the run instructions for brownian dynamics.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    trajs_to_go_min = instruction.trajs_to_go_minimum
    trajs_so_far = instruction.trajs_so_far
    b_surface_anchor_index = instruction.anchor_index
    restart = instruction.restart
    num_transitions = instruction.num_transitions
    if force_overwrite and restart:
        restart = False
    log_string = f"running from b-surface anchor: {b_surface_anchor_index} "\
        +f"restart: {restart} trajectories to run: {trajs_to_go_min} "\
        +f"trajectories so far: {trajs_so_far} "\
        +f"number of transitions: {num_transitions}"
    calculation_dir = get_calculation_directory_bd(bd_engine, model.directory)
    log_filename = os.path.join(calculation_dir, "run.log")
    base.configure_file_and_stdout_logger(
        log_filename, overwrite=force_overwrite)
    logging.info(log_string)
    engines_runner.run_simulation(
        model, stage_index, restart, n_trajectories=trajs_to_go_min, 
        force_overwrite=force_overwrite)