from __future__ import annotations
"""
modules/scales/brownian_dynamics/runner.py

Perform runner functions specific to brownian dynamics stages.
"""

import os
import typing
import logging
import importlib
from collections import namedtuple

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.scales.structures as scales_structures
import seekr.modules.scales.brownian_dynamics.structures as structures_bd
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.runner as engines_runner

logger = logging.getLogger(__name__)

Run_instruction_bd = namedtuple(
    "Run_instruction_bd",
    [
        "trajs_to_go_minimum",
        "trajs_so_far",
        "anchor_index",
        "restart",
        "num_transitions",
        "message"
    ])

def get_calculation_directory_bd(
        bd_engine: engines_structures.Base_engine,
        root_directory: str,
        set_name: str):
    """
    Return the calculation directory for a given BD stage index.
    """
    full_directory = os.path.join(
        root_directory, "anchor_sets", set_name, 
        bd_engine.b_surface_directory)
    return full_directory

def is_finished_bd_with_message(
        model: structures.Seekr_model,
        stage_bd: structures_bd.Stage_bd,
        use_json: bool = True
    ) -> tuple[bool, str | dict]:
    """

    """
    message = None
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    bd_transition_counts = bd_engine.get_bd_transition_counts(
        model.directory, stage_bd.set_name)
    bulk_indices = model.get_bulk_indices()
    assert len(bulk_indices) > 0
    for bulk_index in bulk_indices:
        # Find the lower and upper milestone indices
        bulk_anchor = model.anchors[bulk_index]
        bulk_anchor_neighbor_index = bulk_anchor.neighbor_anchor_indices[0]
        bulk_anchor_neighbor = model.anchors[bulk_anchor_neighbor_index]
        assert len(bulk_anchor_neighbor.milestones) == 2, \
            "Bulk anchor neighbor must have exactly two milestones."
        lower_milestone_index = bulk_anchor_neighbor.milestones[0].index
        upper_milestone_index = bulk_anchor_neighbor.milestones[1].index
        bd_transition_counts = bd_engine.modify_bd_transition_counts(
            bd_transition_counts, lower_milestone_index, upper_milestone_index)
    
    if "b_surface" not in bd_transition_counts:
        num_trajs_so_far = 0
    else:
        num_trajs_so_far = bd_transition_counts["b_surface"]["total"]

    if stage_bd.completion_criteria.type == "number_of_trajectories":
        num_trajs_needed = stage_bd.completion_criteria.number_of_trajectories
        attained = num_trajs_so_far >= num_trajs_needed
        progress = num_trajs_so_far / num_trajs_needed \
            if num_trajs_needed > 0 else 1.0
        if use_json:
            message = {
                "completion_criteria": stage_bd.completion_criteria.type,
                "finished": attained,
                "progress": round(progress, 4),
                "trajectories_needed": num_trajs_needed,
                "trajectories_completed": num_trajs_so_far,
                "num_swarms": 1,
                "swarms": {
                    "0": {
                        "completion_criteria": stage_bd.completion_criteria.type,
                        "finished": attained,
                        "progress": round(progress, 4),
                        "trajectories_needed": num_trajs_needed,
                        "trajectories_completed": num_trajs_so_far
                    }
                }
            }
        else:
            message = f"Trajectories {num_trajs_so_far} of {num_trajs_needed} completed."
        return attained, message
    
    elif stage_bd.completion_criteria.type == "number_of_encounters":
        num_encounters_needed = stage_bd.completion_criteria.number_of_encounters
        progress = num_trajs_so_far / num_encounters_needed \
            if num_encounters_needed > 0 else 1.0
        if num_trajs_so_far == 0:
            min_b_surface_encounters = 0
        else:
            milestone_b_surface_counts = {}
            for key in bd_transition_counts["b_surface"]:
                if isinstance(key, int):
                    milestone_b_surface_counts[key] \
                        = bd_transition_counts["b_surface"][key]
            min_b_surface_encounters = min(
                milestone_b_surface_counts.values())
        attained = min_b_surface_encounters >= num_encounters_needed
        if use_json:
            message = {
                "completion_criteria": stage_bd.completion_criteria.type,
                "finished": attained,
                "progress": round(progress, 4),
                "encounters_needed": num_encounters_needed,
                "encounters_completed": min_b_surface_encounters,
                "trajectories_completed": num_trajs_so_far,
                "num_swarms": 1,
                "swarms": {
                    "0": {
                        "completion_criteria": stage_bd.completion_criteria.type,
                        "finished": attained,
                        "progress": round(progress, 4),
                        "encounters_needed": num_encounters_needed,
                        "encounters_completed": min_b_surface_encounters,
                        "trajectories_completed": num_trajs_so_far
                    }
                }
            }
        else:
            message = f"Encounters {min_b_surface_encounters} of {num_encounters_needed} completed."
        return attained, message

        # TODO: implement BD completion criteria in plugins
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install it or check the spelling.")
            # TODO: see if plugin.modules exists, and if not, raise an error
            #  Then if plugin.modules.completion_criteria doesn't exist, skip quietly
            #  Then if plugin.modules.scales.brownian_dynamics.runner doesn't exist, raise an error
            plugin_module_scales_bd_runner = importlib.import_module(plugin_name+".modules.scales.brownian_dynamics.runner")
            found, completed, message = plugin_module_scales_bd_runner.is_finished_bd_with_message(
                model, stage_bd, use_json)
            if found:
                return completed, message
    assert message is not None, \
        "Completion criteria not found: " + stage_bd.completion_criteria.type

def choose_next_simulation_bd(
        model: structures.Seekr_model,
        stage_bd: structures_bd.Stage_bd,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False,
        benchmark: bool = False
        ) -> list:
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    # Device_index is currently irrelevant because no BD engine
    #  has been implemented that uses a GPU.
    bulk_indices = model.get_bulk_indices()
    assert len(bulk_indices) > 0
    bulk_anchors = []
    for bulk_index in bulk_indices:
        bulk_anchors.append(model.anchors[bulk_index])

    if benchmark:
        if not force_overwrite:
            # then do nothing
            return []
        if hasattr(stage_bd, "completion_criteria"):
            stage_bd.completion_criteria.enable_benchmark_mode()

    if stage_bd.completion_criteria.type == "number_of_trajectories":
        num_trajs_needed = stage_bd.completion_criteria.number_of_trajectories
        min_b_surface_encounters = None
        bd_sim_chunk_size = num_trajs_needed
    elif stage_bd.completion_criteria.type == "number_of_encounters":
        num_trajs_needed = stage_bd.completion_criteria.evaluation_interval
        min_b_surface_encounters = stage_bd.completion_criteria.number_of_encounters
        bd_sim_chunk_size = stage_bd.completion_criteria.evaluation_interval
    else:
        raise NotImplementedError("Completion criteria not yet implemented: "\
                                    f"{stage_bd.completion_criteria.type}.")
        # NOTE: min_b_surface_encounters and bd_sim_chunk_size will need
        #  to be set based on the completion criteria.

    if force_overwrite:
        for bulk_anchor in bulk_anchors:
            engines_runner.cleanse_outputs_bd(
                bulk_anchor, model, stage_bd.set_name)

    run_instructions_unsorted: list[typing.NamedTuple] = []
    bd_transition_counts = bd_engine.get_bd_transition_counts(
        model.directory, stage_bd.set_name)
    
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
    
    completed, message = is_finished_bd_with_message(
        model, stage_bd, use_json=use_json)

    if "b_surface" not in bd_transition_counts:
        run_instruction = Run_instruction_bd(
            num_trajs_needed, 0, bulk_index, False, 0, message)
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
                total_b_surface_encounters, message)
            run_instructions_unsorted.append(run_instruction)
            
        elif min_b_surface_encounters is not None:
            if total_b_surface_encounters < min_b_surface_encounters:
                #total_bd_simulation_length = b_surface_sims_so_far + bd_sim_chunk_size
                total_bd_simulation_length = bd_sim_chunk_size
                run_instruction = Run_instruction_bd(
                    total_bd_simulation_length, b_surface_sims_so_far, 
                    bulk_index, True, total_b_surface_encounters, message)
                run_instructions_unsorted.append(run_instruction)
    
    return run_instructions_unsorted

def do_run_instruction_bd(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: Run_instruction_bd,
        benchmark: bool = False # Not used
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
    stage = model.stages[stage_index]
    calculation_dir = get_calculation_directory_bd(
        bd_engine, model.directory, stage.set_name)
    log_filename = os.path.join(calculation_dir, "run.log")
    base.configure_logging(log_filename, overwrite=force_overwrite)
    logger.info(log_string)
    engines_runner.run_simulation(
        model, stage_index, restart, n_trajectories=trajs_to_go_min, 
        force_overwrite=force_overwrite, benchmark=benchmark)
    if benchmark:
        # If in benchmark mode, clear all the files afterwards for the
        # True run.
        bulk_indices = model.get_bulk_indices()
        assert len(bulk_indices) > 0
        bulk_anchors = []
        for bulk_index in bulk_indices:
            bulk_anchors.append(model.anchors[bulk_index])
        for bulk_anchor in bulk_anchors:
            engines_runner.cleanse_outputs_bd(
                bulk_anchor, model, stage.set_name)