"""
run_base.py

Base objects and functions for running SEEKR calculations.
"""

import os
import typing
import logging
from collections import namedtuple

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.structures as structures
import seekr.modules.engines.runner as engines_runner
import seekr.modules.completion_criteria.runner as completion_criteria_runner

MAX_RESTART_ATTEMPTS = 3

Run_instruction_md = namedtuple(
    "Run_instruction_md",
    [
        "current_step",
        "anchor_index",
        "restart",
        "swarm_index"
    ])

Run_instruction_bd = namedtuple(
    "Run_instruction_bd",
    [
        "trajs_to_go_minimum",
        "trajs_so_far",
        "anchor_index",
        "restart",
        "num_transitions"
    ])

def get_current_md_step(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_index: int | None,
        root_directory: str):
    """
    Return the current step for a given stage index.
    """
    assert len(anchor_or_systemwide.stage_parameters) > stage_index, "Stage index "\
        "provided is out of range."
    assert anchor_or_systemwide.stage_parameters[stage_index] is not None
    if isinstance(anchor_or_systemwide, structures.Anchor):
        if not anchor_or_systemwide.md:
            raise NotImplementedError("BD current step not yet implemented.")
    stage_directory = anchor_or_systemwide.stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, stage_directory)
    checkpoint_info_basename = engines_runner.get_checkpoint_information_basename(
        swarm_index)
    checkpoint_information_filename = os.path.join(
        full_directory, checkpoint_info_basename)
    if os.path.exists(checkpoint_information_filename):
        checkpoint_information = engines_runner.load_checkpoint_information(
            checkpoint_information_filename)
        current_step = checkpoint_information.current_step
    else: 
        current_step = 0
    return current_step

def get_calculation_directory_md(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int, 
        root_directory: str):
    """
    Return the calculation directory for a given MD stage index.
    """
    stage_list_index = stage_index + 1
    assert len(anchor_or_systemwide.stage_parameters) > stage_list_index\
        and stage_index >= 0, "Stage index provided is out of range."
    assert anchor_or_systemwide.stage_parameters[stage_list_index] is not None
    if isinstance(anchor_or_systemwide, structures.Anchor):
        if not anchor_or_systemwide.md:
            raise NotImplementedError("MD not running in this anchor/systemwide.")
    stage_directory = anchor_or_systemwide.stage_parameters[stage_list_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, stage_directory)
    return full_directory

def get_calculation_directory_bd(
        anchor: structures.Anchor,
        root_directory: str):
    """
    Return the calculation directory for a given BD stage index.
    """
    assert anchor.bd
    full_directory = os.path.join(root_directory, anchor.directory)
    return full_directory

def get_restart_checkpoint_filename(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_index: int | None,
        root_directory: str):
    """
    Return the restart checkpoint filename for a given stage index.
    """
    stage_directory = anchor_or_systemwide\
        .stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(
        root_directory, anchor_or_systemwide.directory, stage_directory)
    restart_checkpoint_basename = engines_runner\
        .get_restart_checkpoint_basename(swarm_index)
    restart_checkpoint_filename = os.path.join(
        full_directory, restart_checkpoint_basename)
    return restart_checkpoint_filename

def choose_next_simulation_md(
        stage_md: structures.Stage_md,
        model: structures.Seekr_model, 
        anchor_index: str, 
        force_overwrite: bool,
        device_index: str | None = None
        ) -> list:
    assert model.md_engine is not None, "MD stage provided, but "\
        "MD engine settings not provided."
    if device_index is not None:
        assert model.md_engine.platform_settings is not None
        model.md_engine.platform_settings.device_index = \
            device_index
    
    #run_instructions_unsorted: list = []
    anchor_or_systemwide_index_list: list[int | None] = []
    if stage_md.input_from == "systemwide":
        anchor_or_systemwide_index_list.append(None)
    elif stage_md.input_from == "anchors":
        for alpha, anchor in enumerate(model.anchors):
            if not anchor.md:
                continue
            if anchor_index not in ["any"]:
                try:
                    anchor_index_int = int(anchor_index)
                except ValueError:
                    return []
                if alpha != anchor_index_int:
                    continue
            anchor_or_systemwide_index_list.append(alpha)

    run_instructions_unsorted: list[typing.NamedTuple] = []
    anchor_or_systemwide: engines_structures.MD_systemwide_settings \
        | structures.Anchor
    for anchor_or_systemwide_index in anchor_or_systemwide_index_list:
        if anchor_or_systemwide_index is None:
            assert model.md_engine.systemwide_settings is not None
            anchor_or_systemwide = model.md_engine.systemwide_settings
            partition_name = "systemwide"
        else:
            anchor_or_systemwide = model.anchors[anchor_or_systemwide_index]
            partition_name = f"anchor {anchor_or_systemwide_index}"
        # TODO: remove support for multiple frames in this module - multiple-frame
        #  PDBs will be handled at the preparer level.
        assert anchor_or_systemwide is not None
        num_swarm_frames = anchor_or_systemwide.get_starting_structure_num_frames(
            model.directory, stage_md.input_stage_index)
        calculation_dir = get_calculation_directory_md(
            anchor_or_systemwide, stage_md.index-1, model.directory)
        log_filename = os.path.join(calculation_dir, "run.log")
        base.configure_file_and_stdout_logger(
            log_filename, overwrite=force_overwrite)
        for swarm_frame in range(num_swarm_frames):
            swarm_string = f" swarm {swarm_frame}"
            if num_swarm_frames <= 1:
                swarm_string = ""
            current_step = get_current_md_step(anchor_or_systemwide,
                stage_md.index, swarm_frame, model.directory)
            restart_checkpoint_filename \
                = get_restart_checkpoint_filename(
                    anchor_or_systemwide, stage_md.index, swarm_frame, model.directory)
            if (not os.path.exists(restart_checkpoint_filename)) \
                    or force_overwrite: # or umbrella_restart_mode:
                    # TODO: extract from sampling method whether restarts will be done
                restart = False
                current_step = 0
            else:
                restart = True
            
            if not force_overwrite:
                run_finished, message = completion_criteria_runner.is_finished_md_with_message(
                    model, stage_md, current_step)
            else:
                run_finished = False
                message = "Force overwrite."
            logging.info(f"{partition_name}{swarm_string}: {message}")
            if not run_finished:
                run_instruction = Run_instruction_md(
                    current_step, anchor_or_systemwide_index, restart, swarm_frame)
                run_instructions_unsorted.append(run_instruction)

    # sort the run instructions by current step to run the laggers first
    run_instructions = sorted(
        run_instructions_unsorted,
        key=lambda item: item.current_step)
    return run_instructions
    
def choose_next_simulation_bd(
        stage_bd: structures.Stage_bd, 
        model: structures.Seekr_model, 
        force_overwrite: bool,
        ) -> list:
    assert model.bd_engine is not None, "BD stage provided, but "\
        "BD engine settings not provided."
    # Device_index is currently irrelevant because no BD engine
    #  has been implemented that uses a GPU.
    bulk_index = model.get_bulk_index()
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
        assert bulk_index is not None
        bulk_anchor = model.anchors[bulk_index]
        engines_runner.cleanse_outputs_bd(bulk_anchor, model)

    run_instructions_unsorted: list[typing.NamedTuple] = []
    bd_transition_counts = model.bd_engine.get_bd_transition_counts(model.directory)
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