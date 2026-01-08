"""
modules/scales/molecular_dynamics/runner.py

Perform runner functions specific to molecular dynamics stages.
"""

import os
import glob
import typing
import logging
import importlib
from collections import namedtuple

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.completion_criteria.structures as completion_criteria_structures
import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.engines.runner as engines_runner

Run_instruction_md = namedtuple(
    "Run_instruction_md",
    [
        "current_step",
        "anchor_index",
        "restart",
        "swarm_id",
        "message"
    ])

def filled_and_unfilled_anchors(
        model: structures.Seekr_model,
        stage: md_structures.Stage_md
    ) -> tuple[list[int], list[int]]:
    """
    Return two lists: one of the indices of anchors that have been filled
    and one of the indices of anchors that have not been filled.
    """
    filled_anchors = []
    unfilled_anchors = []
    stage_index = stage.index
    stage_output_globs: list[str] = stage.sampling_method.get_output_positions_glob_list()
    for anchor in model.anchors:
        if "molecular_dynamics" not in anchor.scale_types:
            continue
        found_outputs_for_anchor = False
        stage_parameter = anchor.stage_parameters[stage_index]
        assert stage_parameter is not None, \
            f"Stage parameter for stage {stage_index} is None."
        assert "directory" in stage_parameter, \
            f"Stage parameter for stage {stage_index} does not contain "\
            +"a directory."
        stage_directory = stage_parameter["directory"]
        for stage_output_glob in stage_output_globs:
            output_files_glob = os.path.join(
                model.directory, anchor.directory, stage_directory, 
                stage_output_glob)
            if len(glob.glob(output_files_glob)) > 0:
                found_outputs_for_anchor = True
        
        if found_outputs_for_anchor:
            filled_anchors.append(anchor.index)
        else:
            unfilled_anchors.append(anchor.index)
    
    return filled_anchors, unfilled_anchors


def is_finished_md(
        model: structures.Seekr_model,
        stage: md_structures.Stage_md,
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor,
        current_step: int,
        swarm_id: int = 0,
    ) -> bool:
    """
    Check if the simulation is finished.
    """
    ending, _ = is_finished_md_with_message(model, stage, anchor_or_systemwide, current_step)
    return ending

def is_finished_md_with_message(
        model: structures.Seekr_model,
        stage: md_structures.Stage_md,
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor,
        current_step: int,
        swarm_id: int = 0,
        use_json: bool = True
    ) -> tuple[bool, typing.Union[None, str, dict]]:
    """
    Check if the simulation is finished and return with message.
    """
    completion_criteria: completion_criteria_structures.MD_completion_criteria = \
        stage.completion_criteria
    message = None
    if completion_criteria.type == "number_of_steps":
        if use_json:
            # Use JSON-style message
            message = {
                "completion_criteria": completion_criteria.type,
                "attained": current_step >= completion_criteria.number_of_steps,
                "current_step": current_step,
                "total_steps": completion_criteria.number_of_steps
            }
        else:
            message = f"steps {current_step} of {completion_criteria.number_of_steps} "\
                "completed."
        return current_step >= completion_criteria.number_of_steps, message

    elif completion_criteria.type == "anchors_filled":
        filled_anchor_indices, unfilled_anchor_indices = filled_and_unfilled_anchors(
            model, stage)
        total_number_of_anchors = len(filled_anchor_indices) + \
                len(unfilled_anchor_indices)
        unfilled_only_contains_excluded = True
        for anchor_index in unfilled_anchor_indices:
            if anchor_index not in completion_criteria.exclude:
                unfilled_only_contains_excluded = False
                break
        if unfilled_only_contains_excluded:
            if use_json:
                message = {
                    "completion_criteria": completion_criteria.type,
                    "attained": True,
                    "num_filled_anchors": len(filled_anchor_indices),
                    "total_number_of_anchors": total_number_of_anchors,
                    "current_step": current_step,
                    "maximum_steps": completion_criteria.get_max_sim_len()
                }
            else:
                message = "anchors filled."
            return True, message
            
        elif current_step >= completion_criteria.get_max_sim_len():
            if use_json:
                message = {
                    "completion_criteria": completion_criteria.type,
                    "attained": False,
                    "num_filled_anchors": len(filled_anchor_indices),
                    "total_number_of_anchors": total_number_of_anchors,
                    "current_step": current_step,
                    "maximum_steps": completion_criteria.get_max_sim_len()
                }
            else:
                message = f"Maximum steps exceeded: {current_step} >= "\
                    f"{completion_criteria.get_max_sim_len()}."
            return True, message
        else:
            if use_json:
                message = {
                    "completion_criteria": completion_criteria.type,
                    "attained": False,
                    "num_filled_anchors": len(filled_anchor_indices),
                    "total_number_of_anchors": total_number_of_anchors,
                    "current_step": current_step,
                    "maximum_steps": completion_criteria.get_max_sim_len()
                }
            else:
                message = f"{len(filled_anchor_indices)} of "\
                    f"{total_number_of_anchors} anchors filled."
            return False, message
    
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            spec = importlib.util.find_spec(plugin_name)
            if spec is None:
                raise ImportError(f"Plugin '{plugin_name}' not found. Please install "\
                                  "it or check the spelling.")
            # TODO: see if plugin.modules exists, and if not, raise an error
            #  Then if plugin.modules.completion_criteria doesn't exist, skip quietly
            #  Then if plugin.modules.scales.molecular_dynamics.runner doesn't exist,
            #   raise an error
            plugin_module_scales_md_runner = importlib.import_module(
                plugin_name+".modules.scales.molecular_dynamics.runner")
            found, completed, message = plugin_module_scales_md_runner\
                .is_finished_md_with_message(
                    model, stage, anchor_or_systemwide, current_step, swarm_id)
            if found:
                return completed, message

    assert message is not None, \
        "Completion criteria not found: " + completion_criteria.type

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
        if "molecular_dynamics" not in anchor_or_systemwide.scale_types:
            raise NotImplementedError("MD not running in this anchor/systemwide.")
    stage_directory = anchor_or_systemwide.stage_parameters[stage_list_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, 
                                  stage_directory)
    return full_directory

def get_current_md_step(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_id: int,
        root_directory: str) -> int:
    """
    Return the current step for a given stage index.
    """
    assert len(anchor_or_systemwide.stage_parameters) > stage_index, "Stage index "\
        "provided is out of range."
    assert anchor_or_systemwide.stage_parameters[stage_index] is not None
    if isinstance(anchor_or_systemwide, structures.Anchor):
        if "molecular_dynamics" not in anchor_or_systemwide.scale_types:
            raise NotImplementedError("BD current step not yet implemented.")
    stage_directory = anchor_or_systemwide.stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, 
                                  stage_directory)
    checkpoint_info_basename = engines_runner.get_checkpoint_information_basename(
        swarm_id)
    checkpoint_information_filename = os.path.join(
        full_directory, checkpoint_info_basename)
    if os.path.exists(checkpoint_information_filename):
        checkpoint_information = engines_runner.load_checkpoint_information(
            checkpoint_information_filename)
        current_step = checkpoint_information.current_step
    else: 
        current_step = 0
    return current_step

def get_restart_checkpoint_filename(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_id: int,
        root_directory: str):
    """
    Return the restart checkpoint filename for a given stage index.
    """
    stage_directory = anchor_or_systemwide\
        .stage_parameters[stage_index]["directory"]
    full_directory = os.path.join(
        root_directory, anchor_or_systemwide.directory, stage_directory)
    restart_checkpoint_basename = engines_runner\
        .get_restart_checkpoint_basename(swarm_id)
    restart_checkpoint_filename = os.path.join(
        full_directory, restart_checkpoint_basename)
    return restart_checkpoint_filename

def construct_anchor_or_systemwide_index_list(
        model: structures.Seekr_model,
        stage_md: md_structures.Stage_md,
        anchor_index: str
        ) -> list[int | None]:
    """
    
    """
    anchor_or_systemwide_index_list: list[int | None] = []
    if stage_md.input_from == "systemwide":
        anchor_or_systemwide_index_list.append(None)
    elif stage_md.input_from == "anchors":
        for alpha, anchor in enumerate(model.anchors):
            if "molecular_dynamics" not in anchor.scale_types:
                continue
            if anchor_index not in ["any"]:
                try:
                    anchor_index_int = int(anchor_index)
                except ValueError:
                    return []
                if alpha != anchor_index_int:
                    continue
            anchor_or_systemwide_index_list.append(alpha)
    return anchor_or_systemwide_index_list

def choose_next_simulation_md(
        model: structures.Seekr_model, 
        stage_md: md_structures.Stage_md,
        anchor_index: str, 
        force_overwrite: bool,
        device_index: str | None = None,
        swarm_id_wanted: int | None = None,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    if device_index is not None:
        assert md_engine.platform_settings is not None
        md_engine.platform_settings.device_index = \
            device_index
    
    anchor_or_systemwide_index_list = construct_anchor_or_systemwide_index_list(
        model, stage_md, anchor_index)
    run_instructions_unsorted: list[typing.NamedTuple] = []
    anchor_or_systemwide: engines_structures.MD_systemwide_settings \
        | structures.Anchor
    for anchor_or_systemwide_index in anchor_or_systemwide_index_list:
        if anchor_or_systemwide_index is None:
            assert md_engine.systemwide_settings is not None
            anchor_or_systemwide = md_engine.systemwide_settings
            partition_name = "systemwide"
        else:
            anchor_or_systemwide = model.anchors[anchor_or_systemwide_index]
            partition_name = f"anchor {anchor_or_systemwide_index}"
        assert anchor_or_systemwide is not None
        # TODO: it's not enough to get at how many swarms, one needs to find the
        # index of the valid swarm ids
        #num_swarm_ids = anchor_or_systemwide.get_starting_structure_num_swarms(
        #    model.directory, stage_md.input_stage_index)
        swarm_ids = anchor_or_systemwide.get_starting_structure_swarm_ids(
            model.directory, stage_md.input_stage_index)
        calculation_dir = get_calculation_directory_md(
            anchor_or_systemwide, stage_md.index-1, model.directory)
        log_filename = os.path.join(calculation_dir, "run.log")
        base.configure_file_and_stdout_logger(
            log_filename, overwrite=force_overwrite)
        #for swarm_id in range(num_swarm_ids):
        for swarm_id in swarm_ids:
            if (swarm_id_wanted is not None) and (swarm_id != swarm_id_wanted):
                continue
            swarm_string = f" swarm {swarm_id}"
            current_step = get_current_md_step(anchor_or_systemwide,
                stage_md.index, swarm_id, model.directory)
            restart_checkpoint_filename \
                = get_restart_checkpoint_filename(
                    anchor_or_systemwide, stage_md.index, swarm_id, model.directory)
            if (not os.path.exists(restart_checkpoint_filename)) \
                    or force_overwrite:
                restart = False
                current_step = 0
            else:
                restart = True
            
            if not force_overwrite:
                run_finished, message = is_finished_md_with_message(
                    model, stage_md, anchor_or_systemwide, current_step, swarm_id,
                    use_json=use_json)
            else:
                run_finished = False
                message = "Force overwrite."
            if not silent:
                logging.info(f"{partition_name}{swarm_string}: {message}")
            if not run_finished:
                run_instruction = Run_instruction_md(
                    current_step, anchor_or_systemwide_index, restart, swarm_id, 
                    message)
                run_instructions_unsorted.append(run_instruction)

    # sort the run instructions by current step to run the laggers first
    run_instructions = sorted(
        run_instructions_unsorted,
        key=lambda item: item.current_step)
    return run_instructions

def do_run_instruction_md(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: Run_instruction_md,
        device_index: str | None,
    ) -> None:
    """
    Execute the run instructions for molecular dynamics.
    """
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    anchor_index = instruction.anchor_index
    restart = instruction.restart
    swarm_id = instruction.swarm_id
    swarm_string = f", swarm id: {swarm_id}"
    if force_overwrite and restart:
        restart = False
    if anchor_index is None:
        anchor_or_systemwide = md_engine.systemwide_settings
        partition_name = "systemwide"
    else:
        anchor_or_systemwide = model.anchors[anchor_index]
        partition_name = f"anchor {anchor_index}"
    log_string = f"Running {partition_name}, restart: "\
        +f"{restart}{swarm_string}"
    calculation_dir = get_calculation_directory_md(
        anchor_or_systemwide, stage_index, model.directory)
    log_filename = os.path.join(calculation_dir, "run.log")
    base.configure_file_and_stdout_logger(
        log_filename, overwrite=force_overwrite)
    logging.info(log_string)
    engines_runner.run_simulation(
        model, stage_index, restart, anchor_index=anchor_index, swarm_id=swarm_id, 
        device_index=device_index, force_overwrite=force_overwrite)