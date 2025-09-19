"""
modules/scales/molecular_dynamics/runner.py

Perform runner functions specific to molecular dynamics stages.
"""

import os
import typing
import logging
from collections import namedtuple

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.engines.runner as engines_runner
import seekr.modules.completion_criteria.runner as completion_criteria_runner

Run_instruction_md = namedtuple(
    "Run_instruction_md",
    [
        "current_step",
        "anchor_index",
        "restart",
        "swarm_index"
    ])


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
    full_directory = os.path.join(root_directory, anchor_or_systemwide.directory, stage_directory)
    return full_directory

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
        if "molecular_dynamics" not in anchor_or_systemwide.scale_types:
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
        model: structures.Seekr_model, 
        stage_md: md_structures.Stage_md,
        anchor_index: str, 
        force_overwrite: bool,
        device_index: str | None = None
        ) -> list:
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    if device_index is not None:
        assert md_engine.platform_settings is not None
        md_engine.platform_settings.device_index = \
            device_index
    
    #run_instructions_unsorted: list = []
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
                    model, stage_md, anchor_or_systemwide, current_step, swarm_frame)
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
    swarm_index = instruction.swarm_index
    if swarm_index is None:
        swarm_string = ""
    else:
        swarm_string = f", swarm index: {swarm_index}"
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
        model, stage_index, restart, anchor_index=anchor_index, swarm_index=swarm_index, 
        device_index=device_index, force_overwrite=force_overwrite)