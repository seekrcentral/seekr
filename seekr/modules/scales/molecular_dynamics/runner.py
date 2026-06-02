from __future__ import annotations
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

def is_finished_md(
        model: structures.Seekr_model,
        stage: md_structures.Stage_md,
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings | structures.Anchor,
        current_step: int,
        swarm_id: int = 0,
    ) -> bool:
    """
    Check if the simulation is finished.
    """
    ending, _ = is_finished_md_with_message(model, stage, anchor_or_unpartitioned, current_step)
    return ending

def is_finished_md_with_message(
        model: structures.Seekr_model,
        stage: md_structures.Stage_md,
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings | structures.Anchor,
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
    if completion_criteria.type == "reporter_progress":
        # Resolve the JSON file written by the reporter on disk.
        # Reporter filenames follow the pattern:
        #   {running_dir}/{basename}1.swarm_{swarm_id}.reporter_{i}.{ext}
        # so the JSON sidecar is the same path with .json extension.
        reporter_name = completion_criteria.reporter_name
        reporter_index = None
        for i, reporter in enumerate(stage.reporters):
            if reporter.name == reporter_name:
                reporter_index = i
                break
        if reporter_index is None:
            message = {"completion_criteria": "reporter_progress",
                       "attained": False,
                       "error": f"reporter '{reporter_name}' not found in stage"}
            return False, message
        from seekr.modules.engines import runner as _eng_runner
        from seekr.modules.scales.molecular_dynamics import structures as _md_struct
        stage_directory = anchor_or_unpartitioned\
            .get_stage_parameters_by_index(stage.index)["directory"]
        #running_dir = os.path.join(
        #    model.directory, anchor_or_unpartitioned.directory, stage_directory)
        
        if stage.scope == "unpartitioned":
            running_dir = os.path.join(
                model.directory, anchor_or_unpartitioned.directory,
                stage_directory)
        elif stage.scope == "partitioned":
            running_dir = os.path.join(
                model.directory, "anchor_sets", stage.set_name,
                anchor_or_unpartitioned.directory,
                stage_directory)

        checkpoint_info_basename = _eng_runner.get_checkpoint_information_basename(swarm_id)
        basename = stage.sampling_method.get_output_positions_basename_noext()
        swarm_string = f".swarm_{swarm_id}"
        reporter_ext = stage.reporters[reporter_index].get_reporter_extension()
        reporter_name = stage.reporters[reporter_index].name
        json_glob = os.path.join(
            running_dir,
            f"{basename}*{swarm_string}.{reporter_name}.json")
        json_files = sorted(glob.glob(json_glob))
        progress = 0.0
        if json_files:
            import json as _json
            # Peek at format using the latest file.
            try:
                with open(json_files[-1], "r") as _f:
                    sample_data = _json.load(_f)
            except (OSError, _json.JSONDecodeError):
                sample_data = {}
            if sample_data:
                first_val = next(iter(sample_data.values()))
                if "filled" in first_val:
                    # Anchor_reporter format: per-file JSONs — aggregate ALL files.
                    all_filled: set = set()
                    all_indices: set = set()
                    for jf in json_files:
                        try:
                            with open(jf, "r") as _f:
                                data = _json.load(_f)
                        except (OSError, _json.JSONDecodeError):
                            continue
                        for v in data.values():
                            all_indices.add(int(v["index"]))
                            if v.get("filled", False):
                                all_filled.add(int(v["index"]))
                    if all_indices:
                        progress = len(all_filled) / len(all_indices)
                else:
                    # CV_value_reporter format: latest file is cumulative.
                    n_bins = len(sample_data)
                    filled = sum(
                        1 for v in sample_data.values()
                        if v.get("frame_index", -1) >= 0)
                    progress = filled / n_bins if n_bins > 0 else 1.0
        threshold = completion_criteria.reporter_progress_threshold
        max_steps = completion_criteria.max_number_of_steps
        attained = progress >= threshold
        timed_out = current_step >= max_steps
        ending = attained or timed_out
        if use_json:
            message = {
                "completion_criteria": "reporter_progress",
                "attained": attained,
                "timed_out": timed_out,
                "progress": round(progress, 4),
                "threshold": threshold,
                "current_step": current_step,
                "maximum_steps": max_steps,
            }
        else:
            message = (
                f"reporter_progress: {progress:.3f} / {threshold:.3f} "
                f"(step {current_step})"
            )
        return ending, message

    elif completion_criteria.type == "number_of_steps":
        if use_json:
            # Use JSON-style message
            progress = current_step / completion_criteria.number_of_steps \
                if completion_criteria.number_of_steps > 0 else 1.0
            message = {
                "completion_criteria": completion_criteria.type,
                "attained": current_step >= completion_criteria.number_of_steps,
                "progress": round(progress, 4),
                "current_step": current_step,
                "total_steps": completion_criteria.number_of_steps
            }
        else:
            message = f"steps {current_step} of {completion_criteria.number_of_steps} "\
                "completed."
        return current_step >= completion_criteria.number_of_steps, message
    
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
                    model, stage, anchor_or_unpartitioned, current_step, swarm_id)
            if found:
                return completed, message

    assert message is not None, \
        "Completion criteria not found: " + completion_criteria.type

def get_calculation_directory_md(
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor, 
        stage_index: int, 
        root_directory: str,
        set_name: str):
    """
    Return the calculation directory for a given MD stage index.
    """
    stage_list_index = stage_index + 1
    if isinstance(anchor_or_unpartitioned, structures.Anchor):
        if "molecular_dynamics" not in anchor_or_unpartitioned.scale_types:
            raise NotImplementedError("MD not running in this anchor/unpartitioned.")
        anchor_or_unpartitioned_directory = os.path.join(
            root_directory, "anchor_sets", set_name, 
            anchor_or_unpartitioned.directory
        )
    else:
        anchor_or_unpartitioned_directory = anchor_or_unpartitioned.directory
    stage_directory = anchor_or_unpartitioned.get_stage_parameters_by_index(stage_list_index)["directory"]
    full_directory = os.path.join(root_directory, anchor_or_unpartitioned_directory, 
                                  stage_directory)
    return full_directory

def get_current_md_step(
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_id: int,
        root_directory: str,
        set_name: str) -> int:
    """
    Return the current step for a given stage index.
    """
    if isinstance(anchor_or_unpartitioned, structures.Anchor):
        if "molecular_dynamics" not in anchor_or_unpartitioned.scale_types:
            raise NotImplementedError("BD current step not yet implemented.")
    stage_directory = anchor_or_unpartitioned.get_stage_parameters_by_index(
        stage_index)["directory"]
    if isinstance(anchor_or_unpartitioned, structures.Anchor):
        full_directory = os.path.join(
            root_directory, "anchor_sets", set_name, 
            anchor_or_unpartitioned.directory, stage_directory)
    else:
        full_directory = os.path.join(
            root_directory, anchor_or_unpartitioned.directory, 
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
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor, 
        stage_index: int, 
        swarm_id: int,
        root_directory: str,
        set_name: str):
    """
    Return the restart checkpoint filename for a given stage index.
    """
    stage_directory = anchor_or_unpartitioned.get_stage_parameters_by_index(
        stage_index)["directory"]
    if isinstance(anchor_or_unpartitioned, structures.Anchor):
        full_directory = os.path.join(
            root_directory, "anchor_sets", set_name, 
            anchor_or_unpartitioned.directory, stage_directory)
    else:
        full_directory = os.path.join(
            root_directory, anchor_or_unpartitioned.directory, 
            stage_directory)
    restart_checkpoint_basename = engines_runner\
        .get_restart_checkpoint_basename(swarm_id)
    restart_checkpoint_filename = os.path.join(
        full_directory, restart_checkpoint_basename)
    return restart_checkpoint_filename

def construct_anchor_or_unpartitioned_index_list(
        model: structures.Seekr_model,
        stage_md: md_structures.Stage_md,
        anchor_index: str
        ) -> list[int | None]:
    """
    
    """
    anchor_or_unpartitioned_index_list: list[int | None] = []
    if stage_md.scope == "unpartitioned":
        anchor_or_unpartitioned_index_list.append(None)
    elif stage_md.scope == "partitioned":
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
            anchor_or_unpartitioned_index_list.append(alpha)
    return anchor_or_unpartitioned_index_list

def choose_next_simulation_md(
        model: structures.Seekr_model, 
        stage_md: md_structures.Stage_md,
        anchor_index: str, 
        force_overwrite: bool,
        device_index: str | None = None,
        swarm_id_wanted: int | None = None,
        use_json: bool = True,
        silent: bool = False,
        benchmark: bool = False
        ) -> list:
    if benchmark and not force_overwrite:
        # Don't plan to run anything
        return []
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    if device_index is not None:
        assert md_engine.platform_settings is not None
        md_engine.platform_settings.device_index = \
            device_index
    
    anchor_or_unpartitioned_index_list = construct_anchor_or_unpartitioned_index_list(
        model, stage_md, anchor_index)
    run_instructions_unsorted: list[typing.NamedTuple] = []
    anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
        | structures.Anchor
    for anchor_or_unpartitioned_index in anchor_or_unpartitioned_index_list:
        if anchor_or_unpartitioned_index is None:
            assert md_engine.unpartitioned_settings is not None
            anchor_or_unpartitioned = md_engine.unpartitioned_settings
            partition_name = "unpartitioned"
        else:
            anchor_or_unpartitioned = model.anchors[anchor_or_unpartitioned_index]
            partition_name = f"anchor {anchor_or_unpartitioned_index}"
        assert anchor_or_unpartitioned is not None
        # TODO: it's not enough to get at how many swarms, one needs to find the
        # index of the valid swarm ids
        #num_swarm_ids = anchor_or_unpartitioned.get_starting_structure_num_swarms(
        #    model.directory, stage_md.input_stage_index)
        swarm_ids = anchor_or_unpartitioned.get_starting_structure_swarm_ids(
            model.directory, stage_md.input_stage_index, stage_md.set_name)
        calculation_dir = get_calculation_directory_md(
            anchor_or_unpartitioned, stage_md.index-1, model.directory,
            stage_md.set_name)
        log_filename = os.path.join(calculation_dir, "run.log")
        base.configure_file_and_stdout_logger(
            log_filename, overwrite=False)
        for swarm_id in swarm_ids:
            if (swarm_id_wanted is not None) and (swarm_id != swarm_id_wanted):
                continue
            swarm_string = f" swarm {swarm_id}"
            current_step = get_current_md_step(anchor_or_unpartitioned,
                stage_md.index, swarm_id, model.directory, stage_md.set_name)
            restart_checkpoint_filename \
                = get_restart_checkpoint_filename(
                    anchor_or_unpartitioned, stage_md.index, swarm_id, 
                    model.directory, stage_md.set_name)
            if (not os.path.exists(restart_checkpoint_filename)) \
                    or force_overwrite:
                restart = False
                current_step = 0
            else:
                restart = True
            
            if not force_overwrite:
                run_finished, message = is_finished_md_with_message(
                    model, stage_md, anchor_or_unpartitioned, current_step, swarm_id,
                    use_json=use_json)
            else:
                run_finished = False
                message = "Force overwrite."
            if not silent:
                logging.info(f"{partition_name}{swarm_string}: {message}")
            if not run_finished:
                run_instruction = Run_instruction_md(
                    current_step, anchor_or_unpartitioned_index, restart, swarm_id, 
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
        benchmark: bool = False
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
        anchor_or_unpartitioned = md_engine.unpartitioned_settings
        partition_name = "unpartitioned"
    else:
        anchor_or_unpartitioned = model.anchors[anchor_index]
        partition_name = f"anchor {anchor_index}"
    log_string = f"Running {partition_name}, restart: "\
        +f"{restart}{swarm_string}"
    stage = model.stages[stage_index]
    calculation_dir = get_calculation_directory_md(
        anchor_or_unpartitioned, stage_index, model.directory,
        stage.set_name)
    log_filename = os.path.join(calculation_dir, "run.log")
    base.configure_file_and_stdout_logger(
        log_filename, overwrite=False)
    logging.info(log_string)
    engines_runner.run_simulation(
        model, stage_index, restart, anchor_index=anchor_index, 
        swarm_id=swarm_id, device_index=device_index, 
        force_overwrite=force_overwrite, benchmark=benchmark)