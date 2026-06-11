from __future__ import annotations
"""
modules/scales/runner.py

Differentiate between different scales and call the appropriate
runner functions for each scale.
"""

import importlib

import seekr.modules.base as base
import seekr.modules.scales.base as scales_base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.molecular_dynamics.runner as md_runner
import seekr.modules.scales.brownian_dynamics.runner as bd_runner
import seekr.modules.scales.logistic.runner as logistic_runner

def is_finished_with_message(
        model: structures.Seekr_model,
        stage: scales_base.Base_stage,
        anchor_index: str | None = None,
        use_json: bool = True
    ) -> tuple[bool, list[str]]:
    """
    Check if the given stage is finished, and return a tuple of (is_finished,
    message).
    """
    if use_json:
        final_message = {}
    else:
        final_message = ""
    if stage.scale_type == "molecular_dynamics":
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        anchor_or_unpartitioned_index_list = md_runner\
            .construct_anchor_or_unpartitioned_index_list(model, stage, anchor_index)
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings \
            | structures.Anchor
        if len(anchor_or_unpartitioned_index_list) == 0:
            finished = False
        else:
            finished = True
        for anchor_or_unpartitioned_index in anchor_or_unpartitioned_index_list:
            if anchor_or_unpartitioned_index is None:
                assert md_engine.unpartitioned_settings is not None
                anchor_or_unpartitioned = md_engine.unpartitioned_settings
            else:
                anchor_or_unpartitioned \
                    = model.anchors[anchor_or_unpartitioned_index]
            tmp_finished, message = md_runner.is_finished_md_with_message(
                model, stage, anchor_or_unpartitioned, use_json=use_json)
            if not tmp_finished:
                finished = False
            if use_json:
                final_message[anchor_or_unpartitioned_index] = message
            else:
                if anchor_or_unpartitioned_index is None:
                    final_message += f"    Unpartitioned: {message}\n"
                else:
                    final_message += f"    Anchor {anchor_or_unpartitioned_index}: {message}\n"

    elif stage.scale_type == "brownian_dynamics":
        finished, message = bd_runner.is_finished_bd_with_message(
            model, stage, use_json=use_json)
        if use_json:
            final_message[None] = message
        else:
            final_message += f"    Unpartitioned: {message}\n"

    elif stage.scale_type == "logistic":
        finished, message = logistic_runner.is_finished_logistic_with_message(
            model, stage, use_json=use_json)
        if use_json:
            final_message[None] = message
        else:
            final_message += f"    Unpartitioned: {message}\n"
    else:
        found = False
        for plugin in model.plugins:
            scales_module_str = base.get_plugin_str(
                plugin, submodule_name="scales", require_preparer=True,
                require_openmm_interface=False, require_runner=True)
            if scales_module_str is not None:
                scales_module_run = importlib.import_module(scales_module_str+".runner")
                this_module_found, finished, final_message \
                    = scales_module_run.is_finished_with_message(
                        model, stage, anchor_index, use_json)
                if this_module_found:
                    found = True
                    break

        if not found:
            raise Exception(f"Stage type not recognized: {stage.type}.")
        
    return finished, final_message

def construct_anchor_or_unpartitioned_index_list(
        model: structures.Seekr_model,
        stage: scales_base.Base_stage,
        anchor_index: str
        ) -> list[int | None]:
    """
    Construct a list of anchor or unpartitioned indices based on the stage and
    anchor index.
    """
    if stage.scale_type == "molecular_dynamics":
        return md_runner.construct_anchor_or_unpartitioned_index_list(
            model, stage, anchor_index)
    elif stage.scale_type == "brownian_dynamics":
        raise NotImplementedError("BD stages do not support anchor-specific "\
                                  "operations.")
    else:
        found = False
        for plugin in model.plugins:
            scales_module_str = base.get_plugin_str(
                plugin, submodule_name="scales", require_preparer=True,
                require_openmm_interface=False, require_runner=True)
            if scales_module_str is not None:
                scales_module_run = importlib.import_module(scales_module_str+".runner")
                this_module_found, result \
                    = scales_module_run.construct_anchor_or_unpartitioned_index_list(
                        model, stage, anchor_index)
                if this_module_found:
                    found = True
                    return result

        if not found:
            raise Exception(f"Stage type not recognized: {stage.type}.")

def choose_next_simulation(
        model: structures.Seekr_model,
        stage: scales_base.Base_stage,
        choose_params: dict,
        force_overwrite: bool,
        swarm_id: int | None = None,
        use_json: bool = True,
        silent: bool = False,
        benchmark: bool = False
        ) -> list:
    """
    Choose the next simulation to run based on the stage scale, and return
    a list of instructions to run in this round.
    """
    if stage.scale_type == "molecular_dynamics":
        anchor_index_instruction = choose_params["anchor_index_instruction"]
        device_index = choose_params["device_index"]
        run_instructions = md_runner.choose_next_simulation_md(
            model, stage, anchor_index_instruction, force_overwrite, 
            device_index, swarm_id, use_json, silent=silent, benchmark=benchmark)
    elif stage.scale_type == "brownian_dynamics":
        assert swarm_id is None, "Swarm ID is not used in BD simulations."
        run_instructions = bd_runner.choose_next_simulation_bd(
            model, stage, force_overwrite, use_json, silent=silent, 
            benchmark=benchmark)
    elif stage.scale_type == "logistic":
        run_instructions = logistic_runner.choose_next_simulation_logistic(
            model, stage, force_overwrite, use_json, silent=silent)
    else:
        found = False
        for plugin in model.plugins:
            scales_module_str = base.get_plugin_str(
                plugin, submodule_name="scales", require_preparer=True,
                require_openmm_interface=False, require_runner=True)
            if scales_module_str is not None:
                scales_module_run = importlib.import_module(scales_module_str+".runner")
                this_module_found, run_instructions \
                    = scales_module_run.choose_next_simulation(
                        model, stage, choose_params, force_overwrite, swarm_id, use_json, 
                        silent, benchmark)
                if this_module_found:
                    found = True
                    break

        if not found:
            raise Exception(f"Stage type not recognized: {stage.type}.")
        
    return run_instructions
    
def get_calculation_directory(
        engine: engines_structures.Base_engine,
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
        
    else:
        found = False
        for plugin in engine.model.plugins:
            scales_module_str = base.get_plugin_str(
                plugin, submodule_name="scales", require_preparer=True,
                require_openmm_interface=False, require_runner=True)
            if scales_module_str is not None:
                scales_module_run = importlib.import_module(scales_module_str+".runner")
                this_module_found, full_directory = scales_module_run.get_calculation_directory(
                    engine, root_directory)
                if this_module_found:
                    found = True
                    break
        if not found:
            raise Exception(f"Scale type {engine.scale_type} not recognized.")
        
    return full_directory
    
def do_run_instruction(
        model: structures.Seekr_model,
        force_overwrite: bool,
        stage_index: int,
        instruction: list,
        device_index: str | None,
        benchmark: bool = False
    ) -> None:
    """
    Execute the run instructions for a given scale type.
    """
    stage = model.stages[stage_index]
    if benchmark:
        if hasattr(stage, "completion_criteria"):
            stage.completion_criteria.enable_benchmark_mode()

    if stage.scale_type == "molecular_dynamics":
        md_runner.do_run_instruction_md(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction, device_index=device_index, 
            benchmark=benchmark)
    elif stage.scale_type == "brownian_dynamics":
        bd_runner.do_run_instruction_bd(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction, benchmark=benchmark)
    elif stage.scale_type == "logistic":
        logistic_runner.do_run_instruction_logistic(
            model, force_overwrite, stage_index=stage_index, 
            instruction=instruction)
    else:
        found = False
        for plugin in model.plugins:
            scales_module_str = base.get_plugin_str(
                plugin, submodule_name="scales", require_preparer=True,
                require_openmm_interface=False, require_runner=True)
            if scales_module_str is not None:
                scales_module_run = importlib.import_module(scales_module_str+".runner")
                this_module_found = scales_module_run.do_run_instruction(
                    model, force_overwrite, stage_index, instruction, device_index,
                    benchmark=benchmark)
                if this_module_found:
                    found = True
                    break

        if not found:
            raise Exception(f"Stage type not recognized: {stage.scale_type}.")
    return