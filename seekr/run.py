from __future__ import annotations
"""
run.py

Run simulations required for the different stages and anchors
of the SEEKR calculation.
"""

import os
import sys
import shlex
import argparse
import logging

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.scales.runner as scales_runner

# Max number of times to check for remaining instructions this stage
MAX_ITER = 999

def extract_stage_index(
        model: structures.Seekr_model, 
        stage_arg: str
        ) -> int | None:
    for i, stage_obj in enumerate(model.stages):
        if stage_obj.name == stage_arg:
            return i
    try:
        int_stage = int(stage_arg)
    except ValueError:
        return None
    for i, stage_obj in enumerate(model.stages):
        if stage_obj.index == int_stage:
            return i
    return None

def extract_anchor_index(
        model: structures.Seekr_model, 
        anchor_arg: str
        ) -> str | None:
    """
    Catch instructions that are not valid and throw an error.
    """
    if anchor_arg == "any":
        return "any"
    for alpha, anchor in enumerate(model.anchors):
        if anchor.name == anchor_arg:
            return str(alpha)
    try:
        int_anchor = int(anchor_arg)
    except ValueError:
        return None
    if (int_anchor < model.num_anchors) and (int_anchor >= 0):
        return str(int_anchor)
    return None

def run(
        model: structures.Seekr_model, 
        stage_arg: str, 
        anchor_arg: str = "any", 
        device_index: str | None = None, 
        force_overwrite: bool = False,
        swarm_id: int | None = None,
        benchmark: bool = False
        ) -> None:
    stage_index = extract_stage_index(model, stage_arg)
    assert stage_index is not None, \
        "Stage argument must match a name or index of a stage in the model."
    stage = model.stages[stage_index]
    stage.load_sets_into_model(model)
    anchor_index_instruction = extract_anchor_index(model, anchor_arg)
    if stage.scope == "partitioned":
        # anchor_index only needs to be defined if the stage uses anchors.
        assert anchor_index_instruction is not None, \
            "Anchor argument must match a name or index of an anchor in "\
            "the model."
    assert (device_index is None) \
        or (isinstance(device_index, str)), \
        "The device_index argument must be a string or None."
    curdir = os.getcwd()
    complete = False
    ran_nothing = True
    counter = 0
    choose_params = {
        "device_index": device_index,
        "anchor_index_instruction": anchor_index_instruction,
    }
    while not complete:
        run_instructions = scales_runner.choose_next_simulation(
            model, stage, choose_params, force_overwrite, swarm_id, use_json=False, 
            benchmark=benchmark)
        for instruction in run_instructions:
            scales_runner.do_run_instruction(
                model, force_overwrite, stage_index, instruction, device_index, benchmark)
            
        if len(run_instructions) > 0:
            complete = False
            ran_nothing = False
        else:
            complete = True
        counter += 1
        if counter > MAX_ITER:
            raise Exception("While loop appears to be stuck.")
        force_overwrite = False
        
    os.chdir(curdir)
    if ran_nothing:
        log_string = "Nothing was run because all criteria are satisfied. "\
            "If this is unexpected, make sure that all relevant unpartitioned "\
            "or anchors have assigned starting coordinates (ex: PDB file)."
        synopsis_filename = model.get_synopsis_filename(model.directory)
        base.configure_file_and_stdout_logger(synopsis_filename, False)
        logging.warning(log_string)

    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of the model file for SEEKR calculation. This would "\
        "be the JSON file generated in the prepare stage.")
    argparser.add_argument(
        "stage", metavar="STAGE", type=str,
        help="The name or index of the stage to run.")
    argparser.add_argument(
        "-a", "--anchor", dest="anchor", type=str, default="any",
        help="The name or index of the anchor whose simulation to start. "\
            "or restart. The argument 'any' is allowed - which will run "\
            "whichever anchors that still need to be finished for the "\
            "specified stage. Default: 'any'.")
    argparser.add_argument(
        "-s", "--swarm_id", dest="swarm_id", type=int, default=None,
        help="The swarm ID to run. This is only relevant for stages that "\
            "use swarm sampling methods. If not specified, all swarm IDs "\
            "that need to be run will be run. Default: None.",)
    argparser.add_argument(
        "-c", "--device_index", dest="device_index", default=None,
        help="modify which CUDA (or other GPU platform) device_index to run "\
            "the simulation on when a simulation engine is GPU enabled. For "\
            "example, the number 0 or 1 would suffice. To run on multiple GPU "\
            "indices, simply enter comma separated indices. Example: '0,1'. If "\
            "a value is not supplied, the value in the MODEL_FILE will be used "\
            "by default. If a CPU implementation will be used, this argument is "\
            "ignored. Default: None.", type=str)
    argparser.add_argument(
        "-f", "--force_overwrite", dest="force_overwrite", default=False, 
        help="Toggle whether to overwrite existing files within an anchor. "\
            "If this option is enabled then the anchor's production directory "\
            "will be emptied of all output, trajectory, and backup files for "\
            "the new simulation.", action="store_true")
    argparser.add_argument(
        "-r", "--restart_attempts", dest="restart_attempts", default=0, 
        help="If the simulation fails during The course of a single run, "\
            "attempt to restart the simulation this many number of times "\
            "before giving up. Default: 0.", type=int)
    argparser.add_argument(
        "-b", "--benchmark", dest="benchmark", default=False,
        help="Toggle whether to run in benchmark mode. In this mode, the "\
            "simulation will be run for a short amount of time (ex: 1 ps) "\
            "to test the performance of the simulation engine. Warning: if "\
            "run with the --anchor set to 'any', the benchmark mode will be "\
            "applied to all anchors, and adequate inputs may not be "\
            "generated for later stages.", action="store_true")
    argspace = argparser.parse_args()
    args = vars(argspace)
    model_file = args["model_file"]
    stage = args["stage"]
    anchor = args["anchor"]
    swarm_id = args["swarm_id"]
    device_index = args["device_index"]
    force_overwrite = args["force_overwrite"]
    restart_attempts = args["restart_attempts"]
    benchmark = args["benchmark"]
    model = structures.load_model(model_file)
    num_restart_attempts = 0
    success = False
    # Log the command line arguments for reproducibility
    synopsis_filename = model.get_synopsis_filename(model.directory)
    base.configure_file_only_logger(synopsis_filename, False)
    logging.info(f"Command line: {shlex.join(sys.argv)}")
    while not success:
        try:
            run(model, stage, anchor, device_index, force_overwrite, swarm_id, benchmark)
            success = True
        except Exception:
            if num_restart_attempts < restart_attempts:
                log_string = "Simulation failure. Attempting restart "\
                      f"{num_restart_attempts} of {restart_attempts}"
                base.configure_file_and_stdout_logger(synopsis_filename, False)
                logging.warning(log_string)
                num_restart_attempts += 1
            else:    
                raise