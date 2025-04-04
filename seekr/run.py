"""
run.py

Run simulations required for the different stages and anchors
of the SEEKR calculation.
"""

import os
import argparse
import logging

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.runner as runner
import seekr.modules.engines.runner as engines_runner

MAX_ITER = 999

def run(
        model: structures.Seekr_model, 
        stage_arg: str, 
        anchor_arg: str = "any", 
        device_index: str | None = None, 
        force_overwrite: bool = False
        ) -> None:
    stage_index = extract_stage_index(model, stage_arg)
    assert stage_index is not None, \
        "Stage argument must match a name or index of a stage in the model."
    stage = model.stages[stage_index]
    anchor_index = extract_anchor_index(model, anchor_arg)
    if stage.input_from == "anchors":
        # anchor_index only needs to be defined if the stage uses anchors.
        assert anchor_index is not None, \
            "Anchor argument must match a name or index of an anchor in "\
            "the model."
    assert (device_index is None) \
        or (isinstance(device_index, str)), \
        "The device_index argument must be a string or None."
    curdir = os.getcwd()
    complete = False
    ran_nothing = True
    counter = 0
    while not complete:
        if stage.type == "md_stage":
            info_to_run = runner.choose_next_simulation_md(stage, model, anchor_index,
                                             force_overwrite, device_index)
        elif stage.type == "bd_stage":
            info_to_run = runner.choose_next_simulation_bd(stage, model, force_overwrite)
        else:
            raise Exception(f"Stage type not recognized: {stage.type}.")
        
        for info in info_to_run:
            if stage.type == "md_stage":
                steps_to_go_min, num_transitions, anchor_index, \
                    restart, total_simulation_length, swarm_index = info
                if force_overwrite and restart:
                    restart = False
                log_string = f"running anchor_index: {anchor_index} restart: "\
                    +f"{restart} total_simulation_length: "\
                    +f"{total_simulation_length} num_transitions: "\
                    +f"{num_transitions} swarm index: {swarm_index}"
                if anchor_index is None:
                    anchor_or_systemwide = model.md_engine.systemwide_settings
                else:
                    anchor_or_systemwide = model.anchors[anchor_index]
                calculation_dir = runner.get_calculation_directory_md(
                    anchor_or_systemwide, stage_index, model.directory)
                log_filename = os.path.join(calculation_dir, "run.log")
                base.configure_file_and_stdout_logger(
                    log_filename, overwrite=force_overwrite)
                logging.info(log_string)
                engines_runner.run_md_simulation(
                    model, stage_index, anchor_index, restart, 
                    swarm_index, device_index, force_overwrite)
            elif stage.type == "bd_stage":
                if not model.using_bd():
                    raise Exception("BD engine is not enabled.")
                trajs_to_go_min, num_transitions, b_surface_anchor_index, \
                    restart, total_num_trajs = info
                if force_overwrite and restart:
                    restart = False
                log_string = f"running from b-surface anchor: {b_surface_anchor_index}"\
                    +f"restart: {restart} trajectories to run: {trajs_to_go_min}"\
                    +f"trajectories so far: {total_num_trajs}"\
                    +f"number of transitions: {num_transitions}"
                logging.info(log_string)
                engines_runner.run_bd_simulation(
                    model, trajs_to_go_min, restart, force_overwrite)
            else:
                raise Exception(f"Stage type not recognized: {stage.type}.")
            
        if len(info_to_run) > 0:
            complete = False
            ran_nothing = False
        else:
            complete = True
        counter += 1
        if counter > MAX_ITER:
            raise Exception("While loop appears to be stuck.")
        
    os.chdir(curdir)
    if ran_nothing:
        log_string = "Nothing was run because all criteria are satisfied. "\
            "If this is unexpected, make sure that all relevant systemwide or "\
            "anchors have assigned starting coordinates (ex: PDB file)."
        synopsis_filename = model.get_synopsis_filename(model.directory)
        base.configure_file_and_stdout_logger(synopsis_filename, False)
        logging.warning(log_string)

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
            "specified anchor. Default: 'any'.")
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
    argspace = argparser.parse_args()
    args = vars(argspace)
    model_file = args["model_file"]
    stage = args["stage"]
    anchor = args["anchor"]
    device_index = args["device_index"]
    force_overwrite = args["force_overwrite"]
    restart_attempts = args["restart_attempts"]
    model = structures.load_model(model_file)
    num_restart_attempts = 0
    success = False
    while not success:
        try:
            run(model, stage, anchor, device_index, force_overwrite)
            success = True
        except Exception:
            if num_restart_attempts < restart_attempts:
                log_string = "Simulation failure. Attempting restart "\
                      f"{num_restart_attempts} of {restart_attempts}"
                num_restart_attempts += 1
            else:    
                raise