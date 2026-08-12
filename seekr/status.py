from __future__ import annotations
"""
status.py

Obtain and return the status of a seekr calculation - that is, the progress
of any given stage, as well as things like the number of transitions
and times observed in a MMVT stage. Equivalent to the anchor_steps script
from seekr2.
"""

import argparse

import seekr.modules.structures as structures
import seekr.modules.scales.runner as scales_runner
import seekr.run as run
import seekr.analyze as analyze

AVAILABLE_INSTRUCTIONS = ["info", "progress", "statistics", "all"]

def status(
        model: structures.Seekr_model, 
        instruction: str, 
        stage_arg: str = "all", 
        anchor_arg: str = "any", 
        swarm_id: int | None = None,
        print_json: bool = True
        ) -> str | dict:
    # Stage(s)
    if instruction not in AVAILABLE_INSTRUCTIONS:
        raise ValueError(f"Unsupported instruction: {instruction}")
    if stage_arg == "all":
        stages_to_check = [None] + list(range(len(model.stages)))
    else:
        stage_index = run.extract_stage_index(model, stage_arg)
        if stage_index is None:
            stages_to_check = [None]
        else:
            stages_to_check = [stage_index]
    
    anchor_index_instruction = run.extract_anchor_index(model, anchor_arg)
    # Load info for the initial stage
    if instruction in ["info", "all"]:
        if None in stages_to_check:
            model.load_cvs("main")
            model.load_selections("main")
            model.load_anchor_set("main")
            stage_0_num_anchors = len(model.anchors)
            md_engine = model.get_engine_by_scale_type("molecular_dynamics")
            if md_engine is None:
                stage_0_num_swarms = 1
            else:
                swarm_ids = md_engine.unpartitioned_settings\
                .get_starting_structure_swarm_ids(
                        model.directory, 0, "main")
                stage_0_num_swarms = len(swarm_ids)

    if print_json:
        message = {}
        if instruction in ["info", "all"]:
            if None in stages_to_check:
                message["info"] = {
                    0: {
                        "stage_name": "initial",
                        "scope": "N/A",
                        "num_anchors": stage_0_num_anchors,
                        "num_swarms": stage_0_num_swarms
                    }
                }
            else:
                message["info"] = {}
    else:
        message = ""
        if instruction in ["info", "all"]:
            message += "Info:\n"
            if None in stages_to_check:
                message += f"  Stage 0 (initial):\n"
                message += f"    Scope: N/A\n"
                message += f"    Number of anchors: {stage_0_num_anchors}\n"
                message += f"    Number of swarms: {stage_0_num_swarms}\n"
    stages_to_check = [s for s in stages_to_check if s is not None]
    choose_params = {
        "device_index": None,
        "anchor_index_instruction": anchor_index_instruction,
    }
    for stage_index in stages_to_check:
        stage = model.stages[stage_index]
        stage.load_sets_into_model(model)
        if instruction in ["info", "all"]:
            scope = stage.scope
            if stage.input_stage_index == 0:
                input_scope = scope
            else:
                input_stage = model.stages[stage.input_stage_index-1]
                input_scope = input_stage.scope
            if input_scope == "unpartitioned":
                num_anchors = None
                if (stage.scale_type == "molecular_dynamics") \
                        or (stage.scale_type == "logistic"):
                    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
                    swarm_ids = md_engine.unpartitioned_settings\
                        .get_starting_structure_swarm_ids(
                            model.directory, stage.input_stage_index, 
                            stage.set_name)
                    num_swarms = len(swarm_ids)
                else:
                    num_swarms = 1
                
            elif input_scope == "partitioned":
                num_anchors = len(model.anchors)
                if num_anchors == 0:
                    num_swarms = 0
                else:
                    swarm_ids = model.anchors[0].get_starting_structure_swarm_ids(
                        model.directory, stage.input_stage_index, stage.set_name)
                    num_swarms = len(swarm_ids)
            else:
                raise ValueError(f"Unsupported scope: {scope}")

            if print_json:
                message["info"][stage.index] = {
                    "stage_name": stage.name,
                    "scope": scope,
                    "num_anchors": num_anchors,
                    "num_swarms": num_swarms
                }
            else:
                message += f"  Stage {stage.index} ({stage.name}):\n"
                message += f"    Scope: {scope}\n"
                message += f"    Number of anchors: {num_anchors}\n"
                message += f"    Number of swarms: {num_swarms}\n"

    if print_json:
        if instruction in ["progress", "all"]:
            message["progress"] = {}
        if instruction in ["statistics", "all"]:
            message["statistics"] = {}
    else:
        if instruction in ["progress", "all"]:
            message += "Progress:\n"
    for stage_index in stages_to_check:
        stage = model.stages[stage_index]
        stage.load_sets_into_model(model)
        if instruction in ["progress", "all"]:
            finished_tmp, stage_message_tmp = scales_runner.is_finished_with_message(
                model, stage, anchor_index_instruction, use_json=print_json)
            run_instructions = scales_runner.choose_next_simulation(
                model, stage, choose_params, False, swarm_id, use_json=print_json,
                silent=True)
            if finished_tmp:
                if len(run_instructions) > 0:
                    finished = False
                    #stage_message = str(run_instructions[0].message) + "\n"
                    stage_message = stage_message_tmp
                else:
                    finished = True
                    stage_message = stage_message_tmp
            else:
                finished = False
                stage_message = stage_message_tmp
                
            if print_json:
                message["progress"][stage.index] = {
                    "stage_name": stage.name,
                    "finished": finished,
                    "progress": stage_message
                }
            else:
                message += f"  Stage {stage.index} ({stage.name}):\n" \
                    + f"    Finished: {finished}\n{stage_message}"

    if instruction in ["statistics", "all"]:
        analysis = analyze.analyze_all(model, stage_arg=None, 
                                    skip_checks=True, only_extract_data=True)
        if analysis is None:
            return message
        stats_message = analysis.get_statistics_message(
            model, anchor_index_instruction, use_json=print_json)
        if print_json:
            message["statistics"] = stats_message
        else:
            message += "\nStatistics:\n"
            message += stats_message
            
    return message

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of the model file for SEEKR calculation. This would "\
        "be the JSON file generated in the prepare stage.")
    argparser.add_argument(
        "-i", "--instruction", dest="instruction", type=str, default="all",
        help="Which status information to print. Options are: 'info', 'progress', "\
            "'statistics', or 'all'. Default: 'all'.")
    argparser.add_argument(
        "-S", "--stage", dest="stage", type=str, default="all",
        help="The name or index of the stage to run.")
    argparser.add_argument(
        "-a", "--anchor", dest="anchor", type=str, default="any",
        help="The name or index of the anchor whose simulation to start. "\
            "or restart. The argument 'any' is allowed - which will run "\
            "whichever anchors that still need to be finished for the "\
            "specified anchor. Default: 'any'.")
    argparser.add_argument(
        "-s", "--swarm_id", dest="swarm_id", type=int, default=None,
        help="The swarm ID to run. This is only relevant for stages that "\
            "use swarm sampling methods. If not specified, all swarm IDs "\
            "that need to be run will be run. Default: None.",)
    argparser.add_argument(
        "-j", "--json", dest="json", action="store_true", default=False,
        help="If specified, the status will be printed in JSON format.")
    
    argspace = argparser.parse_args()
    args = vars(argspace)
    model_file = args["model_file"]
    instruction = args["instruction"]
    stage = args["stage"]
    anchor = args["anchor"]
    swarm_id = args["swarm_id"]
    print_json = args["json"]
    model = structures.load_model(model_file)
    message = status(
        model, instruction, stage_arg=stage, anchor_arg=anchor, swarm_id=swarm_id, 
        print_json=print_json)
    if print_json:
        import json
        print(json.dumps(message, indent=4))
    else:
        print(message, end="")