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

AVAILABLE_INSTRUCTIONS = ["progress", "statistics", "all"]

def status(
        model: structures.Seekr_model, 
        instruction: str, 
        stage_arg: str = "all", 
        anchor_arg: str = "any", 
        print_json: bool = True
        ) -> str | dict:
    # Stage(s)
    if instruction not in AVAILABLE_INSTRUCTIONS:
        raise ValueError(f"Unsupported instruction: {instruction}")
    if stage_arg == "all":
        stages_to_check = list(range(len(model.stages)))
    else:
        stage_index = run.extract_stage_index(model, stage_arg)
        stages_to_check = [stage_index]
    # Anchor(s)
    anchor_index_instruction = run.extract_anchor_index(model, anchor_arg)
    if print_json:
        message = {}
        if instruction in ["progress", "all"]:
            message["progress"] = {}
        if instruction in ["statistics", "all"]:
            message["statistics"] = {}
    else:
        message = ""
        if instruction in ["progress", "all"]:
            message += "Progress:\n"

    for stage_index in stages_to_check:
        stage = model.stages[stage_index]
        if instruction in ["progress", "all"]:
            finished, stage_message = scales_runner.is_finished_with_message(
                model, stage, anchor_index_instruction, use_json=print_json)
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
        analysis = analyze.analyze(model, stage_index=None, 
                                    skip_checks=True, only_extract_data=True)
        if analysis is None:
            return message
        stats_message = analysis.get_statistics_message(
            model, anchor_index_instruction, use_json=print_json)
        if print_json:
            message["statistics"] = stats_message
        else:
            message += "Statistics:\n"
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
        help="Which status information to print. Options are: 'progress', "\
            "'statistics', or 'all'. Default: 'all'.")
    argparser.add_argument(
        "-s", "--stage", dest="stage", type=str, default="all",
        help="The name or index of the stage to run.")
    argparser.add_argument(
        "-a", "--anchor", dest="anchor", type=str, default="any",
        help="The name or index of the anchor whose simulation to start. "\
            "or restart. The argument 'any' is allowed - which will run "\
            "whichever anchors that still need to be finished for the "\
            "specified anchor. Default: 'any'.")
    argparser.add_argument(
        "-j", "--json", dest="json", action="store_true", default=False,
        help="If specified, the status will be printed in JSON format.")
    
    argspace = argparser.parse_args()
    args = vars(argspace)
    model_file = args["model_file"]
    instruction = args["instruction"]
    stage = args["stage"]
    anchor = args["anchor"]
    print_json = args["json"]
    model = structures.load_model(model_file)
    message = status(
        model, instruction, stage_arg=stage, anchor_arg=anchor, print_json=print_json)
    if print_json:
        import json
        print(json.dumps(message, indent=4))
    else:
        print(message, end="")