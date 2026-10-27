from __future__ import annotations
"""
analyze.py

Analyze the results of SEEKR simulations for calculation of rate constants,
free energies, convergence, and other properties and plots.
"""

import os
import sys
import json
import typing
import logging
import argparse
import warnings

if "-j" in sys.argv or "--json" in sys.argv:
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated",
        category=UserWarning,
    )

import seekr._version as seekr_version
import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.check as check
import seekr.modules.analyzers.base as analyzer_base
import seekr.modules.analyzers.analyzer as analyzer
import seekr.run as run

logger = logging.getLogger(__name__)

__version__ = seekr_version.__version__

def analyze_all(
        model: structures.Seekr_model,
        stage_arg: str | None = None,
        skip_checks: bool = False,
        extra_arguments: dict | None = None,
        silent: bool = False,
        print_json: bool = False
        ) -> typing.Any:
    if extra_arguments is None:
        extra_arguments = {}
    curdir = os.getcwd()
    os.chdir(model.directory)
    synopsis_filename = model.get_synopsis_filename(model.directory)
    base.configure_logging(synopsis_filename, console=not print_json)
    if not silent and not print_json:
        logger.info(
            f"Analyzing with stage instruction {stage_arg} and extra "\
            f"arguments {extra_arguments}")
    base.log_invocation_metadata(__version__)
    if (stage_arg is None) or (stage_arg == "all"):
        stages_to_analyze = [stage.index for stage in model.stages]
    else:
        stages_to_analyze = [run.extract_stage_index(model, stage_arg)+1]
    anything_analyzed = False
    last_significant_analysis = None
    json_results = {}
    for stage_index in stages_to_analyze:
        if not hasattr(model.stages[stage_index-1], "sampling_method"):
            # Logistic stage without a sampling method.
            continue
        stage = model.stages[stage_index-1]
        stage_extra_arguments = next(
            (
                extra_arguments[key]
                for key in (str(stage.index), stage.index, stage.name)
                if key in extra_arguments
            ),
            {},
        )
        analysis = analyzer.analyze(
            model, stage_index=stage_index, skip_checks=skip_checks,
            stage_extra_arguments=stage_extra_arguments
        )
        only_extract_data = stage_extra_arguments.get("only_extract_data", False)
        if (analysis is not None) and (not only_extract_data):
            stage_results = analysis.print_results(print_json)
            analysis.make_image_directory(analyzer_base.DEFAULT_IMAGE_DIR)
            stage_plots = analysis.save_plots(analyzer_base.DEFAULT_IMAGE_DIR, silent=print_json)
            if print_json:
                if stage_results or stage_plots:
                    json_results[stage_index] = {"name": stage.name}
                    if stage_results:
                        json_results[stage_index]["results"] = stage_results
                    else:
                        json_results[stage_index]["results"] = {}
                    if stage_plots:
                        json_results[stage_index]["plots"] = stage_plots
                    else:
                        json_results[stage_index]["plots"] = {}
            else:
                print(f"Analysis for stage index {stage.name} (index: {stage.index})")
            
            anything_analyzed = True
        if analysis is not None:
            last_significant_analysis = analysis
    
    os.chdir(curdir)
    if print_json:
        print(json.dumps(json_results, indent=4))
        
    if (not anything_analyzed) and (not only_extract_data):
        if not print_json:
            print("No stages were analyzed.")
    # return the last stage analyzed.
    return last_significant_analysis
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of the model file for SEEKR calculation. This would "\
        "be the JSON file generated in the prepare stage.")
    argparser.add_argument(
        "-S", "--stage", dest="stage", type=str, default=None,
        help="The name or index of the stage whose simulation to analyze. "\
            "If not provided, every stage will be analyzed.")
    argparser.add_argument(
        "-s", "--skip_checks", dest="skip_checks", default=False, 
        help="By default, post-simulation checks will be run before the "\
        "analysis is started, and if the checks fail, the analysis "\
        "will not proceed. This argument bypasses those "\
        "checks and allows the analysis to proceed anyways.", 
        action="store_true")
    argparser.add_argument(
        "-e", "--extra_arguments", dest="extra_arguments", default="{}", type=str,
        help="Extra arguments to pass to the analyze_all function. This " \
        "is passed as a JSON string where the extra arguments are provided "\
        "by stage name/index.")
    argparser.add_argument(
        "-j", "--json", dest="json", action="store_true", default=False,
        help="If specified, the analysis results will be printed in JSON format.")
        
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    stage_arg = args["stage"]
    skip_checks = args["skip_checks"]
    extra_arguments_json = args["extra_arguments"]
    extra_arguments = json.loads(extra_arguments_json)
    print_json = args["json"]
    model = structures.load_model(model_file)
    if not skip_checks:
        check.check_post_simulation_all(model, silent=print_json)
    analyze_all(model, stage_arg, skip_checks, 
        extra_arguments=extra_arguments, print_json=print_json)