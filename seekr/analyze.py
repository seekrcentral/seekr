from __future__ import annotations
"""
analyze.py

Analyze the results of SEEKR simulations for calculation of rate constants,
free energies, convergence, and other properties and plots.
"""

import os
import argparse
import logging

import seekr._version as seekr_version
import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.check as check
import seekr.modules.analyzers.base as analyzer_base
import seekr.modules.analyzers.analyzer as analyzer
import seekr.modules.analyzers.mmvt_analyzer as mmvt_analyzer
import seekr.run as run

logger = logging.getLogger(__name__)

__version__ = seekr_version.__version__

def analyze_all(
        model: structures.Seekr_model,
        stage_arg: str | None = None,
        skip_checks: bool = False,
        min_time_ps: float = 0.0,
        max_time_ps: float | None = None,
        error_analysis_method: str = "bootstrap",
        only_extract_data: bool = False,
        num_error_samples: int | None = None
    ) -> typing.Any:
    curdir = os.getcwd()
    os.chdir(model.directory)
    synopsis_filename = model.get_synopsis_filename(model.directory)
    base.configure_logging(synopsis_filename)
    if not only_extract_data:
        logger.info(
            f"Analyzing with stage instruction {stage_arg} (error_analysis_method: "
            f"{error_analysis_method}, num_error_samples: {num_error_samples}, "
            f"min_time_ps: {min_time_ps}, max_time_ps: {max_time_ps}).")
    base.log_invocation_metadata(__version__)
    if stage_arg is None:
        stages_to_analyze = [stage.index for stage in model.stages]
    else:
        stages_to_analyze = [run.extract_stage_index(model, stage_arg)]
    anything_analyzed = False
    last_significant_analysis = None
    for stage_index in stages_to_analyze:
        if not hasattr(model.stages[stage_index-1], "sampling_method"):
            # Logistic stage without a sampling method.
            continue
        analysis = analyzer.analyze(
            model, stage_index=stage_index, skip_checks=skip_checks,
            min_time_ps=min_time_ps, max_time_ps=max_time_ps,
            error_analysis_method=error_analysis_method,
            only_extract_data=only_extract_data,
            num_error_samples=num_error_samples
        )
        stage = model.stages[stage_index-1]
        if (analysis is not None) and (not only_extract_data):
            print(f"Analysis for stage index {stage.name} (index: {stage.index})")
            analysis.print_results()
            analysis.make_image_directory(analyzer_base.DEFAULT_IMAGE_DIR)
            analysis.save_plots(analyzer_base.DEFAULT_IMAGE_DIR)
            anything_analyzed = True
        if analysis is not None:
            last_significant_analysis = analysis
    os.chdir(curdir)
    if (not anything_analyzed) and (not only_extract_data):
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
        "-t", "--minimum_time", dest="minimum_time", default=0.0, type=float,
        help="A user may wish to skip an amount of simulation time for each "\
        "anchor before counting the transitions for milestoning analysis. "\
        "Enter the time (in ps) to skip a portion of the production "\
        "simulations when performing analysis.")
    argparser.add_argument(
        "-T", "--maximum_time", dest="maximum_time", default=None, type=float,
        help="A user may wish to stop the analysis of simulation time for "\
        "each anchor at a particular time. Enter the time (in ps) at which to "\
        "end the analysis at a given anchor if the simulation time exceeds it.")
    argparser.add_argument(
        "-e", "--error_analysis_method", dest="error_analysis_method", type=str,
        default="bootstrap", help="The method to use for error analysis. "\
        "Three options are currently available: 'bootstrap', "\
        "'resample', and 'montecarlo'. See online manual for explanation. "\
        "Default: 'bootstrap'.")
    argparser.add_argument(
        "-n", "--num_error_samples", dest="num_error_samples", default=None, type=int,
        help="The number of samples to use for error analysis. If not provided, " \
        "the default number of samples will be used.")
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    stage_arg = args["stage"]
    skip_checks = args["skip_checks"]
    min_time_ps = args["minimum_time"]
    max_time_ps = args["maximum_time"]
    error_analysis_method = args["error_analysis_method"]
    num_error_samples = args["num_error_samples"]
    model = structures.load_model(model_file)
    if not skip_checks:
        check.check_post_simulation_all(model)
    analyze_all(model, stage_arg, skip_checks, min_time_ps, max_time_ps, 
         error_analysis_method, num_error_samples)