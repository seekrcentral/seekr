from __future__ import annotations
"""
modules/analyzers/analyzer.py

The general hub where the correct analyzer is selected based on the model type,
the analysis is performed, and results are returned in the form of an analysis
object.
"""

import importlib

import numpy as np

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.analyzers.mmvt_analyzer as mmvt_analyzer

AVAILABLE_ANALYZERS = ["mmvt", "metadynamics"]

def analyze(
        model: structures.Seekr_model,
        stage_index: int,
        skip_checks: bool = False,
        min_time_ps: float = 0.0,
        max_time_ps: float | None = None,
        error_analysis_method: str = "bootstrap",
        only_extract_data: bool = False,
        num_error_samples: int | None = None
    ) -> typing.Any:
    """
    Analyze the results of SEEKR simulations for calculation of rate constants,
    free energies, convergence, and other properties and plots.
    
    Parameters
    ----------
    model : structures.Seekr_model
        The SEEKR model to analyze.
    
    Returns
    -------
    
    """
    available_analyzers = []
    available_analyzers += AVAILABLE_ANALYZERS
    for plugin_name in model.plugins:
        analyzers_module_str = base.get_plugin_str(
            plugin_name, submodule_name="analyzers", require_structures=False,
            require_openmm_interface=False)
        if analyzers_module_str is not None:
            plugin_module_analyzer = importlib.import_module(
                analyzers_module_str+".analyzer")
            if hasattr(plugin_module_analyzer, "AVAILABLE_ANALYZERS"):
                available_analyzers += plugin_module_analyzer.AVAILABLE_ANALYZERS

    stage = model.stages[stage_index-1]
    stage.load_sets_into_model(model)
    assert hasattr(stage, "sampling_method"), \
        f"Stage {stage_index} has no sampling method set - cannot analyze."
    if stage.sampling_method.type == "mmvt":
        from seekr.modules.analyzers.mmvt_analyzer import analyze_mmvt
        analysis = analyze_mmvt(
            model, stage_index=stage_index, skip_checks=skip_checks,
            min_time_ps=min_time_ps, max_time_ps=max_time_ps,
            error_analysis_method=error_analysis_method,
            only_extract_data=only_extract_data,
            num_error_samples=num_error_samples
        )
        return analysis
    elif stage.sampling_method.type == "metadynamics":
        from seekr.modules.analyzers.metadynamics_analyzer \
            import analyze_metadynamics
        analysis = analyze_metadynamics(
            model, stage_index=stage_index, skip_checks=skip_checks,
            min_time_ps=min_time_ps, max_time_ps=max_time_ps,
            error_analysis_method=error_analysis_method,
            only_extract_data=only_extract_data,
            num_error_samples=num_error_samples
        )
        return analysis
    
    for plugin_name in model.plugins:
            analyzers_module_str = base.get_plugin_str(
                plugin_name, submodule_name="analyzers", require_structures=False,
                require_openmm_interface=False)
            if analyzers_module_str is not None:
                plugin_module_analyzer = importlib.import_module(
                    analyzers_module_str+".analyzer")
            found, analysis = plugin_module_analyzer.analyze(
                model, stage_index=stage_index, skip_checks=skip_checks,
                min_time_ps=min_time_ps, max_time_ps=max_time_ps,
                error_analysis_method=error_analysis_method,
                only_extract_data=only_extract_data,
                num_error_samples=num_error_samples
            )
            if found:
                return analysis
            
    #raise ValueError(f"Analysis for sampling method type not yet implemented: "
    #                     f"{stage.sampling_method.type}")
    return None