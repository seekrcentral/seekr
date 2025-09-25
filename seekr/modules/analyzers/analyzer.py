"""
modules/analyzers/analyzer.py

The general hub where the correct analyzer is selected based on the model type,
the analysis is performed, and results are returned in the form of an analysis
object.
"""

import numpy as np

import seekr.modules.structures as structures
import seekr.modules.analyzers.mmvt_analyzer as mmvt_analyzer

AVAILABLE_ANALYZERS = ["mmvt"]

# TODO: correct typing return here
def analyze(
        model: structures.Seekr_model,
        stage_index: int | None = None,
        skip_checks: bool = False,
        min_time_ps: float = 0.0,
        max_time_ps: float | None = None,
        error_analysis_method: str = "bootstrap",
        only_extract_data: bool = False
    ) -> mmvt_analyzer.Mmvt_analyzer | None:
    # TODO: create a more specific type for the return value above
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
    # If stage_index is not provided, automatically select the last stage.
    if stage_index is None:
        assert len(model.stages) > 0, "Model has no stages to analyze."
        for test_stage_index in range(len(model.stages) - 1, -1, -1):
            if model.stages[test_stage_index].sampling_method.type in AVAILABLE_ANALYZERS:
                stage_index = test_stage_index+1
                break
    assert 1 <= stage_index < len(model.stages)+1, "Invalid stage index."
    stage = model.stages[stage_index-1]
    if stage.sampling_method.type not in AVAILABLE_ANALYZERS:
        return None
    if max_time_ps is None:
        max_time_ps = np.finfo(np.float32).max
    if stage.sampling_method.type == "mmvt":
        from seekr.modules.analyzers.mmvt_analyzer import analyze_mmvt
        analysis = analyze_mmvt(
            model, stage_index=stage_index, skip_checks=skip_checks,
            min_time_ps=min_time_ps, max_time_ps=max_time_ps,
            error_analysis_method=error_analysis_method,
            only_extract_data=only_extract_data
        )
        return analysis
    # TODO: metaD and maybe SMD?
    # TODO: cycle through plugins?
    else:
        raise ValueError(f"Analysis for sampling method type not yet implemented: "
                         f"{stage.sampling_method.type}")