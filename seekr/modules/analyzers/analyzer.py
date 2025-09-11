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
        minimum_time: float = 0.0,
        maximum_time: float | None = None,
        error_analysis_method: str = "bootstrap"
    ) -> mmvt_analyzer.Mmvt_analyzer:
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
    min_step = int(minimum_time / model.get_md_timestep())
    if maximum_time is None:
        max_step = np.iinfo(np.int64).max
    else:
        max_step = int(maximum_time / model.get_md_timestep())
    if stage.sampling_method.type == "mmvt":
        from seekr.modules.analyzers.mmvt_analyzer import analyze_mmvt
        analysis = analyze_mmvt(
            model, stage_index=stage_index, skip_checks=skip_checks,
            min_step=min_step, max_step=max_step,
            error_analysis_method=error_analysis_method)
        return analysis
    # TODO: metaD and maybe SMD?
    # TODO: cycle through plugins?
    else:
        raise ValueError(f"Analysis for sampling method type not yet implemented: "
                         f"{stage.sampling_method.type}")