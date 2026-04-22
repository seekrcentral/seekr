"""
modules/completion_criteria/openmm_interface.py

OpenMM-specific functions for completion criteria.
"""

import typing
import logging
import importlib

from attrs import define

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as completion_criteria_structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.scales.molecular_dynamics.runner as md_runner
import seekr.modules.reporters.openmm_interface as reporters_openmm_interface

@define
class Base_completion_criteria_openmm_interface:
    """
    A base class for a completion criteria method using the OpenMM engine.
    """
    _completion_criteria: completion_criteria_structures.Base_completion_criteria
    _model: structures.Seekr_model
    _stage: md_structures.Stage_md
    _anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings | structures.Anchor

    def criteria_satisfied(
            self,
            **kwargs: typing.Any
            ) -> bool:
        """
        Evaluate the current simulation state to see if the completion
        criteria are met.
        """
        # NOTE: letting the chunks handle this automatically.
        ending = False
        return ending
    
    def needs_cv_values(self) -> bool:
        """
        Check if the completion criteria requires CV values to be evaluated.
        """
        return False

@define
class Number_of_steps_completion_criteria_openmm_interface(
        Base_completion_criteria_openmm_interface):
    """
    Interface for the Number-of-steps completion criteria using
    the OpenMM engine. Identical to the base class.
    """
    _completion_criteria: completion_criteria_structures.Number_of_steps_completion_criteria
    _model: structures.Seekr_model
    _stage: md_structures.Stage_md

@define
class Reporter_progress_completion_criteria_openmm_interface(
        Base_completion_criteria_openmm_interface):
    """
    Interface for the Reporter-progress completion criteria using
    the OpenMM engine.
    """
    _completion_criteria: completion_criteria_structures.Reporter_progress_completion_criteria
    _model: structures.Seekr_model
    _stage: md_structures.Stage_md
    _anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings | structures.Anchor
    _reporter_interface: reporters_openmm_interface.Base_reporter_openmm_interface

    def criteria_satisfied(
            self,
            **kwargs: typing.Any
            ) -> bool:
        """
        Evaluate the current simulation state to check reporter progress.
        Queries the reporter interface directly for in-memory progress, then
        checks against the threshold and max_steps bail-out.
        """
        print_update: bool = kwargs["print_update"]
        current_step: int = kwargs["current_step"]
        progress = self._reporter_interface.get_progress()
        threshold = self._completion_criteria.reporter_progress_threshold
        max_steps = self._completion_criteria.max_number_of_steps
        ending = (progress >= threshold) or (current_step >= max_steps)
        if print_update:
            logging.info(
                f"reporter_progress: {progress:.3f} / {threshold:.3f} "
                f"(current_step: {current_step})")
        return ending

Completion_criteria_openmm_interface = typing.Union[
    Number_of_steps_completion_criteria_openmm_interface,
    Reporter_progress_completion_criteria_openmm_interface]

def assign_completion_interface(
        completion_criteria: completion_criteria_structures.Base_completion_criteria, 
        model: structures.Seekr_model,
        stage: md_structures.Stage_md,
        anchor_or_unpartitioned: engines_structures.MD_unpartitioned_settings | structures.Anchor,
        reporter_interfaces: list | None = None,
        )-> Base_completion_criteria_openmm_interface:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    md_engine = model.get_engine_by_scale_type(stage.scale_type)
    assert md_engine is not None
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if completion_criteria.type == "number_of_steps":
        return Number_of_steps_completion_criteria_openmm_interface(
            completion_criteria, model, stage, anchor_or_unpartitioned)
    elif completion_criteria.type == "reporter_progress":
        reporter_interface = None
        for this_reporter_interface in reporter_interfaces:
            if this_reporter_interface._reporter.name \
                    == completion_criteria.reporter_name:
                reporter_interface = this_reporter_interface
                break
        assert reporter_interface is not None, \
            f"Reporter {completion_criteria.reporter_name} not found."
        return Reporter_progress_completion_criteria_openmm_interface(
            completion_criteria, model, stage, anchor_or_unpartitioned,
            reporter_interface)
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            compl_crit_module_str = base.get_plugin_str(plugin_name, submodule_name="completion_criteria")
            if compl_crit_module_str is not None:
                plugin_module_compl_crit_interface = importlib.import_module(
                    compl_crit_module_str+".openmm_interface")
            found, compl_crit_interface = plugin_module_compl_crit_interface.assign_completion_interface(
                completion_criteria, model, stage, anchor_or_unpartitioned)
            if found:
                return compl_crit_interface

    raise Exception("Completion criteria not found: " + completion_criteria.type)