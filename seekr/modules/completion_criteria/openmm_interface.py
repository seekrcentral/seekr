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
import seekr.modules.completion_criteria.runner as completion_criteria_runner

# TODO: fill out an object for the anchors-filled completion criteria
# TODO:  Anchors-filled criteria will not be implemented within the integrator,
#  but will use evaluations of the CVs between steps.
@define
class Base_completion_criteria_openmm_interface:
    """
    A base class for a completion criteria method using the OpenMM engine.
    """
    _completion_criteria: completion_criteria_structures.Base_completion_criteria
    _model: structures.Seekr_model
    _stage: structures.Stage_md
    _anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor

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
    _stage: structures.Stage_md

@define
class All_anchors_filled_completion_criteria_openmm_interface(
        Base_completion_criteria_openmm_interface):
    """
    Interface for the All-anchors-filled completion criteria using
    the OpenMM engine. Identical to the base class.
    """
    _completion_criteria: completion_criteria_structures.All_anchors_filled_completion_criteria
    _model: structures.Seekr_model
    _stage: structures.Stage_md
    _anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor

    def criteria_satisfied(
            self,
            **kwargs: typing.Any
            ) -> bool:
        # simulation, self._cv_force
        """
        Evaluate the current simulation state to see if all anchors are filled.
        """
        #model: structures.Seekr_model = kwargs["model"]
        #stage: structures.Stage_md = kwargs["stage"]
        print_update: bool = kwargs["print_update"]
        current_step: int = kwargs["current_step"]
        return_message: bool
        if print_update:
            ending, message = completion_criteria_runner.is_finished_md_with_message(
                self._model, self._stage, self._anchor_or_systemwide, current_step)
            logging.info(message)
        else:
            ending = completion_criteria_runner.is_finished_md(
                self._model, self._stage, self._anchor_or_systemwide, current_step)

        return ending

Completion_criteria_openmm_interface = typing.Union[
    Number_of_steps_completion_criteria_openmm_interface,
    All_anchors_filled_completion_criteria_openmm_interface]

def assign_completion_interface(
        completion_criteria: completion_criteria_structures.Base_completion_criteria, 
        model: structures.Seekr_model,
        stage: structures.Stage_md,
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | structures.Anchor
        )-> Base_completion_criteria_openmm_interface:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    md_engine = model.md_engine
    assert md_engine is not None
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if completion_criteria.type == "number_of_steps":
        return Number_of_steps_completion_criteria_openmm_interface(
            completion_criteria, model, stage, anchor_or_systemwide)
    elif completion_criteria.type == "all_anchors_filled":
        return All_anchors_filled_completion_criteria_openmm_interface(
            completion_criteria, model, stage, anchor_or_systemwide)
    else:
        # Run through plugins
        for plugin_name in model.plugins:
            compl_crit_module_str = base.get_plugin_str(plugin_name, submodule_name="completion_criteria")
            if compl_crit_module_str is not None:
                plugin_module_compl_crit_interface = importlib.import_module(
                    compl_crit_module_str+".openmm_interface")
            found, compl_crit_interface = plugin_module_compl_crit_interface.assign_completion_interface(
                completion_criteria, model, stage, anchor_or_systemwide)
            if found:
                return compl_crit_interface

    raise Exception("Completion criteria not found: " + completion_criteria.type)