"""
modules/completion_criteria/openmm_interface.py

OpenMM-specific functions for completion criteria.
"""

import typing

from attrs import define

import seekr.modules.structure as structures
import seekr.modules.completion_criteria.structures as completion_criteria_structures

# TODO: fill out an object for the anchors-filled completion criteria
# TODO:  Anchors-filled criteria will not be implemented within the integrator,
#  but will use evaluations of the CVs between steps.
@define
class Base_completion_criteria_openmm_interface:
    """
    A base class for a completion criteria method using the OpenMM engine.
    """
    _completion_criteria: completion_criteria_structures.MD_completion_criteria
    def criteria_satisfied(self) -> bool:
        """
        Evaluate the current simulation state to see if the completion
        criteria are met.
        """
        # simulation, self._cv_force
        ending = False
        # if simulation.get_step_count() >= self._number_of_steps:
        #     ending = True
        # OR let the chunks handle this automatically. Don't do both!
        return ending

@define
class Number_of_steps_completion_criteria_openmm_interface(
        Base_completion_criteria_openmm_interface):
    """
    Interface for the Number-of-steps completion criteria using
    the OpenMM engine. Identical to the base class.
    """
    _completion_criteria: completion_criteria_structures.Number_of_steps_completion_criteria

@define
class All_anchors_filled_completion_criteria_openmm_interface(
        Base_completion_criteria_openmm_interface):
    """
    Interface for the All-anchors-filled completion criteria using
    the OpenMM engine. Identical to the base class.
    """
    _completion_criteria: completion_criteria_structures.All_anchors_filled_completion_criteria

    def criteria_satisfied(self) -> bool:
        # simulation, self._cv_force
        """
        Evaluate the current simulation state to see if all anchors are filled.
        """
        ending=False
        # self._cv_force.evaluate_CV(simulation.context)
        # ... or whatever this function is called
        return ending

Completion_criteria_openmm_interface = typing.Union[
    Number_of_steps_completion_criteria_openmm_interface,
    All_anchors_filled_completion_criteria_openmm_interface]

def assign_completion_interface(
        completion_criteria: completion_criteria_structures.MD_completion_criteria, 
        md_engine: structures.MD_engine
        ) -> Completion_criteria_openmm_interface:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if completion_criteria.type == "number_of_steps":
        return Number_of_steps_completion_criteria_openmm_interface(completion_criteria)
    elif completion_criteria.type == "all_anchors_filled":
        return All_anchors_filled_completion_criteria_openmm_interface(completion_criteria)
    else:
        return None