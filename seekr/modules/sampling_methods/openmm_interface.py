"""
modules/sampling_methods/openmm_interface.py

Sampling methods need to interface with different engines for
a SEEKR calculation. This module provides an "intersection"
interface for the available sampling methods here, represented
as classes that implement methods that interface with OpenMM
"""

import typing

from attrs import define

import modules.sampling_methods.base as sampling_methods_base
import modules.engines.data_structures as engines_data_struct

# TODO: replace "integrator" argument in many of the methods below
#  with generic *args and **kwargs.

@define
class Conventional_md_openmm_interface:
    """
    Interface for the conventional MD sampling method using
    the OpenMM engine
    """
    def create_sim_openmm(self) -> None:
        """
        If a sampling method ever needed a special way to make an OpenMM
        simulation object, including the system, integrator, etc., this
        function would be invoked.

        Conventional MD can rely on the default method for this
        """
        return None
    
    def add_global_variables_integrator(self, integrator) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def add_per_DOF_variables_integrator(self, integrator) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def pre_integration_instructions(self, integrator) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def post_integration_instructions(self, integrator) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def pre_simulation_preparations(self):
        """
        Perform any necessary preparations before the simulation starts.
        """
        return None
    
    def add_reporters(
            self, 
            simulation) -> typing.Tuple[bool, bool]:
        """
        Add reporters to an OpenMM simulation object.
        Conventional MD method doesn't need any special reporters,
        so just rely on the standard reporters scheme by returning
        three Trues.
        """
        return True, True
    
    def run(self):
        """
        Run the simulation.
        """
        # TODO: fill out next time!
        return None
    
@define
class Metadynamics_md_openmm_interface:
    """
    Interface for the metadynamics MD sampling method using
    the OpenMM engine
    """
    _metadynamics_obj: openmm_app.Metadynamics

    def create_sim_openmm(self) -> None:
        """
        If a sampling method ever needed a special way to make an OpenMM
        simulation object, including the system, integrator, etc., this
        function would be invoked.

        MetaD can rely on the default method for this
        """
        return None
    
    def add_global_variables_integrator(self, integrator) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        None for MetaD.
        """
        return None
    
    def add_per_DOF_variables_integrator(self, integrator) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        None for MetaD.
        """
        return None
    
    def pre_integration_instructions(self, integrator) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        None for MetaD.
        """
        return None
    
    def post_integration_instructions(self, integrator) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        None for MetaD.
        """
        return None
    
    def pre_simulation_preparations(self):
        """
        Perform any necessary preparations before the simulation starts.
        """
        # TODO: Create the self._metadynamics object
        return None

    def add_reporters(
            self, 
            simulation) -> typing.Tuple[bool, bool]:
        """
        Add reporters to an OpenMM simulation object.
        metaD method doesn't need any special reporters,
        so just rely on the standard reporters scheme by returning
        three Trues.
        """
        return True, True
    
@define
class Milestoning_md_openmm_interface:
    def create_sim_openmm(self) -> None:
        """
        If a sampling method ever needed a special way to make an OpenMM
        simulation object, including the system, integrator, etc., this
        function would be invoked.

        Milestoning can rely on the default method for this
        """
        return None
    
    def add_global_variables_integrator(self, integrator) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        Some global variables for milestoning method.
        """
        # TODO:
        #  1. Force groups representing milestones
        #  2. Whether bounces have occurred in this interval, and which ones.
        #     Have enough ready to store for all steps within an interval.
        #  3. Intervals between bounce checks.
        
        return None
    
    def add_per_DOF_variables_integrator(self, integrator) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        Some per-DOF variables for MetaD.
        """
        # TODO:
        #  1. Storage of position information (optionally)
        return None
    
    def pre_integration_instructions(self, integrator) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        # TODO:
        # 1. Check to make sure no bounce on first step.
        return None
    
    def post_integration_instructions(self, integrator) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        # TODO: 
        # 1. Obtain values of relevant force groups
        # 2. Detect bounces based on this
        # 3. If a bounce occurred, reverse velocities
        return None
    
    def pre_simulation_preparations(self):
        """
        Perform any necessary preparations before the simulation starts.
        """
        return None

    def add_reporters(
            self, 
            simulation) -> typing.Tuple[bool, bool]:
        """
        Add reporters to an OpenMM simulation object.
        milestoning method doesn't need any special reporters,
        so just rely on the standard reporters scheme by returning
        three Trues.
        """
        return True, True

Sampling_md_openmm_interface = typing.Union[Conventional_md_openmm_interface,
                                            Metadynamics_md_openmm_interface,
                                            Milestoning_md_openmm_interface]

def assign_interface(
        sampling_method: sampling_methods_base.MD_sampling_method, 
        md_engine: engines_data_struct.MD_engine
        ) -> Sampling_md_openmm_interface:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if sampling_method.type == "conventional":
        return Conventional_md_openmm_interface()
    elif sampling_method.type == "metadynamics":
        return Metadynamics_md_openmm_interface()
    elif sampling_method.type == "milestoning":
        return Milestoning_md_openmm_interface()
    else:
        raise ValueError(
            "The sampling method type is not recognized by the "\
            f"openmm_interface module: {sampling_method.type}.")