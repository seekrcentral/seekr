"""
modules/sampling_methods/openmm_interface.py

Sampling methods need to interface with different engines for
a SEEKR calculation. This module provides an "intersection"
interface for the available sampling methods here, represented
as classes that implement methods that interface with OpenMM
"""

import os
import typing

from attrs import define
import openmm
import openmm.app as openmm_app

import seekr.modules.structures as structures
import seekr.modules.engines.structure as engines_structures
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.completion_criteria.openmm_interface \
    as completion_criteria_openmm_interface
import seekr.modules.collective_variables.openmm_interface \
    as collective_variables_openmm_interface
import seekr.modules.engines.openmm.sim as sim_openmm

def make_list_from_ambig_entry(value, num_elements):
    """
    Make a list of num_elements with all elements equal to value
    """
    if isinstance(value, list):
        assert len(value) == num_elements, \
            "The value list must have the same length as the number of elements."
        return value
    else:
        return [value for i in range(num_elements)]

@define
class Base_md_openmm_interface:
    """
    A base class for a MD sampling method using the OpenMM engine.
    """
    _sampling_method: sampling_methods_structures.MD_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    def create_sim_openmm(self) -> None:
        """
        If a sampling method ever needed a special way to make an OpenMM
        simulation object, including the system, integrator, etc., this
        function would be invoked.

        Conventional MD can rely on the default method for this
        """
        return None
    
    def add_global_variables_integrator(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        All schemes will have an "abort" variable whose value will 
        depend on the completion criteria.
        """
        integrator: openmm.Integrator = kwargs["integrator"]
        integrator.addGlobalVariable("abort", 0)
        return None
    
    def add_per_DOF_variables_integrator(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def post_integration_instructions(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    # TODO: simulation itself should be passed in here
    def assign_from_sim_openmm(
            self,
            sim_openmm_obj: sim_openmm.Sim_openmm
        ) -> None:
        """
        Assign the simulation object from an OpenMM simulation object.
        """
        self._simulation_obj = sim_openmm_obj.simulation
        return None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ):
        """
        Perform any necessary preparations before the simulation starts.
        """
        return None
    
    def add_reporters(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> typing.Tuple[bool, bool]:
        """
        Add reporters to an OpenMM simulation object.
        Conventional MD method doesn't need any special reporters,
        so just rely on the standard reporters scheme by returning
        two Trues.
        """
        return True, True
    
    def run(
            self,
            steps_between_checkpoints: int,
            steps_between_evals: int,
            start_step: int,
            min_steps: int,
            max_steps: int,
            checkpoint_filename: str,
            lock_filename: str,
            completion_criteria_openmm_interface: \
                completion_criteria_openmm_interface\
                    .Base_completion_criteria_openmm_interface,
            ) -> int:
        """
        Run the simulation.
        """
        # TODO: need to save checkpoint reproducibility files.
        if steps_between_evals == 0:
            steps_between_evals = steps_between_checkpoints
        stopping = False
        num_steps = 0
        on_step = start_step
        lock_string = engines_structures.create_lock(lock_filename)
        while not stopping:
            make_checkpoint = False
            make_eval = False
            steps_since_eval = num_steps % steps_between_evals
            steps_until_eval = steps_between_evals - steps_since_eval
            steps_since_checkpoint = num_steps % steps_between_checkpoints
            steps_until_next_checkpoint = steps_between_checkpoints \
                - steps_since_checkpoint
            if steps_until_next_checkpoint < steps_until_eval:
                steps_to_take = steps_until_next_checkpoint
            else:
                steps_to_take = steps_until_eval
                make_eval = True
            self._simulation_obj.step(steps_to_take)
            num_steps += steps_to_take
            on_step += steps_to_take
            if num_steps % steps_between_checkpoints == 0:
                make_checkpoint = True
            if make_eval:
                #integrator = self._simulation_obj.context.getIntegrator()
                #abort = integrator.getGlobalVariableByName("abort")
                abort = completion_criteria_openmm_interface.criteria_satisfied(
                    self._simulation_obj)
            if make_checkpoint:
                self._simulation_obj.saveCheckpoint(checkpoint_filename)
                lock_string = engines_structures.update_lock(
                    lock_filename, lock_string)
            if on_step >= max_steps:
                stopping = True
            #elif abort > 0:
            elif abort:
                if on_step >= min_steps:
                    stopping = True
            
        self._simulation_obj.saveCheckpoint(checkpoint_filename)
        os.remove(lock_filename)
        return num_steps

@define
class Conventional_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the conventional MD sampling method using
    the OpenMM engine
    """
    _sampling_method: sampling_methods_structures.Conventional_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    
@define
class Metadynamics_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the metadynamics MD sampling method using
    the OpenMM engine
    """
    _sampling_method: sampling_methods_structures.Metadynamics_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    _metadynamics_obj: openmm_app.Metadynamics | None = None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Dict[str, typing.Any]):
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = kwargs["system"]
        model: structures.Model = kwargs["model"]
        md_engine = model.md_engine
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        num_cvs = len(self._sampling_method.cv_name_list)
        metadyn_biasfactor = self._sampling_method.bias_factor
        metadyn_height = self._sampling_method.gaussian_height
        steps_per_metadyn_update = self._sampling_method.steps_per_update
        metadyn_bias_dir = self._sampling_method.bias_dir
        bias_widths = make_list_from_ambig_entry(
            self._sampling_method.gaussian_width, num_cvs)
        grid_widths = make_list_from_ambig_entry(
            self._sampling_method.number_of_points, num_cvs)
        metaD_force_biases = []
        
        for i, cv_name in enumerate(self._sampling_method.cv_name_list):
            cv = model.get_cv_by_name(cv_name)
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                cv, md_engine)
            metaD_force = cv_interface.make_cv_force()
            #metaD_force.setForceGroup(0)
            cv_interface.add_groups_and_variables(metaD_force)
            force_bias = openmm_app.BiasVariable(
                metaD_force, minValue=cv.get_min_value(), maxValue=cv.get_max_value(),
                biasWidth=bias_widths[i], periodic=cv.is_periodic(), gridWidth=grid_widths[i])
            metaD_force_biases.append(force_bias)
        
        # TODO: Create the self._metadynamics object
        openmm_app.Metadynamics(
            system, variables=metaD_force_biases, temperature=model.temperature,
            biasFactor=metadyn_biasfactor, height=metadyn_height, 
            frequency=steps_per_metadyn_update, 
            saveFrequency=steps_per_metadyn_update, biasDir=metadyn_bias_dir)
        return None

@define
class Milestoning_md_openmm_interface:
    """
    Interface for the MMVT MD sampling method using the OpenMM engine
    """
    _sampling_method: sampling_methods_structures.Milestoning_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    def add_global_variables_integrator(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
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
    
    def add_per_DOF_variables_integrator(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        Some per-DOF variables for MetaD.
        """
        # TODO:
        #  1. Storage of position information (optionally)
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        # TODO:
        # 1. Check to make sure no bounce on first step.
        return None
    
    def post_integration_instructions(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        # TODO: 
        # 1. Obtain values of relevant force groups
        # 2. Detect bounces based on this
        # 3. If a bounce occurred, reverse velocities
        return None
    
Sampling_md_openmm_interface = typing.Union[Conventional_md_openmm_interface,
                                            Metadynamics_md_openmm_interface,
                                            Milestoning_md_openmm_interface]

def assign_sampling_method_interface(
        sampling_method: sampling_methods_structures.MD_sampling_method, 
        md_engine: engines_structures.MD_engine
        ) -> Sampling_md_openmm_interface | None:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if sampling_method.type == "conventional":
        return Conventional_md_openmm_interface(sampling_method)
    elif sampling_method.type == "metadynamics":
        return Metadynamics_md_openmm_interface(sampling_method)
    elif sampling_method.type == "milestoning":
        return Milestoning_md_openmm_interface(sampling_method)
    else:
        return None
    
# def assign_completion_criteria_interface(