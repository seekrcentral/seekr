"""
modules/sampling_methods/openmm_interface.py

Sampling methods need to interface with different engines for
a SEEKR calculation. This module provides an "intersection"
interface for the available sampling methods here, represented
as classes that implement methods that interface with OpenMM
"""

import os
import time
import typing
import logging

from attrs import define
import openmm
import openmm.app as openmm_app

import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.collective_variables.structures as collective_variables_structures
import seekr.modules.completion_criteria.openmm_interface \
    as completion_criteria_openmm_interface
import seekr.modules.collective_variables.openmm_interface \
    as collective_variables_openmm_interface

BIAS_DIR_NAME = "metad_bias"
METADYN_NAME = "metadynamics_{}_{}.pdb"
BUFFER_FRAMES = 1 # How many positions variables to store for bounce positions.

# TODO: put into a lower-level module?
def write_pdb_file_from_simulation(
        filename: str,
        simulation: openmm_app.Simulation
        ) -> None:
    """
    Write a PDB file with the current state of the system.
    """
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    with open(filename, "w") as f:
        openmm_app.PDBFile.writeFile(
            simulation.topology, positions, f, 
            keepIds=True)
    return

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
    _stage_index: int
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
    def assign_sim_openmm(
            self,
            simulation: openmm_app.Simulation
        ) -> None:
        """
        Assign the simulation object from an OpenMM simulation object.
        """
        self._simulation_obj = simulation
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
    
    def _step(
            self,
            steps: int
            ) -> None:
        """
        Take a number of steps in the simulation.
        """
        self._simulation_obj.step(steps)
        return

    def perform_eval(self):
        """
        Perform an evaluation at a certain point in the simulation.
        """
        pass

    def run(
            self,
            steps_between_checkpoints: int,
            steps_between_evals: int,
            start_step: int,
            min_steps: int,
            max_steps: int,
            restart_index: int,
            checkpoint_filename: str,
            checkpoint_info_filename: str,
            lock_filename: str,
            completion_openmm_interface: \
                completion_criteria_openmm_interface\
                    .Base_completion_criteria_openmm_interface,
            do_initial_benchmark: bool = True,
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
            abort = False
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
            if num_steps == 0 and do_initial_benchmark:
                start_time = time.time()
            #self._simulation_obj.step(steps_to_take)
            self._step(steps_to_take)
            if num_steps == 0 and do_initial_benchmark:
                total_time = time.time() - start_time
                timestep = self._simulation_obj.context.getIntegrator().getStepSize()
                ns_per_day = steps_to_take * timestep * 1e-3 \
                    * 86400 / total_time
                logging.info(f"Initial benchmark (ns/day): {ns_per_day}")
            num_steps += steps_to_take
            on_step += steps_to_take
            if num_steps % steps_between_checkpoints == 0:
                make_checkpoint = True
            if make_eval:
                #integrator = self._simulation_obj.context.getIntegrator()
                #abort = integrator.getGlobalVariableByName("abort")
                self.perform_eval()
                # NOTE: an logging update is printed every checkpoint. Is this right?
                abort = completion_openmm_interface.criteria_satisfied(
                    current_step=on_step, print_update=make_checkpoint)
            if make_checkpoint:
                self._simulation_obj.saveCheckpoint(checkpoint_filename)
                engines_structures.write_checkpoint_information_file(
                    on_step, restart_index, checkpoint_info_filename)
                lock_string = engines_structures.update_lock(
                    lock_filename, lock_string)
            if on_step >= max_steps:
                stopping = True
            #elif abort > 0:
            elif abort:
                if on_step >= min_steps:
                    stopping = True
            
        self._simulation_obj.saveCheckpoint(checkpoint_filename)
        engines_structures.write_checkpoint_information_file(
            on_step, restart_index, checkpoint_info_filename)
        os.remove(lock_filename)
        return num_steps

@define
class Conventional_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the conventional MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    _sampling_method: sampling_methods_structures.Conventional_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    
@define
class Metadynamics_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the metadynamics MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    _sampling_method: sampling_methods_structures.Metadynamics_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    _metadynamics_obj: openmm_app.Metadynamics | None = None
    _openmm_cv_list: list[collective_variables_structures.Collective_variable] = []
    _model: structures.Seekr_model | None = None
    _current_alpha: int | None = None
    _directory: str | None = None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Dict[str, typing.Any]):
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = kwargs["system"]
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        directory: str = kwargs["directory"]
        self._directory = directory
        md_engine = model.md_engine
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        for i, cv in enumerate(model.collective_variables):
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                cv, md_engine)
            cv_force = cv_interface.make_cv_force()
            cv_interface.add_cv_groups_and_variables()
            cv_name = f"0*x_{i}"
            openmm_cv_object = openmm.CustomCVForce(cv_name)
            openmm_cv_object.addCollectiveVariable(cv_name, cv_force)
            freeGroups = set(range(32)) - set(force.getForceGroup() for force in system.getForces())
            if len(freeGroups) == 0:
                raise RuntimeError('Cannot assign a force group to the metadynamics stage. '
                                'The maximum number (32) of the force groups is already used.')
            openmm_cv_object.setForceGroup(max(freeGroups))
            system.addForce(openmm_cv_object)
            self._openmm_cv_list.append(openmm_cv_object)

        boosted_cvs: list[collective_variables_structures.Collective_variable] = []
        if len(self._sampling_method.cv_names) == 0:
            # Then use all CVs
            for cv in model.collective_variables:
                boosted_cvs.append(cv)
        else:
            for cv_name in self._sampling_method.cv_names:
                cv = model.get_cv_by_name(cv_name)
                boosted_cvs.append(cv)
        assert len(boosted_cvs) > 0, \
            "No CVs were provided for the metadynamics method."
        assert len(boosted_cvs) <= 3, \
            "Too many CVs were provided for the metadynamics method."
        num_boosted_cvs = len(boosted_cvs)
        metadyn_biasfactor = self._sampling_method.bias_factor
        metadyn_height = self._sampling_method.gaussian_height
        steps_per_metadyn_update = self._sampling_method.steps_per_update
        bias_widths = make_list_from_ambig_entry(
            self._sampling_method.gaussian_width, num_boosted_cvs)
        grid_widths = make_list_from_ambig_entry(
            self._sampling_method.number_of_points, num_boosted_cvs)
        if self._sampling_method.write_biases:
            metadyn_bias_dir = os.path.join(directory, BIAS_DIR_NAME)
            save_frequency = steps_per_metadyn_update
            if not os.path.exists(metadyn_bias_dir):
                os.mkdir(metadyn_bias_dir)
        else:
            metadyn_bias_dir = None
            save_frequency = None
        metaD_force_biases = []
        for i, cv in enumerate(boosted_cvs):
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                cv, md_engine)
            metaD_force = cv_interface.make_cv_force()
            #metaD_force.setForceGroup(0)
            cv_interface.add_cv_groups_and_variables()
            force_bias = openmm_app.BiasVariable(
                metaD_force, minValue=cv.get_min_value(), maxValue=cv.get_max_value(),
                biasWidth=bias_widths[i], periodic=cv.is_periodic(), gridWidth=grid_widths[i])
            metaD_force_biases.append(force_bias)
        
        # TODO: Create the self._metadynamics object
        self._metadynamics_obj = openmm_app.Metadynamics(
            system, variables=metaD_force_biases, temperature=model.temperature,
            biasFactor=metadyn_biasfactor, height=metadyn_height, 
            frequency=steps_per_metadyn_update, 
            saveFrequency=save_frequency, biasDir=metadyn_bias_dir)
        self._simulation_obj.context.reinitialize(preserveState=True)
        return None
    
    def perform_eval(self):
        """
        Perform an evaluation at a certain point in the simulation.
        """
        if self._model.anchors is None:
            return
        cv_values: list[float] = []
        for i, cv in enumerate(self._openmm_cv_list):
            cv_value = cv.getCollectiveVariableValues(self._simulation_obj.context)
            cv_values.append(cv_value[0])

        found_anchor = False
        for alpha, anchor in enumerate(self._model.anchors):
            in_anchor = True
            for milestone in anchor.milestones:
                curdir = os.getcwd()
                os.chdir(self._model.directory)
                result = structures.check_values_within_voronoi_boundary(
                    cv_values, milestone.variables)
                os.chdir(curdir)
                if not result:
                    in_anchor = False

            if in_anchor:
                assert not found_anchor, "Found system in two "\
                    "different anchors in the same step."
                found_anchor = True
                if alpha == self._current_alpha:
                    continue
                self._current_alpha = alpha
                current_anchor = self._model.anchors[alpha]
                assert current_anchor.stage_parameters[self._stage_index+1] is not None
                anchor_directory = os.path.join(
                    self._model.directory, current_anchor.directory,
                    current_anchor.stage_parameters[self._stage_index+1]["directory"])
                # Save this anchor's new structure
                if not self._model.using_toy():
                    var_string = current_anchor.make_var_string()
                    output_pdb_basename = METADYN_NAME.format(var_string, 0)
                    output_pdb_filename = os.path.join(anchor_directory, 
                                                        output_pdb_basename)
                    current_anchor.change_pdb_filename(output_pdb_basename, self._stage_index)
                    write_pdb_file_from_simulation(output_pdb_filename, self._simulation_obj)
                    
                else: # TODO: fill this in later
                    pass

    def _step(
            self,
            steps: int
            ) -> None:
        """
        Take a number of steps in the simulation.
        """
        self._metadynamics_obj.step(self._simulation_obj, steps)
        return

@define
class Milestoning_md_openmm_interface:
    """
    Interface for the MMVT MD sampling method using the OpenMM engine
    """
    _stage_index: int
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
        integrator: openmm.System = kwargs["integrator"]
        # TODO: make anchor and/or integrator an attribute of this class?
        anchor: structures.Anchor = kwargs["anchor"]
        # TODO: create variables to handle bootstrapping with replacement.
        #num_included_milestones = 0
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = milestone.alias_index
            if milestone.boundary:
                #integrator.addGlobalVariable(f"milestone_{num_included_milestones}_alias_index", alias_index_i)
                integrator.addGlobalVariable(f"value_{alias_index_i}", 0)
                #num_included_milestones += 1
                for key, value in milestone.variables.items():
                    integrator.addGlobalVariable(f"milestone_{alias_index_i}_{key}", value)
                for j, milestone2 in enumerate(anchor.milestones):
                    alias_index_j = milestone2.alias_index
                    if milestone2.boundary:
                        integrator.addGlobalVariable(f"milestone_{alias_index_i}_{alias_index_j}_count", 0)
                integrator.addGlobalVariable(f"milestone_{alias_index_i}_time", 0)
        
        #integrator.addGlobalVariable("num_milestones", num_included_milestones)
        integrator.addGlobalVariable("step_count", 0)
        integrator.addGlobalVariable("buffer_frames", BUFFER_FRAMES)
        integrator.addGlobalVariable("last_milestone_alias_touched", -1)
        integrator.addGlobalVariable("last_milestone_time_touched", -1)
        integrator.addGlobalVariable("first_crossing_time", -1)
        integrator.addGlobalVariable("first_step_bounce", 0)

        return None
    
    def add_per_DOF_variables_integrator(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        Some per-DOF variables for MetaD.
        """
        
        integrator: openmm.System = kwargs["integrator"]
        # TODO:
        #  1. Storage of position information (optionally)
        for i in range(BUFFER_FRAMES):
            integrator.addPerDofVariable(f"bounce_pos_{i}", 0)
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Dict[str, typing.Any]
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        
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
        integrator: openmm.System = kwargs["integrator"]
        anchor: structures.Anchor = kwargs["anchor"]
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = milestone.alias_index
            integrator.addComputeGlobal(f"value_{alias_index_i}", 
                                        f"energy{alias_index_i}")
        
        # Check to make sure no bounce on first step.
        integrator.beginIfBlock("step_count = 0")
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = i+1
            integrator.beginIfBlock(f"value_{alias_index_i} > 0")
            integrator.addComputeGlobal("first_step_bounce", "1")
            integrator.endBlock()
        integrator.endBlock()

        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = i+1
            integrator.beginIfBlock(f"value_{alias_index_i} > 0")
            
            integrator.beginIfBlock(f"last_milestone_alias_touched > -1")
            for j, milestone2 in enumerate(anchor.milestones):
                alias_index_j = j+1
                integrator.beginIfBlock(f"last_milestone_alias_touched = {alias_index_j}")
                integrator.addComputeGlobal(f"milestone_{alias_index_i}_{alias_index_j}_count", f"milestone_{alias_index_i}_{alias_index_j}_count + 1")
                integrator.endBlock()
            integrator.addComputeGlobal(f"milestone_{alias_index_i}_steps", f"milestone_{alias_index_i}_steps + step_count - last_milestone_step_touched")
            integrator.addComputeGlobal("last_milestone_step_touched", "step_count")
            integrator.endBlock() # End If last_milestone_alias_touched > -1
            
            integrator.beginIfBlock("last_milestone_alias_touched = -1")
            integrator.addComputeGlobal("last_milestone_alias_touched", f"{alias_index_i}")
            integrator.addComputeGlobal("last_milestone_step_touched", "step_count")
            integrator.endBlock() # End If last_milestone_alias_touched = -1
            
            integrator.endBlock() # End If value_{alias_index_i} > 0

        integrator.addComputeGlobal("step_count", "step_count + 1")

        return None
    
Sampling_md_openmm_interface = typing.Union[Conventional_md_openmm_interface,
                                            Metadynamics_md_openmm_interface,
                                            Milestoning_md_openmm_interface]

def assign_sampling_method_interface(
        stage_index: int,
        sampling_method: sampling_methods_structures.MD_sampling_method, 
        md_engine: structures.MD_engine
        ) -> Sampling_md_openmm_interface | None:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if sampling_method.type == "conventional":
        return Conventional_md_openmm_interface(stage_index, sampling_method)
    elif sampling_method.type == "metadynamics":
        return Metadynamics_md_openmm_interface(stage_index, sampling_method)
    elif sampling_method.type == "milestoning":
        return Milestoning_md_openmm_interface(stage_index, sampling_method)
    else:
        return None
    
# def assign_completion_criteria_interface(