"""
modules/sampling_methods/openmm_interface.py

Sampling methods need to interface with different engines for
a SEEKR calculation. This module provides an "intersection"
interface for the available sampling methods here, represented
as classes that implement methods that interface with OpenMM
"""

import os
import time
import glob
import json
import typing
import logging
import importlib

from attrs import define
import numpy as np
import mdtraj
import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.base as engines_base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.collective_variables.structures as collective_variables_structures
import seekr.modules.scales.molecular_dynamics.runner as md_runner
import seekr.modules.sampling_methods.runner as sampling_methods_runner
import seekr.modules.completion_criteria.openmm_interface \
    as completion_criteria_openmm_interface
import seekr.modules.collective_variables.openmm_interface \
    as collective_variables_openmm_interface

BIAS_DIR_NAME = "metad_bias"
#METADYN_NAME = "metadynamics.{}.swarm_{}"
#SMD_NAME = "steered.{}.swarm_{}"
BUFFER_FRAMES = 1 # How many positions variables to store for encounter positions.
INITIAL_BENCHMARK_STEPS = 100 # Number of steps to take for the initial benchmark.
NUM_SMD_WINDOWS = 10 # Number of windows to use for SMD.

# TODO: put into a lower-level module?
def open_pdb_file_for_simulation(
        filename: str,
        simulation: openmm_app.Simulation
        ) -> None:
    """
    Open a PDB file with the current state of the system.
    """
    pdb_obj = openmm_app.PDBFile(filename)
    box_vectors = pdb_obj.topology.getPeriodicBoxVectors()
    simulation.context.setPositions(pdb_obj.getPositions())
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    return

def open_txt_file_for_simulation(
        filename: str,
        simulation: openmm_app.Simulation
        ) -> None:
    """
    Open a TXT file with the current state of the system.
    """
    positions = np.loadtxt(filename, ndmin=2)
    #positions = positions * unit.nanometers
    simulation.context.setPositions(positions)

def write_pdb_file_from_simulation(
        filename: str,
        simulation: openmm_app.Simulation,
        image_molecules: bool = True
        ) -> None:
    """
    Write a PDB file with the current state of the system.
    """
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    if image_molecules:
        mdtraj_top = mdtraj.Topology.from_openmm(simulation.topology)
        positions_np = np.array(positions.value_in_unit(unit.nanometers), dtype=np.float32)
        traj = mdtraj.Trajectory(
            xyz=positions_np[np.newaxis, :, :], 
            topology=mdtraj_top)
        traj.unitcell_vectors=np.array(
            box_vectors.value_in_unit(unit.nanometers), dtype=np.float32).T[np.newaxis, :, :]
        traj.image_molecules(inplace=True)
        positions = traj.xyz[0] * unit.nanometers

    with open(filename, "w") as f:
        openmm_app.PDBFile.writeFile(
            simulation.topology, positions, f, 
            keepIds=True)
    return

def write_txt_file_from_simulation(
        filename: str,
        simulation: openmm_app.Simulation
        ) -> None:
    """
    Write a TXT file with the current state of the system.
    """
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=False)
    positions = state.getPositions().value_in_unit(unit.nanometers)
    np.savetxt(filename, positions)
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
    sampling_method: sampling_methods_structures.Base_md_sampling_method
    _system_obj: openmm.System | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    _openmm_cv_list: list[typing.Any] | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _openmm_cv_object: typing.Any | None = None
    _fail_on_save_to_current_anchor_if_outside_milestones: bool = True

    # TODO: return a list of names?
    def create_openmm_cv_list(self) -> None:
        """
        Create a list of OpenMM CVs for the simulation.
        This can be called, if need be, by a sampling method, or on behalf of a
        completion criteria.
        """
        md_engine = self._model.get_engine_by_scale_type("molecular_dynamics")
        if self._cv_interface_list is not None:
            return None
        elif self._openmm_cv_list is not None:
            return None
        else:
            self._cv_interface_list = []
            self._openmm_cv_list = []
            for i, cv in enumerate(self._model.collective_variables):
                cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                    cv, md_engine, self._model.directory, self._model.plugins)
                assert cv_interface is not None
                self._cv_interface_list.append(cv_interface)
                cv_force = cv_interface.make_cv_force()
                cv_interface.add_cv_groups_and_variables(cv_force)
                self._openmm_cv_list.append(cv_force)

        # else, do nothing - it's already filled out.
        return

    def create_openmm_cv_object(self) -> None:
        """

        """
        if self._openmm_cv_object is not None:
            return None
        else:
            cv_names = []
            for i, openmm_cv in enumerate(self._openmm_cv_list):
                cv_name = f"cv_{i}"
                cv_names.append(cv_name)
            expression = "0.0*(" + " + ".join(cv_names) + ")"
            openmm_cv_object = openmm.CustomCVForce(expression)
            for i, openmm_cv in enumerate(self._openmm_cv_list):
                cv_name = cv_names[i]
                openmm_cv_object.addCollectiveVariable(cv_name, openmm_cv)
            
            self._openmm_cv_object = openmm_cv_object
            freeGroups = set(range(32)) - set(force.getForceGroup() for force in self._system_obj.getForces())
            if len(freeGroups) == 0:
                raise RuntimeError('Cannot assign a force group to the OpenMM CV object. '
                                'The maximum number (32) of the force groups is already used.')
            openmm_cv_object.setForceGroup(max(freeGroups))
            self._system_obj.addForce(openmm_cv_object)
            self._simulation_obj.context.reinitialize(preserveState=True)
            return None

    def create_sim_openmm(self) -> None:
        """
        If a sampling method ever needed a special way to make an OpenMM
        simulation object, including the system, integrator, etc., this
        function would be invoked.

        Conventional MD can rely on the default method for this
        """
        return
    
    def add_global_variables_integrator(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        All schemes will have an "abort" variable whose value will 
        depend on the completion criteria.
        """
        integrator: openmm.Integrator = kwargs["integrator"]
        integrator.addGlobalVariable("abort", 0)
        return
    
    def add_per_DOF_variables_integrator(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return
    
    def post_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return
    
    def assign_system_openmm(
            self,
            system: openmm.System
        ) -> None:
        """
        Assign the system object from an OpenMM system object.
        """
        self._system_obj = system
        return

    # TODO: simulation itself should be passed in here
    def assign_sim_openmm(
            self,
            simulation: openmm_app.Simulation
        ) -> None:
        """
        Assign the simulation object from an OpenMM simulation object.
        """
        self._simulation_obj = simulation
        return

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ):
        """
        Perform any necessary preparations before the simulation starts.
        """
        return
    
    def _step(
            self,
            steps: int
            ) -> None:
        """
        Take a number of steps in the simulation.
        """
        assert self._simulation_obj is not None
        self._simulation_obj.step(steps)
        return

    def perform_eval(
            self, 
            _restart_index: int, 
            _swarm_id: int
            ) -> bool:
        """
        Perform an evaluation at a certain point in the simulation. If returning
        True, the simulation will stop.
        """
        return False

    def perform_early_simulation_eval(
            self, 
            restart_index: int, 
            swarm_id: int
            ) -> None:
        """
        Perform an evaluation once at an early point in the simulation.
        """
        pass
    
    def get_cv_values(
            self,
            ) -> list | None:
        cv_values = self._openmm_cv_object.getCollectiveVariableValues(
            self._simulation_obj.context)
        return cv_values

    def run(
            self,
            steps_between_checkpoints: int,
            steps_between_evals: int,
            start_step: int,
            min_steps: int,
            max_steps: int,
            restart_index: int,
            swarm_id: int,
            checkpoint_filename: str,
            checkpoint_info_filename: str,
            lock_filename: str,
            completion_openmm_interface: \
                completion_criteria_openmm_interface\
                    .Completion_criteria_openmm_interface,
            do_initial_benchmark: bool = True,
            needs_cv_values: bool = False
            ) -> int:
        """
        Run the simulation.
        """
        # TODO: handle minimizations if step == 0, and print the following warning for
        # certain sampling methods: 
        #  Warning: running minimizations. It is recommended that structures are minimized and 
        #  verified by the user before running SEEKR, since minimizations might cause the system 
        #  to drift out of the MMVT cell.
        # TODO: need to save checkpoint reproducibility files.
        self._steps_between_evals = steps_between_evals
        if steps_between_evals == 0:
            steps_between_evals = steps_between_checkpoints
        stopping = False
        cv_values = None
        num_steps = 0
        on_step = start_step
        lock_string = engines_base.create_lock(lock_filename)
        assert self._simulation_obj is not None
        steps_to_take_after_benchmark = 0
        finish_benchmark = False
        while not stopping:
            criteria_satisfied = False
            early_simulation_abort = False
            make_checkpoint = False
            make_eval = False
            steps_since_eval = num_steps % steps_between_evals
            steps_until_eval = steps_between_evals - steps_since_eval
            steps_since_checkpoint = num_steps % steps_between_checkpoints
            steps_until_next_checkpoint = steps_between_checkpoints \
                - steps_since_checkpoint
            timestep = self._simulation_obj.context.getIntegrator().getStepSize()\
                .value_in_unit(unit.picoseconds)
            if steps_until_next_checkpoint < steps_until_eval:
                steps_to_take = steps_until_next_checkpoint
            else:
                steps_to_take = steps_until_eval
                make_eval = True
            if num_steps == 0:
                if needs_cv_values:
                    self.create_openmm_cv_list()
                    self.create_openmm_cv_object()
                if do_initial_benchmark:
                    if INITIAL_BENCHMARK_STEPS < steps_to_take:
                        steps_to_take_after_benchmark = steps_to_take - INITIAL_BENCHMARK_STEPS
                        steps_to_take = INITIAL_BENCHMARK_STEPS
                    self._step(1)  # warm-up: flush JIT/kernel compilation before timing
                    num_steps += 1
                    on_step += 1
                    steps_to_take -= 1
                    if steps_to_take > 0:
                        finish_benchmark = True
                    start_time = time.time()
            self._step(steps_to_take)
            num_steps += steps_to_take
            on_step += steps_to_take
            if finish_benchmark:
                total_time = time.time() - start_time
                early_simulation_abort = self.perform_early_simulation_eval(
                    restart_index, swarm_id)
                if not early_simulation_abort:
                    ns_per_day = steps_to_take * timestep * 1e-3 \
                        * 86400 / total_time
                    logging.info(f"Initial benchmark (ns/day): {ns_per_day}")
                    self._step(steps_to_take_after_benchmark)
                    num_steps += steps_to_take_after_benchmark
                    on_step += steps_to_take_after_benchmark

                finish_benchmark = False

            if early_simulation_abort:
                raise Exception("Simulation aborted early.")
            else:
                if num_steps % steps_between_checkpoints == 0:
                    make_checkpoint = True
                if make_eval:
                    stopping = self.perform_eval(restart_index, swarm_id)
                    # NOTE: an logging update is printed every checkpoint. Is this right?
                    if needs_cv_values:
                        #cv_values = self._openmm_cv_object.getCollectiveVariableValues(
                        #    self._simulation_obj.context)
                        cv_values = self.get_cv_values()
                        
                    criteria_satisfied = completion_openmm_interface.criteria_satisfied(
                        current_step=on_step, print_update=False, cv_values=cv_values, 
                        swarm_id=swarm_id)
                if make_checkpoint:
                    self._simulation_obj.saveCheckpoint(checkpoint_filename)
                    engines_structures.write_checkpoint_information_file(
                        on_step, restart_index, timestep, cv_values, checkpoint_info_filename)
                    lock_string = engines_base.update_lock(
                        lock_filename, lock_string)
                    
            if on_step >= max_steps:
                stopping = True
                
            elif criteria_satisfied:
                if on_step >= min_steps:
                    stopping = True
                    
            
        self._simulation_obj.saveCheckpoint(checkpoint_filename)
        engines_structures.write_checkpoint_information_file(
            on_step, restart_index, timestep, cv_values, checkpoint_info_filename)
        os.remove(lock_filename)
        return num_steps
    
    def find_current_anchor_index(
            self,
            cv_values: list[float]
            ) -> int | None:
        """
        Find the current anchor index the simulation is in.
        """
        found_anchor = False
        current_anchor_index: int | None = None
        for alpha, anchor in enumerate(self._model.anchors):
            in_anchor = True
            for milestone in anchor.milestones:
                curdir = os.getcwd()
                os.chdir(self._model.directory)
                #result = structures.check_values_within_voronoi_boundary(
                #    cv_values, milestone.variables)
                result = milestone.is_point_within(cv_values)
                os.chdir(curdir)
                if not result:
                    in_anchor = False

            if in_anchor:
                assert not found_anchor, "Found system in two "\
                    "different anchors in the same step."
                found_anchor = True
                current_anchor_index = alpha

        # NOTE: the system might be in no anchors if there are no anchors
        #  in the model.
        #assert (current_anchor_index is not None) and (current_anchor_index >= 0), \
        #    "No anchor was found for the current simulation step."
        return current_anchor_index
    
    def get_evaluation_interval(self) -> int:
        return 0       
    
            

@define
class Conventional_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the conventional MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.Conventional_md_sampling_method
    _system_obj: openmm.System | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    _openmm_cv_list: list[typing.Any] | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _openmm_cv_object: typing.Any | None = None
    _fail_on_save_to_current_anchor_if_outside_milestones: bool = True

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        return
    
@define
class Metadynamics_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the metadynamics MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.Metadynamics_md_sampling_method
    _system_obj: openmm.System | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    _openmm_cv_list: list[typing.Any] | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _openmm_cv_object: typing.Any | None = None
    _fail_on_save_to_current_anchor_if_outside_milestones: bool = True
    _metadynamics_obj: openmm_app.Metadynamics | None = None
    _current_alpha: int | None = None
    #_directory: str | None = None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        #system: openmm.System = kwargs["system"]
        system = self._system_obj
        assert system is not None, \
            "The system object must be assigned before calling this method."
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        directory: str = kwargs["directory"]
        #self._directory = directory
        md_engine = self._model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine is not None
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        
        self.create_openmm_cv_list()
        #self.create_openmm_cv_object()

        boosted_cvs: list[collective_variables_structures.Collective_variable] = []
        if len(self.sampling_method.cv_names) == 0:
            # Then use all CVs
            for cv in model.collective_variables:
                boosted_cvs.append(cv)
        else:
            for cv_name in self.sampling_method.cv_names:
                cv = model.get_cv_by_name(cv_name)
                boosted_cvs.append(cv)
        assert len(boosted_cvs) > 0, \
            "No CVs were provided for the metadynamics method."
        assert len(boosted_cvs) <= 3, \
            "Too many CVs were provided for the metadynamics method."
        num_boosted_cvs = len(boosted_cvs)
        metadyn_biasfactor = self.sampling_method.bias_factor
        metadyn_height = self.sampling_method.gaussian_height
        steps_per_metadyn_update = self.sampling_method.steps_per_update
        bias_widths = make_list_from_ambig_entry(
            self.sampling_method.gaussian_width, num_boosted_cvs)
        grid_widths = make_list_from_ambig_entry(
            self.sampling_method.number_of_points, num_boosted_cvs)
        if self.sampling_method.write_biases:
            metadyn_bias_dir = os.path.join(directory, BIAS_DIR_NAME)
            save_frequency = steps_per_metadyn_update
            if not os.path.exists(metadyn_bias_dir):
                os.mkdir(metadyn_bias_dir)
        else:
            metadyn_bias_dir = None
            save_frequency = None
        metaD_force_biases = []
        #for i, cv in enumerate(boosted_cvs):
        for i, cv in enumerate(model.collective_variables):
            boosted_cv_index = -1
            for j, boosted_cv in enumerate(boosted_cvs):
                if cv.index == boosted_cv.index:
                    boosted_cv_index = j
                    break
            metaD_force = self._openmm_cv_list[cv.index]
            if boosted_cv_index < 0:
                gridWidth = 3
                biasWidth = 9e9
            else:
                gridWidth = grid_widths[boosted_cv_index]
                biasWidth = bias_widths[boosted_cv_index]
            force_bias = openmm_app.BiasVariable(
                metaD_force, 
                minValue=cv.get_min_value(), 
                maxValue=cv.get_max_value(),
                #biasWidth=bias_widths[i], 
                biasWidth=biasWidth,
                periodic=cv.is_periodic(), 
                gridWidth=gridWidth)
                #gridWidth=grid_widths[i])
            metaD_force_biases.append(force_bias)
        
        assert self._simulation_obj is not None
        self._metadynamics_obj = openmm_app.Metadynamics(
            system, variables=metaD_force_biases, temperature=model.temperature,
            biasFactor=metadyn_biasfactor, height=metadyn_height, 
            frequency=steps_per_metadyn_update, 
            saveFrequency=save_frequency, biasDir=metadyn_bias_dir)
        self._simulation_obj.context.reinitialize(preserveState=True)
        return
    
    def get_cv_values(
            self
            ) -> list | None:
        cv_values = self._metadynamics_obj.getCollectiveVariables(
            self._simulation_obj)
        return cv_values

    def perform_eval(
            self, 
            _restart_index: int, 
            _swarm_id: int
            ) -> bool:
        """
        Perform an evaluation at a certain point in the simulation.
        """
        return False

    def _step(
            self,
            steps: int
            ) -> None:
        """
        Take a number of steps in the simulation.
        """
        assert self._metadynamics_obj is not None
        assert self._simulation_obj is not None
        self._metadynamics_obj.step(self._simulation_obj, steps)
        return

@define
class Steered_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the Steered MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.Steered_md_sampling_method
    _system_obj: openmm.System | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    _openmm_cv_list: list[typing.Any] | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _openmm_cv_object: openmm.CustomCVForce | None = None
    _fail_on_save_to_current_anchor_if_outside_milestones: bool = True

    _current_alpha: int | None = None
    #_directory: str | None = None
    _procedure: list | None = None
    _current_step_in_procedure: int = 0
    _windows_list_zipped: list[list[float]] | None = None
    _on_window: int = 0
    _anchor_positions_cache: dict | None = None
    _running_directory: str | None = None
    _smd_reporter: typing.Any | None = None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        #system: openmm.System = kwargs["system"]
        #self._system_obj = system
        system = self._system_obj
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        #directory: str = kwargs["directory"]
        #self._directory = directory
        md_engine = self._model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine is not None
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        
        stage = model.stages[self._stage_index]
        assert stage.type == "md_stage"
        
        expr_list = []
        self.create_openmm_cv_list()
        cv_name_list = []
        value_name_list = []
        for cv_index, openmm_cv in enumerate(self._openmm_cv_list):
            cv_name = f"cv_{cv_index}"
            cv_name_list.append(cv_name)
            value_name = f"value_{cv_index}"
            value_name_list.append(value_name)
            cv_expr = f"({cv_name} - {value_name})^2"
            expr_list.append(cv_expr)

        full_expr = "smd_force_const*(" + " + ".join(expr_list) + ")"
        self._openmm_cv_object = openmm.CustomCVForce(full_expr)
        self._openmm_cv_object.addGlobalParameter("smd_force_const", self.sampling_method.force_constant)
        for cv_index, openmm_cv in enumerate(self._openmm_cv_list):
            cv_name = cv_name_list[cv_index]
            value_name = value_name_list[cv_index]
            self._openmm_cv_object.addGlobalParameter(value_name, 0.0)
            self._openmm_cv_object.addCollectiveVariable(cv_name, openmm_cv)
            
        freeGroups = set(range(32)) - set(myforce.getForceGroup() for myforce in system.getForces())
        if len(freeGroups) == 0:
            raise RuntimeError('Cannot assign a force group to the steered MD stage. '
                            'The maximum number (32) of the force groups is already used.')
        self._openmm_cv_object.setForceGroup(max(freeGroups))
        system.addForce(self._openmm_cv_object)
        self._openmm_cv_list.append(self._openmm_cv_object)

        assert self._simulation_obj is not None
        self._simulation_obj.context.reinitialize(preserveState=True)

        #cv_values: list[float] = []
        assert self._simulation_obj is not None
        assert self._openmm_cv_list is not None
        cv_values = self.get_cv_values()
        for cv_index, cv_value in enumerate(cv_values):
            for i in range(self._openmm_cv_object.getNumGlobalParameters()):
                param_name = self._openmm_cv_object.getGlobalParameterName(i)
                if param_name == f"value_{cv_index}":
                    self._simulation_obj.context.setParameter(param_name, cv_value)

        self._running_directory = kwargs["directory"]
        self._anchor_positions_cache = {}

        # Determine filled/unfilled anchors from Anchor_reporter JSON sidecar.
        smd_reporter = None
        for r in stage.reporters:
            if r.type == "anchor":
                smd_reporter = r
                break
        self._smd_reporter = smd_reporter

        filled_anchor_indices: list[int] = []
        if smd_reporter is not None:
            basename = stage.sampling_method.get_output_positions_basename_noext()
            ext = smd_reporter.get_reporter_extension()
            reporter_glob = os.path.join(
                self._running_directory,
                f"{basename}*.{smd_reporter.name}.{ext}")
            reporter_files = sorted(glob.glob(reporter_glob))
            if reporter_files:
                # Aggregate filled anchors across all per-file restart JSONs.
                filled_set: set[int] = set()
                for reporter_file in reporter_files:
                    json_sidecar = os.path.splitext(reporter_file)[0] + ".json"
                    if not os.path.exists(json_sidecar):
                        continue
                    with open(json_sidecar, "r") as f:
                        json_data = json.load(f)
                    for v in json_data.values():
                        if v.get("filled", False):
                            filled_set.add(int(v["index"]))
                filled_anchor_indices = sorted(filled_set)

        all_md_anchor_indices = [
            a.index for a in self._model.anchors
            if "molecular_dynamics" in a.scale_types
        ]
        unfilled_anchor_indices = [
            i for i in all_md_anchor_indices if i not in filled_anchor_indices
        ]
        # If nothing has been sampled yet, seed from the current position.
        if not filled_anchor_indices:
            current_anchor_index = self.find_current_anchor_index(list(cv_values))
            if current_anchor_index is not None:
                filled_anchor_indices = [current_anchor_index]
                if current_anchor_index in unfilled_anchor_indices:
                    unfilled_anchor_indices.remove(current_anchor_index)
                # Pre-cache current positions so _load_anchor_from_cache works
                # without needing a reporter file on disk.
                self._save_to_anchor_cache(current_anchor_index)

        self._procedure = sampling_methods_runner.get_smd_procedure(
            self._model, filled_anchor_indices, unfilled_anchor_indices, cv_values)
        logging.info("Steered MD procedure summary:")
        for step in self._procedure:
            logging.info(f"anchor {step[0]} to anchor {step[1]}")
        if self._procedure:
            new_source_anchor_index = \
                self._procedure[self._current_step_in_procedure][0]
            self._load_anchor_from_cache(new_source_anchor_index)
        else:
            logging.info("All anchors already filled; no SMD steps needed.")

        return
    
    def assign_new_windows(
            self,
            cv_values: list[float]
            ) -> None:
        """

        """
        # If a new set of windows need to be assigned:
        source_anchor_index = self._procedure[self._current_step_in_procedure][0]
        target_anchor_index = self._procedure[self._current_step_in_procedure][1]
        source_anchor: structures.Anchor = self._model.anchors[source_anchor_index]
        target_anchor: structures.Anchor = self._model.anchors[target_anchor_index]
        windows_list_unzipped = []
        total_sq_distance = 0.0
        logging.info(f"Assigning a new set of windows for SMD:")
        for variable_key in source_anchor.variables:
            if not variable_key.startswith("value_"):
                continue
            var_cv = int(variable_key.split("_")[1])
            if variable_key in target_anchor.variables:
                start_value = cv_values[var_cv]
                last_value = target_anchor.variables[variable_key]
                displacement = (last_value - start_value)
                total_sq_distance += displacement ** 2
                
        distance = np.sqrt(total_sq_distance)
        total_time_ps = 1000.0 * distance / self.sampling_method.velocity
        timestep = self._model.get_timestep_by_type("molecular_dynamics")
        total_num_timesteps = int(total_time_ps / timestep)
        num_smd_windows = (total_num_timesteps // self._steps_between_evals) + 1

        for variable_key in source_anchor.variables:
            if not variable_key.startswith("value_"):
                continue
            var_cv = int(variable_key.split("_")[1])
            if variable_key in target_anchor.variables:
                start_value = cv_values[var_cv]
                last_value = target_anchor.variables[variable_key]
                cv_name = self._model.collective_variables[var_cv].name
                logging.info(
                    f"Running CV {cv_name} steered MD from value {start_value} to "
                    f"{last_value}")
                windows = np.linspace(start_value, last_value, num_smd_windows)
                windows_list_unzipped.append(windows)

        self._windows_list_zipped = list(zip(*windows_list_unzipped))
        distance = np.sqrt(total_sq_distance)
        steps_in_window = int(abs(distance) / (self.sampling_method.velocity * timestep))
        assert steps_in_window > 0, \
            "The number of steps in each SMD window must be greater than 0. "\
            "Consider reducing the SMD velocity."
        self._on_window = 0
        
        return

    def _save_to_anchor_cache(self, anchor_index: int) -> None:
        """
        Cache the current simulation positions for anchor_index.
        Called when SMD reaches a destination anchor.
        """
        state = self._simulation_obj.context.getState(
            getPositions=True, enforcePeriodicBox=False)
        positions_np = state.getPositions(asNumpy=True)\
            .value_in_unit(unit.nanometers)
        try:
            bv = state.getPeriodicBoxVectors()
        except Exception:
            bv = None
        self._anchor_positions_cache[anchor_index] = (positions_np, bv)

    def _load_anchor_from_cache(self, anchor_index: int) -> None:
        """
        Load positions for anchor_index into the simulation context.
        Uses the in-memory cache when available; otherwise reads the
        Anchor_reporter trajectory from disk (restart-safe).
        """
        logging.info(f"Loading positions for anchor {anchor_index}.")
        if anchor_index not in self._anchor_positions_cache:
            self._load_anchor_from_reporter_file(anchor_index)
            return
        positions_np, bv = self._anchor_positions_cache[anchor_index]
        self._simulation_obj.context.setPositions(positions_np * unit.nanometers)
        if bv is not None:
            self._simulation_obj.context.setPeriodicBoxVectors(*bv)

    def _load_anchor_from_reporter_file(self, anchor_index: int) -> None:
        """
        Read ALL Anchor_reporter trajectories and JSON sidecars from disk,
        iterating oldest-to-newest, and build an anchor→(positions, bv) map.
        Later restart files overwrite earlier ones so the most recent frame per
        anchor is always used. Then load the frame for anchor_index into the
        simulation and cache it.
        """
        stage = self._model.stages[self._stage_index]
        basename = stage.sampling_method.get_output_positions_basename_noext()
        ext = self._smd_reporter.get_reporter_extension()
        reporter_glob = os.path.join(
            self._running_directory,
            f"{basename}*.{self._smd_reporter.name}.{ext}")
        reporter_files = sorted(glob.glob(reporter_glob))
        assert reporter_files, \
            f"No reporter files found to load anchor {anchor_index}. " \
            f"Glob: {reporter_glob}"

        mdtraj_top = mdtraj.Topology.from_openmm(self._simulation_obj.topology)

        # Build anchor → (positions_nm, bv) from every restart file.
        # Iterating oldest→newest so later files overwrite earlier ones.
        anchor_frame_map: dict[int, tuple] = {}
        for reporter_file in reporter_files:
            json_sidecar = os.path.splitext(reporter_file)[0] + ".json"
            if not os.path.exists(json_sidecar):
                continue
            with open(json_sidecar, "r") as f:
                json_data = json.load(f)
            file_filled_indices = sorted(
                v["index"] for v in json_data.values() if v.get("filled", False)
            )
            if not file_filled_indices:
                continue
            if ext == "pdb":
                traj = mdtraj.load(reporter_file)
            else:
                traj = mdtraj.load(reporter_file, top=mdtraj_top)
            for fn, ai in enumerate(file_filled_indices):
                if fn >= len(traj):
                    break
                positions_nm = traj.xyz[fn].astype(np.float64)
                bv = None
                if traj.unitcell_vectors is not None:
                    bv_np = traj.unitcell_vectors[fn]
                    bv = (
                        openmm.Vec3(*bv_np[0]) * unit.nanometers,
                        openmm.Vec3(*bv_np[1]) * unit.nanometers,
                        openmm.Vec3(*bv_np[2]) * unit.nanometers,
                    )
                anchor_frame_map[ai] = (positions_nm, bv)

        assert anchor_index in anchor_frame_map, \
            f"Anchor {anchor_index} not found in any reporter file. " \
            f"Glob: {reporter_glob}"

        positions_nm, bv = anchor_frame_map[anchor_index]
        self._simulation_obj.context.setPositions(positions_nm * unit.nanometers)
        if bv is not None:
            self._simulation_obj.context.setPeriodicBoxVectors(*bv)
        self._anchor_positions_cache[anchor_index] = (positions_nm, bv)

    def perform_eval(
            self, 
            _restart_index: int, 
            _swarm_id: int
            ) -> bool:
        """
        Perform an evaluation at a certain point in the simulation.
        """
        assert self._model is not None
        if self._model.anchors is None:
            return
        target_anchor_index = self._procedure[self._current_step_in_procedure][1]
        #cv_values_dummy: list[float] = []
        assert self._simulation_obj is not None
        assert self._openmm_cv_list is not None
        #cv_values = self._openmm_cv_object.getCollectiveVariableValues(self._simulation_obj.context)
        cv_values = self.get_cv_values()
        if self._windows_list_zipped is None:
            self.assign_new_windows(cv_values)

        window_values = self._windows_list_zipped[self._on_window]
        for cv_index, window_value in enumerate(window_values):
            # TODO: will this ever be toggled off? Maybe if the CVs change between Anchors.
            toggle = 1
            for i in range(self._openmm_cv_object.getNumGlobalParameters()):
                param_name = self._openmm_cv_object.getGlobalParameterName(i)
                if param_name == f"value_{cv_index}":
                    self._simulation_obj.context.setParameter(param_name, window_value)
            
        if self._on_window < len(self._windows_list_zipped) - 1:
            self._on_window += 1
        else:
            # The destination is reached
            logging.info(f"Reached anchor {target_anchor_index}.")
            self._current_step_in_procedure += 1
            #self.save_to_current_anchor(target_anchor_index, cv_values, SMD_NAME)
            self._save_to_anchor_cache(target_anchor_index)
            if self._current_step_in_procedure < len(self._procedure):
                new_source_anchor_index = self._procedure[self._current_step_in_procedure][0]
                self._load_anchor_from_cache(new_source_anchor_index)
                cv_values = self.get_cv_values()
                self.assign_new_windows(cv_values)
                for cv_index, cv_value in enumerate(cv_values):
                    for i in range(self._openmm_cv_object.getNumGlobalParameters()):
                        param_name = self._openmm_cv_object.getGlobalParameterName(i)
                        if param_name == f"value_{cv_index}":
                            self._simulation_obj.context.setParameter(param_name, cv_value)
            else:
                # Simulation is complete: let the completion criteria finish
                pass

        return False

@define
class MMVT_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the MMVT MD sampling method using the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.MMVT_md_sampling_method
    _system_obj: openmm.System | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    _openmm_cv_list: list[typing.Any] | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _openmm_cv_object: typing.Any | None = None
    _fail_on_save_to_current_anchor_if_outside_milestones: bool = True
    _anchor_obj: structures.Anchor | None = None

    def add_global_variables_integrator(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        Some global variables for MMVT method.
        """
        integrator: openmm.Integrator = kwargs["integrator"]
        anchor: structures.Anchor = kwargs["anchor"]
        self._anchor_obj = anchor
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = milestone.alias_index
            if milestone.boundary:
                #integrator.addGlobalVariable(f"milestone_{num_included_milestones}_alias_index", alias_index_i)
                integrator.addGlobalVariable(f"collision_{alias_index_i}", 0)
                #num_included_milestones += 1
                for key, value in milestone.variables.items():
                    integrator.addGlobalVariable(f"milestone_{alias_index_i}_{key}", value)
                for j, milestone2 in enumerate(anchor.milestones):
                    alias_index_j = milestone2.alias_index
                    if milestone2.boundary:
                        integrator.addGlobalVariable(f"milestone_{alias_index_i}_{alias_index_j}_count", 0)
                integrator.addGlobalVariable(f"milestone_{alias_index_i}_time", 0)
        
        #integrator.addGlobalVariable("num_milestones", len(anchor.milestones))
        integrator.addGlobalVariable("step_count", 0)
        integrator.addGlobalVariable("time_count", 0)
        integrator.addGlobalVariable("buffer_frames", BUFFER_FRAMES)
        integrator.addGlobalVariable("last_milestone_alias_touched", -1)
        integrator.addGlobalVariable("last_milestone_time_touched", -1)
        integrator.addGlobalVariable("first_encounter_time", -1)
        integrator.addGlobalVariable("first_step_encounter", 0)
        integrator.addGlobalVariable("encounter_this_step", 0)
        integrator.addGlobalVariable("bitvar", 0)
        integrator.addGlobalVariable("threshold", 0)
        return None
    
    def add_per_DOF_variables_integrator(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        Some per-DOF variables for MMVT.
        """
        
        integrator: openmm.Integrator = kwargs["integrator"]
        for i in range(BUFFER_FRAMES):
            integrator.addPerDofVariable(f"encounter_pos_{i}", 0)
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for MMVT.
        """
        integrator: openmm.Integrator = kwargs["integrator"]
        #integrator.addComputePerDof("old_x", "x")
        #integrator.addComputePerDof("old_v", "v")
        anchor = self._anchor_obj
        assert anchor is not None
        # Check to make sure no encounter on first step.
        integrator.beginIfBlock("step_count = 0")
        integrator.beginIfBlock("energy1 > 0")
        integrator.addComputeGlobal("encounter_this_step", "1")
        integrator.addComputeGlobal("first_step_encounter", "1")
        integrator.endBlock()
        integrator.endBlock()
        return None
    
    def post_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add post-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        integrator: openmm.Integrator = kwargs["integrator"]
        anchor = self._anchor_obj
        assert anchor is not None
        integrator.addComputeGlobal("encounter_this_step", "0")
        integrator.addComputeGlobal("bitvar", "energy1")
        
        for j, milestone in enumerate(anchor.milestones):
            i = len(anchor.milestones) - (j + 1)
            integrator.addComputeGlobal("threshold", f"(2^{i})-0.5")
            integrator.beginIfBlock(f"bitvar > threshold")
            integrator.addComputeGlobal(f"collision_{i+1}", "1")
            integrator.addComputeGlobal("bitvar", f"bitvar - (2^{i})")
            integrator.endBlock()
            
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = i+1
            integrator.beginIfBlock(f"collision_{alias_index_i} > 0")
            integrator.addComputeGlobal("encounter_this_step", "1")
            integrator.beginIfBlock(f"last_milestone_alias_touched > -1")
            for j, milestone2 in enumerate(anchor.milestones):
                alias_index_j = j+1
                integrator.beginIfBlock(f"last_milestone_alias_touched = {alias_index_j}")
                integrator.addComputeGlobal(f"milestone_{alias_index_j}_{alias_index_i}_count", f"milestone_{alias_index_j}_{alias_index_i}_count + 1")
                integrator.addComputeGlobal(f"milestone_{alias_index_j}_time", f"milestone_{alias_index_j}_time + time_count - last_milestone_time_touched")
                integrator.endBlock()
            integrator.addComputeGlobal("last_milestone_time_touched", "time_count")
            integrator.addComputeGlobal("last_milestone_alias_touched", f"{alias_index_i}")
            integrator.endBlock() # End If last_milestone_alias_touched > -1
            
            integrator.beginIfBlock("last_milestone_alias_touched = -1")
            integrator.addComputeGlobal("last_milestone_alias_touched", f"{alias_index_i}")
            integrator.addComputeGlobal("first_encounter_time", "time_count")
            integrator.addComputeGlobal("last_milestone_time_touched", "time_count")
            integrator.endBlock() # End If last_milestone_alias_touched = -1
            
            integrator.endBlock() # End If collision_{alias_index_i} > 0
            integrator.addComputeGlobal(f"collision_{alias_index_i}", "0")

        # If using MMVT, then reverse velocities.
        integrator.beginIfBlock(f"encounter_this_step = 1")
        integrator.addComputePerDof("v", "-old_v")
        integrator.addComputePerDof("x", "old_x")
        integrator.endBlock()

        integrator.addComputeGlobal("step_count", "step_count + 1")
        integrator.addComputeGlobal("time_count", "time_count + dt")
        
        return None
    
    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = self._system_obj
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        md_engine = self._model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        self.create_openmm_cv_list()
        
        anchor = self._anchor_obj
        assert anchor is not None
        anchor_cv_indices = anchor.get_cv_indices()
        num_cvs = len(anchor_cv_indices)
        boundary_expr_list: list[str] = []
        if num_cvs == 1:
            my_cv_index = anchor_cv_indices[0]
            cv_interface = self._cv_interface_list[my_cv_index]
            assert cv_interface is not None, f"An interface was not found for cv {cv_index}"
            variables: dict[str, unit.Quantity] = {}
            for i, milestone in enumerate(anchor.milestones):
                alias_index_i = milestone.alias_index
                bitcode = 2**(alias_index_i-1)
                avg_value = milestone.variables["avg_value"]
                bitcode_expr = f"bitcode_{alias_index_i}"
                variables[bitcode_expr] = bitcode
                k_expr = f"k_{alias_index_i}"
                variables[k_expr] = milestone.variables["k"]
                value_expr = f"value_{alias_index_i}"
                variables[value_expr] = avg_value
                boundary_force_expr = cv_interface.make_boundary_force_expr(alias_index_i)
                boundary_expr_list.append(bitcode_expr + "*" + boundary_force_expr)

            full_expr = " + ".join(boundary_expr_list)
            boundary_force = cv_interface.make_boundary_force(full_expr)
            cv_interface.add_boundary_groups_and_variables(
                boundary_force, variables)
            boundary_force.setForceGroup(1)
            system.addForce(boundary_force)
            
        else: # num_cvs > 1
            me_expr_list_all_milestones = []
            neighbor_expr_list_all_milestones = []
            me_force_list = []
            neighbor_force_list = []
            bitcode_expr_dict: dict[str, int] = {}
            for i, milestone in enumerate(anchor.milestones):
                me_expr_list_this_milestone = []
                neighbor_expr_list_this_milestone = []
                
                alias_index_i = milestone.alias_index
                bitcode_expr = f"bitcode_{alias_index_i}"
                bitcode = 2**(alias_index_i-1)
                bitcode_expr_dict[bitcode_expr] = bitcode
                for variable in milestone.variables:
                    if variable.startswith("me"):
                        cv_index = int(variable.split("_")[1])
                        cv_interface = self._cv_interface_list[cv_index]
                        assert cv_interface is not None, f"An interface was not found for cv {cv_index}"
                        me_key_name = f"me_{cv_index}"
                        neighbor_key_name = f"neighbor_{cv_index}"
                        # TODO: make sure this is right - didn't I make a function to handle VT
                        # of multiple types of CVs? And why isn't there a squared term?
                        me_force, neighbor_force = cv_interface.make_voronoi_cv_boundary_forces(
                            milestone.variables[me_key_name], 
                            milestone.variables[neighbor_key_name])
                        me_force_list.append(me_force)
                        neighbor_force_list.append(neighbor_force)
                        me_cv_expr = f"me_cv_{cv_index}_alias_{alias_index_i}"
                        me_expr_list_this_milestone.append(me_cv_expr)
                        me_expr_list_all_milestones.append(me_cv_expr)
                        neighbor_cv_expr = f"neighbor_cv_{cv_index}_alias_{alias_index_i}"
                        neighbor_expr_list_this_milestone.append(neighbor_cv_expr)
                        neighbor_expr_list_all_milestones.append(neighbor_cv_expr)

                me_expr = " + ".join(me_expr_list_this_milestone)
                neighbor_expr = " + ".join(neighbor_expr_list_this_milestone)
                cv_expr = f"{bitcode_expr}*step((" + me_expr + ") - (" + neighbor_expr + "))"
                boundary_expr_list.append(cv_expr)

            full_expr = " + ".join(boundary_expr_list)
            openmm_cv_object = openmm.CustomCVForce(full_expr)
            for i in range(len(me_expr_list_all_milestones)):
                me_expr = me_expr_list_all_milestones[i]
                neighbor_expr = neighbor_expr_list_all_milestones[i]
                me_force = me_force_list[i]
                neighbor_force = neighbor_force_list[i]
                openmm_cv_object.addCollectiveVariable(me_expr, me_force)
                openmm_cv_object.addCollectiveVariable(neighbor_expr, neighbor_force)

            for bitcode_expr, bitcode in bitcode_expr_dict.items():
                openmm_cv_object.addGlobalParameter(bitcode_expr, bitcode)

            openmm_cv_object.setForceGroup(1)
            system.addForce(openmm_cv_object)
            
        assert self._simulation_obj is not None
        self._simulation_obj.context.reinitialize(preserveState=True)
        return

    def perform_eval(
            self, 
            restart_index: int, 
            swarm_id: int
            ) -> bool:
        """
        This is when milestoning information is written to the output file.
        """
        assert self._model is not None
        assert self._simulation_obj is not None
        integrator: openmm.Integrator = self._simulation_obj.context.getIntegrator()
        anchor = self._anchor_obj
        stage_name = self._model.stages[self._stage_index].name
        stage_set_name = self._model.stages[self._stage_index].set_name
        assert anchor is not None
        anchor_directory = os.path.join(
            self._model.directory, "anchor_sets", stage_set_name, anchor.directory,
            anchor.stage_parameters[stage_name]["directory"])
        output_basename = f"milestoning_restart{restart_index}_swarm_{swarm_id}.out"
        output_filename = os.path.join(anchor_directory, output_basename)
        
        milestone_count_dict = {}
        milestone_time_dict = {}
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = i+1
            
            for j, milestone2 in enumerate(anchor.milestones):
                alias_index_j = j+1
                milestone_count_key = f"milestone_{alias_index_i}_{alias_index_j}_count"
                milestone_count_dict[milestone_count_key] \
                    = int(integrator.getGlobalVariableByName(milestone_count_key))

            milestone_time_key = f"milestone_{alias_index_i}_time"
            milestone_time_dict[milestone_time_key] \
                = float(integrator.getGlobalVariableByName(milestone_time_key))

        self.sampling_method.write_output_file(
            filename=output_filename,
            step_count=int(integrator.getGlobalVariableByName("step_count")),
            time_count=float(integrator.getGlobalVariableByName("time_count")),
            first_encounter_time=float(integrator.getGlobalVariableByName("first_encounter_time")),
            last_milestone_alias_touched=int(integrator.getGlobalVariableByName("last_milestone_alias_touched")),
            last_milestone_time_touched=float(integrator.getGlobalVariableByName("last_milestone_time_touched")),
            milestone_count_dict=milestone_count_dict,
            milestone_time_dict=milestone_time_dict
        )
        return False
    
    def perform_early_simulation_eval(
            self, 
            restart_index: int, 
            swarm_id: int
            ) -> bool:
        """
        Perform an evaluation once at an early point in the simulation.
        """
        abort = False
        integrator = self._simulation_obj.context.getIntegrator()
        first_step_encounter=int(integrator.getGlobalVariableByName("first_step_encounter"))
        if first_step_encounter == 1:
            logging.critical("Milestoning simulation encounter on first step: the system is trapped "\
            "behind a boundary. Check and revise milestone boundary definitions and atomic positions.")
            abort = True
            
        return abort
    
Sampling_md_openmm_interface = typing.Union[Conventional_md_openmm_interface,
                                            Metadynamics_md_openmm_interface,
                                            MMVT_md_openmm_interface]

def assign_sampling_method_interface(
        stage_index: int,
        sampling_method: sampling_methods_structures.Base_md_sampling_method, 
        md_engine: structures.MD_engine,
        plugin_list: list[str]
        ) -> Sampling_md_openmm_interface:
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
    elif sampling_method.type == "steered":
        return Steered_md_openmm_interface(stage_index, sampling_method)
    elif sampling_method.type == "mmvt":
        return MMVT_md_openmm_interface(stage_index, sampling_method)
    else:
        # Run through plugins
        for plugin_name in plugin_list:
            sampl_meth_module_str = base.get_plugin_str(plugin_name, submodule_name="sampling_methods")
            if sampl_meth_module_str is not None:
                plugin_module_sampl_meth_interface = importlib.import_module(
                    sampl_meth_module_str+".openmm_interface")
            found, sampl_meth_interface = plugin_module_sampl_meth_interface.assign_sampling_method_interface(
                stage_index, sampling_method)
            if found:
                return sampl_meth_interface

    raise Exception("Sampling method not found: " + sampling_method.type)
