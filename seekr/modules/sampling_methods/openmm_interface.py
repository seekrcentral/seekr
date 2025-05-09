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
import typing
import logging

from attrs import define
import numpy as np
import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import seekr.modules.structures as structures
import seekr.modules.engines.base as engines_base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.collective_variables.structures as collective_variables_structures
import seekr.modules.completion_criteria.runner as completion_criteria_runner
import seekr.modules.sampling_methods.runner as sampling_methods_runner
import seekr.modules.completion_criteria.openmm_interface \
    as completion_criteria_openmm_interface
import seekr.modules.collective_variables.openmm_interface \
    as collective_variables_openmm_interface

BIAS_DIR_NAME = "metad_bias"
METADYN_NAME = "metadynamics.{}.swarm_{}"
SMD_NAME = "steered.{}.swarm_{}"
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
    sampling_method: sampling_methods_structures.MD_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None

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
            **kwargs: typing.Any
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
            **kwargs: typing.Any
            ) -> None:
        """
        Add per-degree-of-freedom variables to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        None for conventional MD.
        """
        return None
    
    def post_integration_instructions(
            self, 
            **kwargs: typing.Any
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
            **kwargs: typing.Any
            ):
        """
        Perform any necessary preparations before the simulation starts.
        """
        return None
    
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
            restart_index: int, 
            swarm_index: int
            ) -> None:
        """
        Perform an evaluation at a certain point in the simulation.
        """
        pass

    def perform_early_simulation_eval(
            self, 
            restart_index: int, 
            swarm_index: int
            ) -> None:
        """
        Perform an evaluation once at an early point in the simulation.
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
            swarm_index: int,
            checkpoint_filename: str,
            checkpoint_info_filename: str,
            lock_filename: str,
            completion_openmm_interface: \
                completion_criteria_openmm_interface\
                    .Completion_criteria_openmm_interface,
            do_initial_benchmark: bool = True,
            ) -> int:
        """
        Run the simulation.
        """
        # TODO: need to save checkpoint reproducibility files.
        self._steps_between_evals = steps_between_evals
        if steps_between_evals == 0:
            steps_between_evals = steps_between_checkpoints
        stopping = False
        num_steps = 0
        on_step = start_step
        lock_string = engines_base.create_lock(lock_filename)
        assert self._simulation_obj is not None
        steps_to_take_after_benchmark = 0
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
            if num_steps == 0 and do_initial_benchmark:
                if INITIAL_BENCHMARK_STEPS < steps_to_take:
                    steps_to_take_after_benchmark = steps_to_take - INITIAL_BENCHMARK_STEPS
                    steps_to_take = INITIAL_BENCHMARK_STEPS
                start_time = time.time()
            #self._simulation_obj.step(steps_to_take)
            self._step(steps_to_take)
            if num_steps == 0 and do_initial_benchmark:
                total_time = time.time() - start_time
                early_simulation_abort = self.perform_early_simulation_eval(
                    restart_index, swarm_index)
                if not early_simulation_abort:
                    ns_per_day = steps_to_take * timestep * 1e-3 \
                        * 86400 / total_time
                    logging.info(f"Initial benchmark (ns/day): {ns_per_day}")
                    self._step(steps_to_take_after_benchmark)
            num_steps += steps_to_take
            on_step += steps_to_take
            if early_simulation_abort:
                raise Exception("Simulation aborted early.")
            else:
                if num_steps % steps_between_checkpoints == 0:
                    make_checkpoint = True
                if make_eval:
                    #integrator = self._simulation_obj.context.getIntegrator()
                    #abort = integrator.getGlobalVariableByName("abort")
                    self.perform_eval(restart_index, swarm_index)
                    # NOTE: an logging update is printed every checkpoint. Is this right?
                    criteria_satisfied = completion_openmm_interface.criteria_satisfied(
                        current_step=on_step, print_update=make_checkpoint)
                if make_checkpoint:
                    self._simulation_obj.saveCheckpoint(checkpoint_filename)
                    engines_structures.write_checkpoint_information_file(
                        on_step, restart_index, timestep, checkpoint_info_filename)
                    lock_string = engines_base.update_lock(
                        lock_filename, lock_string)
            if on_step >= max_steps:
                stopping = True
            elif criteria_satisfied:
                if on_step >= min_steps:
                    stopping = True
            
        self._simulation_obj.saveCheckpoint(checkpoint_filename)
        engines_structures.write_checkpoint_information_file(
            on_step, restart_index, timestep, checkpoint_info_filename)
        os.remove(lock_filename)
        return num_steps
    
    def find_current_anchor_index(
            self,
            cv_values: list[float]
            ) -> int:
        """
        Find the current anchor index the simulation is in.
        """
        found_anchor = False
        current_anchor_index: int = -1
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

        assert current_anchor_index >= 0, \
            "No anchor was found for the current simulation step."
        return current_anchor_index
                
    def save_to_current_anchor(
            self, 
            current_anchor_index: int,
            cv_values: list[float],
            basename: str
            ) -> None:
        """
        Save the current state of the simulation to the appropriate anchor.
        """
        assert self._model is not None
        current_anchor = self._model.anchors[current_anchor_index]
        # Check to make sure the system is inside the target anchor
        if not current_anchor.md:
            logging.critical(f"System has reached non-MD (BD only?) anchor {current_anchor_index}. "\
                            "Consider imposing a Proximity restraint or other restraint.")
            raise Exception("Simulation has reached a non-MD anchor. Aborting.")
        for milestone in current_anchor.milestones:
            curdir = os.getcwd()
            os.chdir(self._model.directory)
            result = milestone.is_point_within(cv_values)
            os.chdir(curdir)
            if not result:
                logging.critical(f"System has not reached the target anchor {current_anchor_index}.")
                raise Exception("Target anchor not reached.")
        anchor_directory = os.path.join(
            self._model.directory, current_anchor.directory,
            current_anchor.stage_parameters[self._stage_index+1]["directory"])
        # Save this anchor's new structure
        if self._model.using_toy():
            extension = ".txt"
        else:
            extension = ".pdb"
        
        var_string = current_anchor.make_var_string()
        output_positions_basename_noext = basename.format(var_string, 0)
        output_positions_basename = output_positions_basename_noext + extension
        output_positions_filename = os.path.join(anchor_directory, 
                                            output_positions_basename)
        current_anchor.change_positions_filename(output_positions_basename_noext, self._stage_index)
        if self._model.using_toy():
            write_txt_file_from_simulation(output_positions_filename, self._simulation_obj)
        else:
            write_pdb_file_from_simulation(output_positions_filename, self._simulation_obj)

    def load_current_anchor(
            self, 
            current_anchor_index: int,
            ) -> None:
        """
        Load the current state of the simulation from the appropriate anchor.
        """
        assert self._model is not None
        current_anchor = self._model.anchors[current_anchor_index]
        # Check to make sure the system is inside the target anchor
        if not current_anchor.md:
            logging.critical(f"Attempting to load a non-MD anchor {current_anchor_index}. Aborting.")
            raise Exception("Cannot read non-MD anchor. Aborting.")
        stage_parameter = current_anchor.stage_parameters[self._stage_index+1]
        assert stage_parameter is not None, \
            f"Stage parameter for stage {self._stage_index} is None."
        assert "directory" in stage_parameter, \
            f"Stage parameter for stage {self._stage_index} does not contain "\
            +"a directory."
        stage_directory = stage_parameter["directory"]
        stage = self._model.stages[self._stage_index]
        stage_output_globs: list[str] = stage.sampling_method.get_output_positions_glob_list()
        for stage_output_glob in stage_output_globs:
            output_files_glob = os.path.join(
                self._model.directory, current_anchor.directory, stage_directory, 
                stage_output_glob)
            output_files_full = glob.glob(output_files_glob)
            if len(output_files_full) > 0:
                loading_filename = output_files_full[0]
                break

        if self._model.using_toy():
            open_txt_file_for_simulation(loading_filename, self._simulation_obj)
        else:
            open_pdb_file_for_simulation(loading_filename, self._simulation_obj)
        return            
            

@define
class Conventional_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the conventional MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.Conventional_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _model: structures.Seekr_model | None = None
    
@define
class Metadynamics_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the metadynamics MD sampling method using
    the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.Metadynamics_md_sampling_method
    _simulation_obj: openmm_app.Simulation | None = None
    _metadynamics_obj: openmm_app.Metadynamics | None = None
    _steps_between_evals: int = 0
    _openmm_cv_list: list[typing.Any] | None = None
    _model: structures.Seekr_model | None = None
    _current_alpha: int | None = None
    _directory: str | None = None

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = kwargs["system"]
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        directory: str = kwargs["directory"]
        self._directory = directory
        md_engine = model.md_engine
        assert md_engine is not None
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        if self._openmm_cv_list is None:
            self._openmm_cv_list = []
        for i, cv in enumerate(model.collective_variables):
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                cv, md_engine)
            assert cv_interface is not None
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
        
        assert self._simulation_obj is not None
        self._metadynamics_obj = openmm_app.Metadynamics(
            system, variables=metaD_force_biases, temperature=model.temperature,
            biasFactor=metadyn_biasfactor, height=metadyn_height, 
            frequency=steps_per_metadyn_update, 
            saveFrequency=save_frequency, biasDir=metadyn_bias_dir)
        self._simulation_obj.context.reinitialize(preserveState=True)
        return
    
    def perform_eval(
            self, 
            restart_index: int, 
            swarm_index: int
            ) -> None:
        """
        Perform an evaluation at a certain point in the simulation.
        """
        assert self._model is not None
        if self._model.anchors is None:
            return
        cv_values: list[float] = []
        assert self._simulation_obj is not None
        assert self._openmm_cv_list is not None
        for i, cv in enumerate(self._openmm_cv_list):
            cv_value = cv.getCollectiveVariableValues(self._simulation_obj.context)
            cv_values.append(cv_value[0])

        current_anchor_index = self.find_current_anchor_index()
        if current_anchor_index != self._current_alpha:
            self._current_alpha = current_anchor_index
            self.save_to_current_anchor(current_anchor_index, cv_values, METADYN_NAME)

        # TODO: remove once metadynamics is validated
        """
        found_anchor = False
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
                if alpha == self._current_alpha:
                    continue
                self._current_alpha = alpha
                self.save_to_current_anchor(alpha, cv_values)
                
                current_anchor = self._model.anchors[alpha]
                if not current_anchor.md:
                    logging.critical(f"System has reached non-MD (BD only?) anchor {alpha}. "\
                                     "Consider imposing a Proximity restraint or other restraint.")
                    raise Exception("Metadynamics simulation has reached a non-MD anchor. Aborting.")
                assert current_anchor.stage_parameters[self._stage_index+1] is not None
                # MYPY: indexing problems with this dictionary
                anchor_directory = os.path.join(
                    self._model.directory, current_anchor.directory,
                    current_anchor.stage_parameters[self._stage_index+1]["directory"])
                # Save this anchor's new structure
                if self._model.using_toy():
                    extension = ".txt"
                else:
                    extension = ".pdb"
                
                var_string = current_anchor.make_var_string()
                output_positions_basename_noext = METADYN_NAME.format(var_string, 0)
                output_positions_basename = output_positions_basename_noext + extension
                output_positions_filename = os.path.join(anchor_directory, 
                                                    output_positions_basename)
                current_anchor.change_positions_filename(output_positions_basename_noext, self._stage_index)
                if self._model.using_toy(): # TODO: fill this in later
                    write_txt_file_from_simulation(output_positions_filename, self._simulation_obj)
                else:
                    write_pdb_file_from_simulation(output_positions_filename, self._simulation_obj)
                """

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
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _openmm_cv_list: list[typing.Any] | None = None
    _openmm_cv_object: openmm.CustomCVForce | None = None
    _cv_interface_list: list[collective_variables_openmm_interface\
                             .CV_openmm_interface] | None = None
    _model: structures.Seekr_model | None = None
    _current_alpha: int | None = None
    _directory: str | None = None
    _procedure: list | None = None
    _current_step_in_procedure: int = 0
    _windows_list_zipped: list[list[float]] | None = None
    _on_window: int = 0

    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = kwargs["system"]
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        directory: str = kwargs["directory"]
        self._directory = directory
        md_engine = model.md_engine
        assert md_engine is not None
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        
        stage = model.stages[self._stage_index]
        assert stage.type == "md_stage"
        if self._openmm_cv_list is None:
            self._openmm_cv_list = []

        expr_list = []
        force_list = []
        self._cv_interface_list = []
        # TODO: need a way to toggle CVs within this expression on or off
        for cv_index, cv in enumerate(model.collective_variables):
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                    cv, md_engine)
            assert cv_interface is not None, f"An interface was not found for cv {cv_index}"
            force = cv_interface.make_voronoi_cv_restraining_forces()
            self._cv_interface_list.append(cv_interface)
            force_list.append(force)
            cv_expr = f"(cv_{cv_index} - value_{cv_index})^2" #f"toggle_{cv_index} * cv_{cv_index}"
            expr_list.append(cv_expr)
        
        full_expr = "smd_force_const*(" + " + ".join(expr_list) + ")"
        self._openmm_cv_object = openmm.CustomCVForce(full_expr)
        self._openmm_cv_object.addGlobalParameter("smd_force_const", self.sampling_method.force_constant)
        for i in range(len(expr_list)):
            force = force_list[i]
            cv_name = f"cv_{i}"
            self._openmm_cv_object.addGlobalParameter(f"value_{i}", 0.0)
            self._openmm_cv_object.addCollectiveVariable(cv_name, force)
            self._openmm_cv_list.append(force)

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
        cv_values = self._openmm_cv_object.getCollectiveVariableValues(self._simulation_obj.context)
        current_anchor_index = self.find_current_anchor_index(cv_values)
        self.save_to_current_anchor(current_anchor_index, cv_values, SMD_NAME)

        filled_anchors, unfilled_anchors = completion_criteria_runner.filled_and_unfilled_anchors(
            model, stage)
        self._procedure = sampling_methods_runner.get_smd_procedure(
            self._model, filled_anchors, unfilled_anchors)
        logging.info("Steered MD procedure summary:")
        for step in self._procedure:
            logging.info(f"anchor {step[0]} to anchor {step[1]}")
        new_source_anchor_index = self._procedure[self._current_step_in_procedure][0]
        self.load_current_anchor(new_source_anchor_index)
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
        for variable_key in source_anchor.variables:
            var_cv = int(variable_key.split("_")[1])
            if variable_key in target_anchor.variables:
                start_value = cv_values[var_cv]
                last_value = target_anchor.variables[variable_key]
                displacement = (last_value - start_value)
                total_sq_distance += displacement ** 2
                
        distance = np.sqrt(total_sq_distance)
        total_time_ps = 1000.0 * distance / self.sampling_method.velocity
        timestep = self._model.get_md_timestep()
        total_num_timesteps = int(total_time_ps / timestep)
        num_smd_windows = (total_num_timesteps // self._steps_between_evals) + 1

        for variable_key in source_anchor.variables:
            var_cv = int(variable_key.split("_")[1])
            if variable_key in target_anchor.variables:
                start_value = cv_values[var_cv]
                last_value = target_anchor.variables[variable_key]
                windows = np.linspace(start_value, last_value, num_smd_windows)
                windows_list_unzipped.append(windows)

        self._windows_list_zipped = list(zip(*windows_list_unzipped))
        distance = np.sqrt(total_sq_distance)
        steps_in_window = int(abs(distance) / (self.sampling_method.velocity * timestep))
        assert steps_in_window > 0
        self._on_window = 0
        return

    def perform_eval(
            self, 
            restart_index: int, 
            swarm_index: int
            ) -> None:
        """
        Perform an evaluation at a certain point in the simulation.
        """
        assert self._model is not None
        if self._model.anchors is None:
            return
        target_anchor_index = self._procedure[self._current_step_in_procedure][1]
        target_anchor: structures.Anchor = self._model.anchors[target_anchor_index]
        #cv_values_dummy: list[float] = []
        assert self._simulation_obj is not None
        assert self._openmm_cv_list is not None
        #for i, cv in enumerate(self._openmm_cv_list):
        #    cv_value = cv.getCollectiveVariableValues(self._simulation_obj.context)
        #    cv_values_dummy.append(cv_value[0])
        cv_values = self._openmm_cv_object.getCollectiveVariableValues(self._simulation_obj.context)
        
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
            self._current_step_in_procedure += 1
            self.save_to_current_anchor(target_anchor_index, cv_values, SMD_NAME)
            if self._current_step_in_procedure < len(self._procedure):
                new_source_anchor_index = self._procedure[self._current_step_in_procedure][0]
                self.load_current_anchor(new_source_anchor_index)
                cv_values = self._openmm_cv_object.getCollectiveVariableValues(self._simulation_obj.context)
                self.assign_new_windows(cv_values)
            else:
                # Simulation is complete: let the completion criteria finish
                pass



@define
class MMVT_md_openmm_interface(Base_md_openmm_interface):
    """
    Interface for the MMVT MD sampling method using the OpenMM engine
    """
    _stage_index: int
    sampling_method: sampling_methods_structures.MMVT_md_sampling_method
    _model: structures.Seekr_model | None = None
    _simulation_obj: openmm_app.Simulation | None = None
    _steps_between_evals: int = 0
    _openmm_cv_list: list[typing.Any] | None = None
    _anchor_obj: structures.Anchor | None = None

    def add_global_variables_integrator(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add global variables to an OpenMM CustomIntegrator object.

        Some global variables for milestoning method.
        """
        integrator: openmm.System = kwargs["integrator"]
        # TODO: make anchor and/or integrator an attribute of this class?
        anchor: structures.Anchor = kwargs["anchor"]
        self._anchor_obj = anchor
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

        Some per-DOF variables for MetaD.
        """
        
        integrator: openmm.System = kwargs["integrator"]
        integrator.addPerDofVariable("old_x", 0)
        integrator.addPerDofVariable("old_v", 0)
        for i in range(BUFFER_FRAMES):
            integrator.addPerDofVariable(f"encounter_pos_{i}", 0)
        return None
    
    def pre_integration_instructions(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add pre-integration instructions to an OpenMM CustomIntegrator object.

        Some steps to do for milestoning.
        """
        integrator: openmm.System = kwargs["integrator"]
        integrator.addComputePerDof("old_x", "x")
        integrator.addComputePerDof("old_v", "v")
        anchor = self._anchor_obj
        assert anchor is not None
        # Check to make sure no encounter on first step.
        integrator.beginIfBlock("step_count = 0")
        #integrator.beginIfBlock(f"energy1 > 0")
        #integrator.addComputeGlobal("encounter_this_step", "1")
        #integrator.addComputeGlobal("first_step_encounter", "1")
        #integrator.endBlock()
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
        # TODO: 
        # 1. Obtain values of relevant force groups
        # 2. Detect encounters based on this
        # 3. If a encounter occurred, reverse velocities
        integrator: openmm.System = kwargs["integrator"]
        #anchor: structures.Anchor = kwargs["anchor"]
        anchor = self._anchor_obj
        assert anchor is not None
        integrator.addComputeGlobal("bitvar", "energy1")
        
        for j, milestone in enumerate(anchor.milestones):
            i = len(anchor.milestones) - (j + 1)
            integrator.addComputeGlobal("threshold", f"(2^{i})-0.5")
            integrator.beginIfBlock(f"bitvar > threshold")
            integrator.addComputeGlobal(f"value_{i+1}", "1")
            integrator.addComputeGlobal("bitvar", f"bitvar - (2^{i})")
            integrator.endBlock()
            
        for i, milestone in enumerate(anchor.milestones):
            alias_index_i = i+1
            integrator.beginIfBlock(f"value_{alias_index_i} > 0")
            integrator.addComputeGlobal("encounter_this_step", "1")
            integrator.beginIfBlock(f"last_milestone_alias_touched > -1")
            for j, milestone2 in enumerate(anchor.milestones):
                alias_index_j = j+1
                integrator.beginIfBlock(f"last_milestone_alias_touched = {alias_index_j}")
                integrator.addComputeGlobal(f"milestone_{alias_index_i}_{alias_index_j}_count", f"milestone_{alias_index_i}_{alias_index_j}_count + 1")
                integrator.endBlock()
            integrator.addComputeGlobal(f"milestone_{alias_index_i}_time", f"milestone_{alias_index_i}_time + time_count - last_milestone_time_touched")
            integrator.addComputeGlobal("last_milestone_time_touched", "time_count")
            integrator.addComputeGlobal("last_milestone_alias_touched", f"{alias_index_i}")
            integrator.endBlock() # End If last_milestone_alias_touched > -1
            
            integrator.beginIfBlock("last_milestone_alias_touched = -1")
            integrator.addComputeGlobal("last_milestone_alias_touched", f"{alias_index_i}")
            integrator.addComputeGlobal("first_encounter_time", "time_count")
            integrator.addComputeGlobal("last_milestone_time_touched", "time_count")
            integrator.endBlock() # End If last_milestone_alias_touched = -1
            
            integrator.endBlock() # End If value_{alias_index_i} > 0
            integrator.addComputeGlobal(f"value_{alias_index_i}", "0")

        # If using MMVT, then reverse velocities.
        integrator.beginIfBlock(f"encounter_this_step = 1")
        integrator.addComputePerDof("v", "-old_v")
        integrator.addComputePerDof("x", "old_x")
        integrator.endBlock()

        integrator.addComputeGlobal("step_count", "step_count + 1")
        integrator.addComputeGlobal("time_count", "time_count + dt")
        integrator.addComputeGlobal("encounter_this_step", "0")

        return None
    
    def pre_simulation_preparations(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Perform any necessary preparations before the simulation starts.
        """
        system: openmm.System = kwargs["system"]
        model: structures.Seekr_model = kwargs["model"]
        self._model = model
        md_engine = model.md_engine
        assert md_engine is not None
        if self._openmm_cv_list is None:
            self._openmm_cv_list = []
        assert md_engine.type == "openmm", \
            "Only OpenMM engines can be used in the openmm_interface module."
        
        anchor = self._anchor_obj
        assert anchor is not None
        anchor_cv_indices = anchor.get_cv_indices()
        num_cvs = len(anchor_cv_indices)
        boundary_expr_list: list[str] = []
        if num_cvs == 1:
            my_cv_index = anchor_cv_indices[0]
            my_cv = model.collective_variables[my_cv_index]
            cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                my_cv, md_engine)
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
                        cv = model.collective_variables[cv_index]
                        cv_interface = collective_variables_openmm_interface.assign_cv_interface(
                            cv, md_engine)
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
            self._openmm_cv_list.append(openmm_cv_object)
            
        assert self._simulation_obj is not None
        self._simulation_obj.context.reinitialize(preserveState=True)
        return

    def perform_eval(
            self, 
            restart_index: int, 
            swarm_index: int
            ) -> None:
        """
        This is when milestoning information is written to the output file.
        """
        assert self._model is not None
        assert self._simulation_obj is not None
        integrator = self._simulation_obj.context.getIntegrator()
        anchor = self._anchor_obj
        assert anchor is not None
        assert anchor.stage_parameters[self._stage_index+1] is not None
        anchor_directory = os.path.join(
            self._model.directory, anchor.directory,
            anchor.stage_parameters[self._stage_index+1]["directory"])
        output_basename = f"milestoning_restart{restart_index}_swarm_{swarm_index}.out"
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
        return
    
    def perform_early_simulation_eval(
            self, 
            restart_index: int, 
            swarm_index: int
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
    elif sampling_method.type == "steered":
        return Steered_md_openmm_interface(stage_index, sampling_method)
    elif sampling_method.type == "mmvt":
        return MMVT_md_openmm_interface(stage_index, sampling_method)
    else:
        return None