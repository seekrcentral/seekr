"""
modules/engines/openmm/runner.py

Runner object for OpenMM simulation stage.
"""

import os
import sys
import glob
import time
import logging

from attrs import define
import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.run_base as run_base
import seekr.modules.structures as structures
import seekr.modules.engines.openmm.structures as openmm_structures
import seekr.modules.collective_variables.prepare as cv_preparer
import seekr.modules.engines.openmm.sim as sim_openmm
import seekr.modules.sampling_methods.openmm_interface as sampling_openmm_interface
import seekr.modules.completion_criteria.openmm_interface as completion_openmm_interface
import seekr.modules.restraints.openmm_interface as restraints_openmm_interface

@define
class Runner_openmm():
    """
    Prepare and run SEEKR2 simulation using the OpenMM simulation 
    engine.
    """
    model: structures.Seekr_model
    stage_output_index: int
    anchor_index: int | None

    # TODO: fill out argument typing
    def prepare(
            self, 
            restart=False, 
            force_overwrite=False, 
            swarm_index=None
            ):
        """
        This function gets run before the sim_openmm object is created
        so that the proper paths can be found, etc.
        """
        if self.anchor_index is None:
            anchor = None
            anchor_systemwide = model.md_engine.systemwide_settings
            self.md_parameters_topology \
                = model.md_engine.systemwide_settings.md_parameters_topology
        else:
            anchor = model.anchors[self.anchor_index]
            anchor_systemwide = anchor
            self.md_parameters_topology = anchor.md_parameters_topology

        stage_output = model.stages[self.stage_output_index]
        anchor_systemwide_dir = anchor_systemwide.directory
        stage_output_directory \
            = anchor_systemwide.stage_parameters[self.stage_output_index]["directory"]
        self.ensemble = stage_output.ensemble
        # Sampling method interfaces
        self.sampling_method = stage_output.sampling_method
        self.sampling_engine_interface = sampling_openmm_interface\
            .assign_sampling_method_interface(
                self.sampling_method, self.model.md_engine)
        # Completion criteria interfaces.
        self.completion_criteria = stage_output.completion_criteria
        self.completion_criteria_interface = completion_openmm_interface\
            .assign_completion_interface(
                self.completion_criteria, self.model.md_engine)
        # Restraints interfaces.
        self.restraints_interfaces = []
        for restraint in stage_output.restraints:
            self.restraints_interfaces.append(
                restraints_openmm_interface.assign_restraints_interface(
                    restraint, self.model)
            )
        # TODO: check in loaded modules from model JSON to see if any of these
        #  modules contain a "assign_interface" function, and run them.
        assert self.sampling_engine_interface is not None, \
            "The sampling method type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {self.sampling_method.type}, "\
            f"engine: {self.model.md_engine.type}."
        assert self.completion_criteria_interface is not None, \
            "The completion criteria type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {self.completion_criteria.type}, "\
            f"engine: {self.model.md_engine.type}."
        
        input_stage_index = stage_output.input_stage_index
        #input_stage = model.stages[input_stage_index]
        if stage_output.input_from_systemwide_settings:
            input_stage_parameter \
                = model.md_engine.systemwide_settings\
                    .stage_parameters[input_stage_index]
        else:
            input_stage_parameter = anchor.stage_parameters[input_stage_index]
        
        self.input_positions = input_stage_parameter["positions"]
        self.input_stage_directory = input_stage_parameter["directory"]
        self.input_directory = os.path.join(
            model.directory, anchor_systemwide_dir, self.input_stage_directory)
        self.output_directory = os.path.join(
            model.directory, anchor_systemwide_dir, stage_output_directory)
        self.basename = stage_output.sampling_method.get_output_basename()
        self.extension = stage_output.sampling_method.get_output_extension()
        self.glob = stage_output.sampling_method.get_output_glob()
        self.checkpoint_information_filename = None
        self.restart_checkpoint_filename = None
        self.restart_checkpoint_interval = None
        self.start_step = None
        self.min_steps = None
        self.max_steps = None
        self.steps_per_chunk = None
        self.swarm_string = ""
        model = self.model
        output_stage = model.stages[self.stage_output_index]
        assert self.model.md_engine.type == "openmm", \
            "This model was not prepared for OpenMM."
        restart_index = 1

        if swarm_index is None:
            self.swarm_string = ""
        else:
            self.swarm_string = ".swarm_{}".format(swarm_index)
            self.glob = f"{output_stage.sampling_method.get_output_basename()}"\
                        f"{self.swarm_string}*."\
                        f"{output_stage.sampling_method.get_output_extension()}"
        output_files_glob = os.path.join(
            self.output_directory, self.glob)
        output_restarts_list = glob.glob(output_files_glob)
        if restart:
            output_restarts_list = base.order_files_numerically(
                output_restarts_list)
            assert len(output_restarts_list) > 0, \
                "No simulation has yet been run: cannot use restart mode."
            restart_index = len(output_restarts_list) + 1
            default_output_filename = os.path.join(
                self.output_directory, 
                "%s%s.restart%d.%s" % (self.basename, self.swarm_string, 
                                       restart_index, self.extension))
        else:
            if len(output_restarts_list) > 0:
                # if not force_overwrite and not umbrella_restart_mode:
                # TODO: extract something from sampling method to avoid restarts?
                if not force_overwrite:
                    logging.critical(
                        "This anchor already has existing output files "\
                        "and the entered command would overwrite them. "\
                        "If you desire to overwrite the existing files, "\
                        "then use the --force_overwrite (-f) option, and "\
                        "all outputs will be deleted and replace by a new "\
                        "run.")
                    raise Exception("Cannot overwrite existing outputs.")
                else:
                    if self.anchor is None:
                        cv_preparer.cleanse_outputs(model.md_engine.systemwide_settings, model)
                    else:
                        cv_preparer.cleanse_outputs(self.anchor, model)
                
            
            default_output_filename = os.path.join(
                self.output_directory, 
                "%s%s.restart%d.%s" % (self.basename, self.swarm_string, 1, 
                               self.extension))
        
        checkpoint_information_filename = run_base\
            .get_checkpoint_information_basename(swarm_index)
        restart_checkpoint_basename = run_base\
            .get_restart_checkpoint_filename(swarm_index)
        self.checkpoint_information_filename = os.path.join(
            self.output_directory, checkpoint_information_filename)
        self.restart_checkpoint_filename = os.path.join(
            self.output_directory, restart_checkpoint_basename)
        
        return default_output_filename, restart_index
    
    # TODO: fill out argument typing
    def run(
            self, 
            sim_openmm_obj, 
            restart=False, 
            restart_index=1
            ):
        """
        Run the SEEKR simulation.
        """
        self.sim_openmm = sim_openmm_obj
        self.sampling_engine_interface.assign_sim_openmm(self.sim_openmm)
        model = self.model
        output_stage = model.stages[self.stage_output_index]
        self.restart_checkpoint_interval \
            = output_stage.md_restart_checkpoint_interval
        simulation = self.sim_openmm.simulation
        # TODO: have multiple kinds of simulation? Like umbrella stage, other stages?
        # This can be handled by making elber umbrella vs. elber reverse vs. elber forward
        if restart:
            simulation.loadCheckpoint(self.restart_checkpoint_filename)
            currentStep = simulation.context.getState().getStepCount()
            self.start_step = int(
                currentStep // self.restart_checkpoint_interval)
            
            simulation.currentStep = currentStep
            logging.info("restarting from saved checkpoint:", 
                  self.restart_checkpoint_filename, "at step:", currentStep)
            # see how many restart files have already been created
            traj_filename = os.path.join(
                self.output_directory, 
                f"{self.basename}{restart_index}{self.swarm_string}.dcd"
            )
        else:
            self.start_step = 0
            traj_filename = os.path.join(
                self.output_directory, 
                f"{self.basename}1{self.swarm_string}.dcd")
        
        self.min_steps = output_stage.completion_criteria.get_min_sim_len()
        self.max_steps = output_stage.completion_criteria.get_max_sim_len()
        self.sampling_engine_interface.pre_simulation_preparations(
            system=self.sim_openmm.system,
            model=model)
        use_state_reporter, use_traj_reporter\
            = self.sampling_engine_interface.add_reporters(traj_filename)
        if use_state_reporter:
            self.add_state_reporter()
        if use_traj_reporter:
            self.add_traj_reporter(traj_filename)
        lock_filename = os.path.join(
            self.output_directory, "LOCK")
        starttime = time.time()
        num_steps = self.sampling_engine_interface.run(
            steps_between_checkpoints=self.steps_per_chunk,
            steps_between_evals\
                =output_stage.completion_criteria.get_evaluation_interval(),
            start_step=self.start_step,
            min_steps=self.min_steps, max_steps=self.max_steps,
            checkpoint_filename=self.restart_checkpoint_filename,
            lock_filename=lock_filename,
            completion_openmm_interface=self.completion_criteria_interface,)
            #restraints_openmm_interfaces=self.restraints_interfaces)
        total_time = time.time() - starttime
        ns_per_day = num_steps * self.steps_per_chunk\
            * self.sim_openmm.timestep * 1e-3 \
            * 86400 / total_time
        # TODO: log anchor information or log to a different file
        #  to differentiate benchmarks. Write to file within directory?
        # TODO: optionally include a benchmarking mode - run a
        #  short simulation, report benchmark, then continue run.
        #  "estimated benchmark" and "final benchmark"
        logging.info("Benchmark (ns/day):", ns_per_day)
        return
    
    def add_state_reporter(self):
        stage_output = self.model.stages[self.stage_output_index]
        state_reporter_interval = stage_output.md_state_output_interval
        if state_reporter_interval is not None:
            self.sim_openmm.simulation.reporters.append(
                openmm_app.StateDataReporter(
                    # TODO: output to log
                    sys.stdout, state_reporter_interval, step=True, 
                    potentialEnergy=True, temperature=True, volume=True))
        # TODO: implement something like this - using a method in the stage
        # If stage_output.get_num_production_steps() is not None:
        #    if energy_reporter_interval > stage_output.get_num_production_steps():
        #        energy_reporter_interval = stage_output.get_num_production_steps()
        return

    def add_traj_reporter(self, traj_filename):
        stage_output = self.model.stages[self.stage_output_index]
        traj_reporter_interval = stage_output.md_trajectory_reporter_interval
        if traj_reporter_interval is not None:
            self.sim_openmm.simulation.reporters.append(
                openmm_app.DCDReporter(traj_filename, 
                    traj_reporter_interval, enforcePeriodicBox=False))
        # TODO: implement something like this - using a method in the stage
        #if self.restart_checkpoint_interval is not None:
        #        assert trajectory_reporter_interval >= \
        #            self.restart_checkpoint_interval
        #    if trajectory_reporter_interval \
        #            > calc_settings.num_production_steps:
        #        trajectory_reporter_interval \
        #            = calc_settings.num_production_steps
        return
    
def run_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        anchor_index: int, 
        restart: bool, 
        total_simulation_length: int, 
        swarm_index: int | None = None, 
        device_index: str | None = None,
        force_overwrite: bool = False):
    """
    Run an OpenMM simulation.
    """
    md_engine = model.md_engine
    assert md_engine is not None, "No MD engine is set in the model."
    assert md_engine.type == "openmm", "The MD engine is not OpenMM."
    if anchor_index is not None:
        assert anchor_index >= 0, "Invalid anchor index provided. "\
            "Anchor index must be non-negative."
        assert anchor_index < len(model.anchors), \
            "Invalid anchor index provided. Anchor index must be less "\
            "than the number of anchors."
    if device_index is not None:
        assert md_engine.platform_settings is not None, \
            "A device_index is provided, yet a platform is not set."
        md_engine.platform_settings.device_index = device_index

    # NOTE: does old_num_production_steps need to be saved and reset?
    runner = runner.Runner_openmm(model, stage_index, anchor_index)
    default_output_file, restart_index = runner.prepare(
        restart, force_overwrite, swarm_index)
    if swarm_index is None:
        frame = 0
    else:
        frame = swarm_index
    # TODO: get info from enhanced sampling object to pass to next func
    #sim_openmm_obj = md_engine.sampling_method.create_sim_openmm()
    if sim_openmm_obj is None:
        sim_openmm_obj = sim_openmm.create_sim_openmm(
            model, runner.input_directory, runner.input_positions, 
            runner.md_parameters_topology, frame, runner.ensemble,
            sampling_engine_interface=runner.sampling_engine_interface, 
            restraints_interfaces=runner.restraints_interfaces)
            
    runner.run(sim_openmm_obj, restart, restart_index)
