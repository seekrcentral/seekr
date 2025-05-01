"""
modules/engines/openmm/runner.py

Runner object for OpenMM simulation stage.
"""

import os
import sys
import glob
import time
import logging

import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.preparer as preparer
import seekr.modules.engines.runner as engines_runner
import seekr.modules.engines.openmm.sim as sim_openmm
import seekr.modules.sampling_methods.openmm_interface as sampling_openmm_interface
import seekr.modules.completion_criteria.openmm_interface as completion_openmm_interface
import seekr.modules.restraints.openmm_interface as restraints_openmm_interface

# TODO: I like the word "running" more than "output"

class Runner_openmm():
    """
    Prepare and run SEEKR2 simulation using the OpenMM simulation 
    engine.
    """
    def __init__(
            self, 
            model: structures.Seekr_model, 
            stage_output_index: int,
            anchor_index: int | None = None):
        self._model = model
        self._stage_output_index = stage_output_index
        self._anchor_index = anchor_index
        self._ensemble: structures.Ensemble | None = None

    def get_ensemble(self) -> structures.Ensemble:
        assert self._ensemble is not None, "The ensemble is not set."
        return self._ensemble

    def prepare(
            self, 
            restart: bool = False, 
            force_overwrite: bool = False, 
            swarm_index: int | None = None
            ) -> tuple[str, int]:
        """
        This function gets run before the sim_openmm object is created
        so that the proper paths can be found, etc.
        """
        # TODO: find a better way to deal with the swarm index that eliminates the None
        if swarm_index is None:
            swarm_index = 0
        md_parameters_half_directory = None
        if self._anchor_index is None:
            anchor = None
            anchor_systemwide = self._model.md_engine.systemwide_settings
            self.md_parameters_topology \
                = self._model.md_engine.systemwide_settings.md_parameters_topology
            # TODO: handle the situation where the md_parameters_topology changes across stages
            assert self._model.md_engine.systemwide_settings.stage_parameters[0] is not None, \
                "Systemwide lacks an initial stage set of parameters."
            md_parameters_half_directory = os.path.join(
                    self._model.md_engine.systemwide_settings.directory, 
                    self._model.md_engine.systemwide_settings.stage_parameters[0]["directory"])
        else:
            anchor = self._model.anchors[self._anchor_index]
            anchor_systemwide = anchor
            
            if anchor.md_parameters_topology is None:
                assert self._model.md_engine.systemwide_settings.stage_parameters[0] is not None, \
                    "Systemwide lacks an initial stage set of parameters."
                md_parameters_half_directory = os.path.join(
                    self._model.md_engine.systemwide_settings.directory, 
                    self._model.md_engine.systemwide_settings\
                    .stage_parameters[0]["directory"])
                self.md_parameters_topology = self._model.md_engine.systemwide_settings\
                    .md_parameters_topology
            else:
                assert anchor.stage_parameters[0] is not None, \
                    "Anchor lacks an initial stage set of parameters."
                md_parameters_half_directory = os.path.join(
                    anchor.directory, anchor.stage_parameters[0]["directory"])
                self.md_parameters_topology = anchor.md_parameters_topology
            
        assert md_parameters_half_directory is not None, \
            "No initial directory from which to take MD parameters/topology."
        self.md_parameters_directory = os.path.join(
            self._model.directory, md_parameters_half_directory)
        stage_output = self._model.stages[self._stage_output_index]
        anchor_systemwide_dir = anchor_systemwide.directory
        stage_output_directory \
            = anchor_systemwide.stage_parameters[self._stage_output_index+1]["directory"]
        #self._output_positions = anchor_systemwide.stage_parameters[
        #    self._stage_output_index+1]["positions"]
        self._ensemble = stage_output.ensemble
        # Sampling method interfaces
        sampling_method = stage_output.sampling_method
        self._output_positions_basename = sampling_method.get_final_pdb_output_basename(
            toy_mode=self._model.using_toy(),
            frame_number=swarm_index)
        self.sampling_engine_interface = sampling_openmm_interface\
            .assign_sampling_method_interface(
                self._stage_output_index, sampling_method, self._model.md_engine)
        # Completion criteria interfaces.
        completion_criteria = stage_output.completion_criteria
        self.completion_criteria_interface = completion_openmm_interface\
            .assign_completion_interface(
                completion_criteria, self._model, stage_output)
        # Restraints interfaces.
        self.restraints_interfaces = []
        for restraint in stage_output.restraints:
            self.restraints_interfaces.append(
                restraints_openmm_interface.assign_restraints_interface(
                    restraint, self._model.md_engine)
            )
        # TODO: check in loaded modules from model JSON to see if any of these
        #  modules contain a "assign_interface" function, and run them.
        assert self.sampling_engine_interface is not None, \
            "The sampling method type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {sampling_method.type}, "\
            f"engine: {self._model.md_engine.type}."
        assert self.completion_criteria_interface is not None, \
            "The completion criteria type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {completion_criteria.type}, "\
            f"engine: {self._model.md_engine.type}."
        
        input_stage_index = stage_output.input_stage_index
        #input_stage = model.stages[input_stage_index]
        if stage_output.input_from == "systemwide":
            input_stage_parameter \
                = self._model.md_engine.systemwide_settings\
                    .stage_parameters[input_stage_index]
        elif stage_output.input_from == "anchors":
            input_stage_parameter = anchor.stage_parameters[input_stage_index]
        else:
            raise Exception("Invalid input stage type: "\
                f"{stage_output.input_from}.")
        
        if self._model.using_toy():
            extension = "txt"
        else:
            extension = "pdb"
        
        input_stage_directory = input_stage_parameter["directory"]
        self.input_directory = os.path.join(
            self._model.directory, anchor_systemwide_dir, input_stage_directory)
        self._running_directory = os.path.join(
            self._model.directory, anchor_systemwide_dir, stage_output_directory)
        self._basename = stage_output.sampling_method.get_output_positions_basename_noext()
        
        input_positions_glob_basename = f"{input_stage_parameter['positions']}*.swarm_{swarm_index}.{extension}"
        input_positions_glob = os.path.join(
            self.input_directory, input_positions_glob_basename)
        input_positions_filenames = glob.glob(input_positions_glob)
        assert len(input_positions_filenames) == 1, \
            "Ambiguity detected in input positions glob: "\
            f"{input_positions_glob}. "
        self.input_positions = os.path.basename(input_positions_filenames[0])

        glob_ = stage_output.sampling_method.get_output_positions_glob_list()
        
        if stage_output.input_from == "systemwide":
            if stage_output.output_to == "systemwide":
                output_files_glob_dir = os.path.join(
                    self._running_directory)

            elif stage_output.output_to == "anchors":
                anchor_dirs_glob = os.path.join(
                    self._model.directory, "anchor_*", stage_output_directory)
                output_files_glob_dir = os.path.join(
                    anchor_dirs_glob)

            else:
                raise Exception("Invalid output_to")
        elif stage_output.input_from == "anchors":
            if stage_output.output_to == "systemwide":
                systemwide_dir_glob = os.path.join(
                    self._model.directory, "systemwide", stage_output_directory)
                output_files_glob_dir = os.path.join(
                    systemwide_dir_glob)

            elif stage_output.output_to == "anchors":
                output_files_glob_dir = os.path.join(
                    self._running_directory)

            else:
                raise Exception("Invalid output_to")
            
        self._restart_checkpoint_interval = None
        self._start_step = None
        self._min_steps = None
        self._max_steps = None
        self._swarm_index = swarm_index
        self._swarm_string = ""
        #output_stage = self._model.stages[self._stage_output_index]
        assert self._model.md_engine.type == "openmm", \
            "This model was not prepared for OpenMM."
        restart_index = 1
        if swarm_index is None:
            self._swarm_string = ""
        else:
            self._swarm_string = ".swarm_{}".format(swarm_index)
            #glob_ = f"{output_stage.sampling_method.get_output_basename()}"\
            #            f"{self._swarm_string}*."\
            #            f"{output_stage.sampling_method.get_output_extension()}"
        
        #output_files_list = glob.glob(output_files_glob)
        output_files_list = base.glob_or(glob_, output_files_glob_dir)
        checkpoint_information_basename = engines_runner\
            .get_checkpoint_information_basename(swarm_index)
        restart_checkpoint_basename = engines_runner\
            .get_restart_checkpoint_basename(swarm_index)
        self.checkpoint_information_filename = os.path.join(
            self._running_directory, checkpoint_information_basename)
        self.restart_checkpoint_filename = os.path.join(
            self._running_directory, restart_checkpoint_basename)
        if restart:
            assert os.path.exists(self.checkpoint_information_filename), \
                "Previous checkpoint information file does not exist: cannot run restart."
            checkpoint_information = engines_runner.load_checkpoint_information(
                self.checkpoint_information_filename)
            restart_index = checkpoint_information.restart_index + 1
            #default_output_filename = os.path.join(
            #    self._running_directory, 
            #    "%s%s.restart%d.%s" % (self._basename, self._swarm_string, 
            #                           restart_index, extension))
        else:
            if len(output_files_list) > 0:
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
                    for output_file in output_files_list:
                        os.remove(output_file)
                    if anchor is None:
                        preparer.cleanse_outputs_stage(
                            self._model, self._model.md_engine.systemwide_settings, 
                            self._stage_output_index)
                    else:
                        preparer.cleanse_outputs_stage(
                            self._model, anchor, self._stage_output_index)
            
            #default_output_filename = os.path.join(
            #    self._running_directory, 
            #    "%s%s.restart%d.%s" % (self._basename, self._swarm_string, 1, 
            #                   extension))
        
        return restart_index #default_output_filename, restart_index
    
    def run(
            self, 
            sim_openmm_obj: sim_openmm.Sim_openmm, 
            restart: bool = False, 
            restart_index: int = 1
            ) -> None:
        """
        Run the SEEKR simulation.
        """
        self.sim_openmm = sim_openmm_obj
        simulation = self.sim_openmm.simulation
        self.sampling_engine_interface.assign_sim_openmm(simulation)
        output_stage = self._model.stages[self._stage_output_index]
        self._restart_checkpoint_interval \
            = output_stage.md_restart_checkpoint_interval
        # TODO: have multiple kinds of simulation? Like umbrella stage, other stages?
        # This can be handled by making elber umbrella vs. elber reverse vs. elber forward
        if restart:
            simulation.loadCheckpoint(self.restart_checkpoint_filename)
            currentStep = simulation.context.getState().getStepCount()
            self._start_step = currentStep #int(
            #    currentStep // self._restart_checkpoint_interval)
            
            simulation.currentStep = currentStep
            logging.info(
                  "restarting from saved checkpoint:"\
                +f"{self.restart_checkpoint_filename} at step: {currentStep}")
            # see how many restart files have already been created
            traj_filename = os.path.join(
                self._running_directory, 
                f"{self._basename}{restart_index}{self._swarm_string}.dcd"
            )
        else:
            self._start_step = 0
            traj_filename = os.path.join(
                self._running_directory, 
                f"{self._basename}1{self._swarm_string}.dcd")
        
        self._min_steps = output_stage.completion_criteria.get_min_sim_len()
        self._max_steps = output_stage.completion_criteria.get_max_sim_len()
        self.sampling_engine_interface.pre_simulation_preparations(
            system=self.sim_openmm.system,
            model=self._model,
            directory=self._running_directory,)
        use_state_reporter, use_traj_reporter\
            = self.sampling_engine_interface.add_reporters()
        if use_state_reporter:
            self.add_state_reporter()
        if use_traj_reporter:
            self.add_traj_reporter(traj_filename)
        lock_filename = os.path.join(
            self._running_directory, "LOCK")
        steps_between_evals = output_stage.completion_criteria\
            .get_evaluation_interval()
        start_time = time.time()
        num_steps = self.sampling_engine_interface.run(
            steps_between_checkpoints=self._restart_checkpoint_interval,
            steps_between_evals=steps_between_evals,
            start_step=self._start_step, min_steps=self._min_steps, 
            max_steps=self._max_steps, restart_index=restart_index,
            swarm_index=self._swarm_index,
            checkpoint_filename=self.restart_checkpoint_filename,
            checkpoint_info_filename=self.checkpoint_information_filename,
            lock_filename=lock_filename,
            completion_openmm_interface=self.completion_criteria_interface,
            #restraints_openmm_interfaces=self.restraints_interfaces)
            do_initial_benchmark=True)
        total_time = time.time() - start_time
        ns_per_day = num_steps * self.sim_openmm.timestep * 1e-3 \
            * 86400 / total_time
        logging.info(f"Final benchmark (ns/day): {ns_per_day}")
        # Write final PDB file
        if self._output_positions_basename is not None:
            final_positions_filename = os.path.join(
                self._running_directory, self._output_positions_basename)
            if self._model.using_toy():
                sampling_openmm_interface.write_txt_file_from_simulation(
                    final_positions_filename, simulation)
            else:
                sampling_openmm_interface.write_pdb_file_from_simulation(
                    final_positions_filename, simulation)
                #self.sim_openmm.write_pdb_file(final_positions_filename)
        return
    
    def add_state_reporter(self) -> None:
        stage_output = self._model.stages[self._stage_output_index]
        state_reporter_interval = stage_output.md_state_output_interval
        if state_reporter_interval is not None:
            if state_reporter_interval > self._max_steps:
                state_reporter_interval = self._max_steps
            self.sim_openmm.simulation.reporters.append(
                openmm_app.StateDataReporter(
                    # TODO: output to log
                    sys.stdout, state_reporter_interval, step=True, 
                    potentialEnergy=True, temperature=True, volume=True))
        return

    def add_traj_reporter(
            self, 
            traj_filename: str,
            ) -> None:
        stage_output = self._model.stages[self._stage_output_index]
        traj_reporter_interval = stage_output.md_trajectory_output_interval
        if traj_reporter_interval is not None:
            assert traj_reporter_interval >= \
                self._restart_checkpoint_interval
            if traj_reporter_interval > self._max_steps:
                traj_reporter_interval = self._max_steps
            self.sim_openmm.simulation.reporters.append(
                openmm_app.DCDReporter(traj_filename, 
                    traj_reporter_interval, enforcePeriodicBox=False))
        return
    
def run_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        anchor_index: int, 
        restart: bool, 
        swarm_index: int | None = None, 
        device_index: str | None = None,
        force_overwrite: bool = False
        ) -> None:
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
    runner = Runner_openmm(model, stage_index, anchor_index)
    #default_output_file, restart_index = runner.prepare(
    #    restart, force_overwrite, swarm_index)
    restart_index = runner.prepare(restart, force_overwrite, swarm_index)
    if swarm_index is None:
        frame = 0
    else:
        frame = swarm_index
    sim_openmm_obj = None
    # TODO: get sim_openmm_obj from enhanced sampling object to pass to next func?
    #sim_openmm_obj = md_engine.sampling_method.create_sim_openmm()
    if sim_openmm_obj is None:
        sim_openmm_obj = sim_openmm.create_sim_openmm(
            model=model, 
            anchor_index=anchor_index,
            positions_directory=runner.input_directory, 
            positions_basename=runner.input_positions, 
            md_parameters_directory=runner.md_parameters_directory,
            md_parameters_topology=runner.md_parameters_topology, 
            frame=frame, 
            ensemble=runner.get_ensemble(),
            sampling_engine_interface=runner.sampling_engine_interface,
            completion_criteria_interface=runner.completion_criteria_interface, 
            restraints_interfaces=runner.restraints_interfaces)
            
    runner.run(sim_openmm_obj, restart, restart_index)
    return
