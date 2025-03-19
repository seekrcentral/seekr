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
import seekr.modules.run_base as run_base
import seekr.modules.sampling_methods.openmm_interface as openmm_interface

class Runner_openmm():
    """
    Prepare and run SEEKR2 simulation using the OpenMM simulation 
    engine.
    
    Attributes:
    -----------
    model : Model()
        The model contains all the settings, parameters, directories,
        and file names used to perform a SEEKR2 simulation.
        
    sim_openmm : Sim_openmm()
        The sim_openmm object contains all the OpenMM simulation
        objects, including the Simulation(), System(), Integrator(),
        etc.
    
    stage_index : int
        The stage index for which this runner is preparing and running

    anchor_index : int
        The anchor index for this simulation.
    
    output_directory : str
        A path to the directory where output will be written.
    
    """
    # TODO: fill out argument typing
    def __init__(
            self, 
            model, 
            stage_index, 
            anchor_index):
        self.model = model
        self.sim_openmm = None
        self.stage_output_index = stage_index
        self.anchor_index = anchor_index
        if anchor_index is None:
            anchor = None
            anchor_systemwide = model.md_engine.systemwide_settings
            self.md_parameters_topology \
                = model.md_engine.systemwide_settings.md_parameters_topology
        else:
            anchor = model.anchors[anchor_index]
            anchor_systemwide = anchor
            self.md_parameters_topology = anchor.md_parameters_topology

        stage_output = model.stages[stage_index]
        anchor_systemwide_dir = anchor_systemwide.directory
        stage_output_directory \
            = anchor_systemwide.stage_parameters[stage_index]["directory"]
        self.ensemble = stage_output.ensemble
        self.sampling_method = stage_output.sampling_method
        self.sampling_engine_interface = openmm_interface.assign_interface(
            self.sampling_method, self.model.md_engine)
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
        self.start_chunk = None
        self.end_chunk = None
        self.steps_per_chunk = None
        self.swarm_string = ""
    
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
        model = self.model
        output_stage = model.stages[self.stage_output_index]
        assert self.model.md_engine.type == "openmm", "This model was not prepared for OpenMM."
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
                        model.md_engine.systemwide_settings.cleanse_outputs(model)
                    else:
                        self.anchor.cleanse_outputs(model)
                
            
            default_output_filename = os.path.join(
                self.output_directory, 
                "%s%s.restart%d.%s" % (self.basename, self.swarm_string, 1, 
                               self.extension))
        
        checkpoint_information_filename = run_base\
            .get_checkpoint_information_filename(swarm_index)
        restart_checkpoint_basename = run_base\
            .get_restart_checkpoint_filename(swarm_index)
        self.checkpoint_information_filename = os.path.join(
            self.output_directory, checkpoint_information_filename)
        self.restart_checkpoint_filename = os.path.join(
            self.output_directory, restart_checkpoint_basename)
        
        return default_output_filename, restart_index
    
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
            self.start_chunk = int(
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
            self.start_chunk = 0
            traj_filename = os.path.join(
                self.output_directory, 
                f"{self.basename}1{self.swarm_string}.dcd")
        
        self.sampling_engine_interface.pre_simulation_preparations()
        use_state_reporter, use_traj_reporter\
            = self.sampling_engine_interface.add_reporters(traj_filename)
        if use_state_reporter:
            self.add_state_reporter()
        if use_traj_reporter:
            self.add_traj_reporter(traj_filename)
        
        starttime = time.time()
        num_steps = self.sampling_engine_interface.run(traj_filename)
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
    
    def run_mmvt(self, traj_filename):
        """Run the SEEKR2 MMVT calculation."""
        calc_settings = self.model.calculation_settings
        simulation = self.sim_openmm.simulation
        trajectory_reporter_interval \
            = calc_settings.trajectory_reporter_interval
        energy_reporter_interval = calc_settings.energy_reporter_interval
        traj_reporter = self.sim_openmm.traj_reporter
        if trajectory_reporter_interval is not None:
            simulation.reporters.append(traj_reporter(
                traj_filename, trajectory_reporter_interval, 
                #enforcePeriodicBox=False))
                # Turning this on
                enforcePeriodicBox=None))
            if self.restart_checkpoint_interval is not None:
                assert trajectory_reporter_interval >= \
                    self.restart_checkpoint_interval
            if trajectory_reporter_interval \
                    > calc_settings.num_production_steps:
                trajectory_reporter_interval \
                    = calc_settings.num_production_steps
            
        if energy_reporter_interval is not None:
            simulation.reporters.append(
                self.sim_openmm.energy_reporter(
                    sys.stdout, energy_reporter_interval, step=True, 
                    potentialEnergy=True, temperature=True, volume=True))
            if energy_reporter_interval > calc_settings.num_production_steps:
                energy_reporter_interval = calc_settings.num_production_steps
            
        if self.restart_checkpoint_interval is not None:
            if self.restart_checkpoint_interval \
                    > calc_settings.num_production_steps:
                self.restart_checkpoint_interval \
                    = calc_settings.num_production_steps
            self.end_chunk = calc_settings.num_production_steps \
                // self.restart_checkpoint_interval
            self.steps_per_chunk = self.restart_checkpoint_interval
        else:
            self.end_chunk = self.start_chunk + 1
            self.steps_per_chunk = calc_settings.num_production_steps
        
        lock_filename, lock_string = create_lock(self.model, self.anchor)
        for chunk in range(self.start_chunk, self.end_chunk):
            if self.save_one_state_for_all_boundaries:
                if all_boundaries_have_state(self.state_prefix+"*", 
                                             self.anchor):
                    if self.save_all_states:
                        self.sim_openmm.integrator.setSaveStateFileName(
                            self.state_prefix)
                    else:
                        self.sim_openmm.integrator.setSaveStateFileName("")
                        self.save_one_state_for_all_boundaries = False
                    bounce_counter = get_last_bounce(
                        self.sim_openmm.output_filename)
                    if bounce_counter is None:
                        bounce_counter = self.start_bounce_counter
                    self.sim_openmm.integrator.setBounceCounter(bounce_counter)
                    self.sim_openmm.simulation.context.reinitialize(
                        preserveState=True)
                    
                else:
                    self.sim_openmm.integrator.setSaveStateFileName(
                        self.state_prefix)
                    bounce_counter = get_last_bounce(
                        self.sim_openmm.output_filename)
                    if bounce_counter is None:
                        bounce_counter = self.start_bounce_counter
                    self.sim_openmm.integrator.setBounceCounter(bounce_counter)
                    self.sim_openmm.simulation.context.reinitialize(
                        preserveState=True)
            
            if chunk > 0:    
                saveCheckpoint(self.sim_openmm, self.restart_checkpoint_filename)
            
            self.sim_openmm.simulation.step(self.steps_per_chunk)
            lock_string = update_lock(lock_filename, lock_string, self.anchor)
        
        # TODO: save validation checkpoints at chosen intervals, including the beginning
        saveCheckpoint(self.sim_openmm, self.restart_checkpoint_filename)
        os.remove(lock_filename)
        
        return
    
    def run_elber(self, traj_filename):
        """Run the SEEKR2 Elber calculation."""
        openmm_settings = self.model.openmm_settings
        calc_settings = self.model.calculation_settings
        umbrella_simulation = self.sim_openmm.umbrella_simulation
        umbrella_trajectory_reporter_interval \
            = calc_settings.umbrella_trajectory_reporter_interval
        umbrella_energy_reporter_interval \
            = calc_settings.umbrella_energy_reporter_interval
        umbrella_traj_reporter = self.sim_openmm.umbrella_traj_reporter
        directory = os.path.dirname(traj_filename)
        basename = os.path.basename(traj_filename)
        suffix = re.sub(elber_cv_base.OPENMM_ELBER_BASENAME, "", basename)
        umbrella_basename = elber_cv_base.ELBER_UMBRELLA_BASENAME+suffix
        umbrella_traj_filename = os.path.join(directory, umbrella_basename)
        if umbrella_trajectory_reporter_interval is not None \
                and not self.umbrellas_already_exist_mode:
            umbrella_simulation.reporters.append(umbrella_traj_reporter(
                umbrella_traj_filename, umbrella_trajectory_reporter_interval, 
                #enforcePeriodicBox=False))
                enforcePeriodicBox=None))
            if calc_settings.num_umbrella_stage_steps \
                    < umbrella_trajectory_reporter_interval:
                umbrella_trajectory_reporter_interval \
                    = calc_settings.num_umbrella_stage_steps
            #assert umbrella_trajectory_reporter_interval <= calc_settings.num_umbrella_stage_steps, \
            #    "The umbrella trajectory reporter interval must be less or equal to "\
            #    "the length of the simulation."
        
        if umbrella_energy_reporter_interval is not None:
            umbrella_simulation.reporters.append(
                self.sim_openmm.umbrella_energy_reporter(
                    sys.stdout, umbrella_energy_reporter_interval, step=True, 
                    potentialEnergy=True, temperature=True, volume=True))
            if calc_settings.num_umbrella_stage_steps \
                    < umbrella_energy_reporter_interval:
                umbrella_energy_reporter_interval \
                    = calc_settings.num_umbrella_stage_steps
            #assert umbrella_energy_reporter_interval \
            #        <= calc_settings.num_umbrella_stage_steps, \
            #    "The umbrella energy reporter interval must be less "\
            #    "than or equal to the length of the simulation."
        
        rev_simulation = self.sim_openmm.rev_simulation
        rev_trajectory_reporter_interval \
            = calc_settings.rev_trajectory_reporter_interval
        rev_energy_reporter_interval \
            = calc_settings.rev_energy_reporter_interval
        rev_traj_reporter = self.sim_openmm.rev_traj_reporter
                
        fwd_simulation = self.sim_openmm.fwd_simulation
        fwd_trajectory_reporter_interval \
            = calc_settings.fwd_trajectory_reporter_interval
        fwd_energy_reporter_interval \
            = calc_settings.fwd_energy_reporter_interval
        fwd_traj_reporter = self.sim_openmm.fwd_traj_reporter
        
        assert calc_settings.fwd_rev_interval \
            < calc_settings.num_umbrella_stage_steps, \
            "The interval between reversal/forward trajectories must be less "\
            "than the length of the umbrella simulation."
        assert calc_settings.num_umbrella_stage_steps \
            % calc_settings.fwd_rev_interval == 0, \
            "The number of umbrella stage steps should be a multiple of the "\
            "interval between reversal/forward trajectories."
        
        self.end_chunk = calc_settings.num_umbrella_stage_steps \
            // calc_settings.fwd_rev_interval
        self.steps_per_chunk = calc_settings.fwd_rev_interval
        rev_data_file_name = self.sim_openmm.rev_output_filename
        if os.path.exists(rev_data_file_name):
            os.remove(rev_data_file_name)
        fwd_data_file_name = self.sim_openmm.fwd_output_filename
        if os.path.exists(fwd_data_file_name):
            os.remove(fwd_data_file_name)
        num_errors = 0
        
        # If the user provides an existing umbrella trajectory, don't re-
        # simulation, just load the frames and skip to rev and fwd.
        if self.umbrellas_already_exist_mode:
            print("Umbrella trajectories found already existing in "\
                  "anchor {}. Skipping umbrella simulations.".format(
                      self.anchor.index))
            # load DCD using mdtraj and use it as the umbrella state
            umbrella_traj = check.load_structure_with_mdtraj(
                self.model, self.anchor, mode="elber_umbrella")
            assert umbrella_traj.n_frames == self.end_chunk, \
                "Umbrella trajectories found have total length of "\
                "{} frames ".format(umbrella_traj.n_frames) \
                +"while this calculation is configured to run "\
                "{} frames.".format(self.end_chunk)
            assert openmm_settings.barostat is None, "Cannot use existing" \
                "umbrella trajectories in constant pressure mode."
        for chunk in range(self.start_chunk, self.end_chunk):
            if self.umbrellas_already_exist_mode:
                # extract frame from trajectory and load the coordinates
                print("loading umbrella frame {}.".format(chunk))
                umbrella_simulation.context.setPositions(
                    umbrella_traj.openmm_positions(chunk))
                umbrella_state = \
                    umbrella_simulation.context.getState(
                        getPositions=True, getVelocities=True)
            else:
                umbrella_simulation.saveCheckpoint(
                    self.restart_checkpoint_filename)
                umbrella_simulation.step(self.steps_per_chunk)
                umbrella_state = \
                    umbrella_simulation.context.getState(
                        getPositions=True, getVelocities=True)
                
            # for each velocity restart
            for launch_id in range(calc_settings.num_rev_launches):
                crossing_counter = chunk * calc_settings.num_rev_launches \
                    + launch_id
                counter_str = "_%d" % crossing_counter
                if self.save_one_state_for_all_boundaries:
                    if all_boundaries_have_state(self.state_prefix+"*", 
                                                 self.anchor):
                        if self.save_all_states:
                            self.sim_openmm.rev_integrator.setSaveStateFileName(
                                self.state_prefix+counter_str+"r")
                            self.sim_openmm.fwd_integrator.setSaveStateFileName(
                                self.state_prefix+counter_str+"f")
                        else:
                            self.sim_openmm.rev_integrator\
                                .setSaveStateFileName("")
                            self.sim_openmm.fwd_integrator\
                                .setSaveStateFileName("")
                            self.save_one_state_for_all_boundaries = False
                        self.sim_openmm.rev_simulation.context.reinitialize(
                            preserveState=True)
                        self.sim_openmm.fwd_simulation.context.reinitialize(
                            preserveState=True)
                        
                    else:
                        self.sim_openmm.rev_integrator.setSaveStateFileName(
                            self.state_prefix+counter_str+"r")
                        self.sim_openmm.fwd_integrator.setSaveStateFileName(
                            self.state_prefix+counter_str+"f"+str(launch_id))
                        self.sim_openmm.rev_simulation.context.reinitialize(
                            preserveState=True)
                        self.sim_openmm.fwd_simulation.context.reinitialize(
                            preserveState=True)
                rev_simulation.context.setPositions(
                    umbrella_state.getPositions())
                rev_simulation.context.setVelocitiesToTemperature(
                    self.model.openmm_settings.initial_temperature \
                    * unit.kelvin)
                rev_simulation.context.setPeriodicBoxVectors(
                    *umbrella_state.getPeriodicBoxVectors())
                self.sim_openmm.rev_integrator.setCrossingCounter(
                    crossing_counter)
                if rev_trajectory_reporter_interval is not None:
                    rev_traj_filename = os.path.join(
                        self.output_directory, 
                        "reverse_%d.dcd" % crossing_counter)
                    print("rev_traj_filename", rev_traj_filename)
                    rev_simulation.reporters = [rev_traj_reporter(
                        rev_traj_filename, rev_trajectory_reporter_interval, 
                        #enforcePeriodicBox=False)]
                        enforcePeriodicBox=None)]
                if rev_energy_reporter_interval is not None:
                    rev_simulation.reporters.append(
                        self.sim_openmm.rev_energy_reporter(
                            sys.stdout, rev_energy_reporter_interval, step=True, 
                            potentialEnergy=True, temperature=True, 
                            volume=True))
                rev_simulation.context.reinitialize(preserveState=True)
                rev_start_state = rev_simulation.context.getState(
                    getPositions=True, getVelocities=True)
                rev_data_file_length = get_data_file_length(rev_data_file_name)
                rev_block_counter = 0
                had_error = False
                while get_data_file_length(rev_data_file_name) \
                        == rev_data_file_length:
                    rev_data_file_length = get_data_file_length(
                        rev_data_file_name)
                    try:
                        rev_simulation.step(REV_STAGE_STEPS_PER_BLOCK)
                    except Exception: 
                        # if there was a NAN error
                        print("Error encountered. Continuing with the next "\
                              "umbrella frame.")
                        num_errors += 1
                        had_error = True
                        # don't want to log this as a success
                        break 
                    rev_block_counter += 1
                    if rev_block_counter > MAX_REVERSE_ITER:
                        print("maximum iterations exceeded.")
                        break
                
                rev_simulation.context.setTime(0.0)
                rev_simulation.context.setStepCount(0)
                rev_simulation.currentStep = 0
                if had_error == True:
                    # move on to the next frame
                    break 
                
                if read_reversal_data_file_last(rev_data_file_name):
                    fwd_simulation.context.setPositions(
                        rev_start_state.getPositions())
                    fwd_simulation.context.setVelocities(
                        -1.0*rev_start_state.getVelocities())
                    fwd_simulation.context.setPeriodicBoxVectors(
                        *rev_start_state.getPeriodicBoxVectors())
                    self.sim_openmm.fwd_integrator.setCrossingCounter(
                        crossing_counter)
                    fwd_simulation.context.reinitialize(preserveState=True)
                    # TODO: add reporter update here
                    fwd_data_file_length = get_data_file_length(
                        fwd_data_file_name)
                    if fwd_trajectory_reporter_interval is not None:
                        fwd_traj_filename = os.path.join(
                            self.output_directory, 
                            "forward_%d.dcd" % crossing_counter)
                        fwd_simulation.reporters = [fwd_traj_reporter(
                            fwd_traj_filename, fwd_trajectory_reporter_interval, 
                            #enforcePeriodicBox=False)]
                            enforcePeriodicBox=None)]
                    if fwd_energy_reporter_interval is not None:
                        fwd_simulation.reporters.append(
                            self.sim_openmm.fwd_energy_reporter(
                                sys.stdout, fwd_energy_reporter_interval, 
                                step=True, potentialEnergy=True, 
                                temperature=True, volume=True))
                    fwd_block_counter = 0
                    had_error = False
                    while get_data_file_length(fwd_data_file_name) \
                            == fwd_data_file_length:
                        fwd_data_file_length = get_data_file_length(
                            fwd_data_file_name)
                        try:
                            fwd_simulation.step(FWD_STAGE_STEPS_PER_BLOCK)
                        except Exception: 
                            # if there was a NAN error
                            print("Error encountered. Continuing with the "\
                                  "next umbrella frame.")
                            num_errors += 1
                            had_error = True
                            break
                        fwd_block_counter += 1
                        if fwd_block_counter > MAX_FORWARD_ITER:
                            print("maximum iterations exceeded.")
                            break
                    
                    fwd_simulation.context.setTime(0.0)
                    fwd_simulation.context.setStepCount(0)
                    fwd_simulation.currentStep = 0
                                        
        umbrella_simulation.saveCheckpoint(self.restart_checkpoint_filename)
        return