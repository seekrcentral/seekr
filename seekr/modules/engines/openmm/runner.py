from __future__ import annotations
"""
modules/engines/openmm/runner.py

Runner object for OpenMM simulation stage.
"""

import os
import time
import logging

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.preparer as preparer
import seekr.modules.engines.runner as engines_runner
import seekr.modules.engines.openmm.sim as sim_openmm
import seekr.modules.sampling_methods.openmm_interface as sampling_openmm_interface
import seekr.modules.completion_criteria.openmm_interface as completion_openmm_interface
import seekr.modules.restraints.openmm_interface as restraints_openmm_interface
import seekr.modules.reporters.openmm_interface as reporters_openmm_interface

logger = logging.getLogger(__name__)

class Runner_openmm():
    """
    Prepare and run SEEKR2 simulation using the OpenMM simulation 
    engine.
    """
    def __init__(
            self, 
            model: structures.Seekr_model, 
            stage_output_index: int,
            anchor_index: int | None = None,
            benchmark: bool = False
            ) -> None:
        self.model = model
        self.stage_output_index = stage_output_index
        self.anchor_index = anchor_index
        self.benchmark = benchmark
        self.ensemble: base.Ensemble | None = None
        self.restart_checkpoint_interval: int | None = None
        self.start_step: int = 0
        self.min_steps: int = 0
        self.max_steps: int = 0
        self.run_minimization: bool = False
        self.timestep: float | None = None

    def get_ensemble(self) -> base.Ensemble:
        assert self.ensemble is not None, "The ensemble is not set."
        return self.ensemble

    def prepare(
            self, 
            restart: bool = False, 
            force_overwrite: bool = False, 
            swarm_id: int = 0
            ) -> int:
        """
        This function gets run before the sim_openmm object is created
        so that the proper paths can be found, etc.
        """
        md_parameters_half_directory = None
        md_engine = self.model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine is not None
        assert md_engine.type == "openmm", \
            "This model was not prepared for OpenMM."
        assert md_engine.unpartitioned_settings is not None, \
            "No unpartitioned settings are set in the model."
        anchor_unpartitioned: structures.Anchor | engines_structures.MD_unpartitioned_settings
        if self.anchor_index is None:
            anchor = None
            anchor_unpartitioned = md_engine.unpartitioned_settings
            self.md_parameters_topology \
                = md_engine.unpartitioned_settings.md_parameters_topology
            # TODO: handle the situation where the md_parameters_topology changes across stages
            assert md_engine.unpartitioned_settings.stage_parameters["initial"] is not None, \
                "Unpartitioned lacks an initial stage set of parameters."
            md_parameters_half_directory = os.path.join(
                    md_engine.unpartitioned_settings.directory, 
                    md_engine.unpartitioned_settings.stage_parameters["initial"]["directory"])
        else:
            anchor = self.model.anchors[self.anchor_index]
            anchor_unpartitioned = anchor
            anchor_md_parameters_topology \
                = anchor.scale_parameters_topology.get("molecular_dynamics")
            if anchor_md_parameters_topology is None:
                assert md_engine.unpartitioned_settings.stage_parameters["initial"] is not None, \
                    "Unpartitioned lacks an initial stage set of parameters."
                md_parameters_half_directory = os.path.join(
                    md_engine.unpartitioned_settings.directory, 
                    md_engine.unpartitioned_settings.stage_parameters["initial"]["directory"])
                self.md_parameters_topology = md_engine.unpartitioned_settings\
                    .md_parameters_topology
            else:
                assert anchor.stage_parameters["initial"] is not None, \
                    "Anchor lacks an initial stage set of parameters."
                md_parameters_half_directory = os.path.join(
                    anchor.directory, anchor.stage_parameters["initial"]["directory"])
                self.md_parameters_topology = anchor_md_parameters_topology
            
        assert md_parameters_half_directory is not None, \
            "No initial directory from which to take MD parameters/topology."
        self.md_parameters_directory = os.path.join(
            self.model.directory, md_parameters_half_directory)
        stage_output = self.model.stages[self.stage_output_index]
        if stage_output.scope == "unpartitioned":
            anchor_unpartitioned_dir = anchor_unpartitioned.directory
        elif stage_output.scope == "partitioned":
            anchor_unpartitioned_dir = os.path.join(
                "anchor_sets", stage_output.set_name, 
                anchor_unpartitioned.directory)
        else:
            raise Exception("Invalid scope type: "
                f"{stage_output.scope}.")
        # MYPY: ignore mypy error here - the dictionary is having trouble with typing
        #stage_output_directory = anchor_unpartitioned.get_stage_output_directory(
        #    self.stage_output_index + 1)
        next_stage_params = anchor_unpartitioned\
            .get_stage_parameters_by_index(self.stage_output_index+1)
        stage_output_directory = next_stage_params["directory"]
        assert stage_output.type == "md_stage"
        self.ensemble = stage_output.ensemble
        self.run_minimization = stage_output.run_minimization
        self.needs_cv_values = False
        # Sampling method interfaces
        sampling_method = stage_output.sampling_method
        self.output_positions_basename = sampling_method.get_final_output_basename(
            toy_mode=self.model.using_toy(),
            frame_number=swarm_id)
        self.sampling_method_interface = sampling_openmm_interface\
            .assign_sampling_method_interface(
                self.stage_output_index, sampling_method, md_engine,
                self.model.plugins)
        self.sampling_method_interface.set_random_seed(
            self.model.get_random_seed()+swarm_id*10000)
        # Reporters interfaces.
        self.reporters_interfaces = []
        for reporter in stage_output.reporters:
            reporter_interface \
                = reporters_openmm_interface.assign_reporter_interface(
                    reporter, md_engine, self.model.plugins,
                    self.model.collective_variables, swarm_id=swarm_id)
            self.reporters_interfaces.append(reporter_interface)
        # Completion criteria interfaces.
        completion_criteria = stage_output.completion_criteria
        self.completion_criteria_interface = completion_openmm_interface\
            .assign_completion_interface(
                completion_criteria, self.model, stage_output, anchor_unpartitioned,
                reporter_interfaces=self.reporters_interfaces)
        if self.completion_criteria_interface.needs_cv_values():
            self.needs_cv_values = True
        # Restraints interfaces.
        self.restraints_interfaces = []
        for restraint in stage_output.restraints:
            self.restraints_interfaces.append(
                restraints_openmm_interface.assign_restraints_interface(
                    restraint, md_engine, self.model.plugins)
            )
        
        # TODO: check in loaded modules from model JSON to see if any of these
        #  modules contain a "assign_interface" function, and run them.
        assert self.sampling_method_interface is not None, \
            "The sampling method type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {sampling_method.type}, "\
            f"engine: {md_engine.type}."
        assert self.completion_criteria_interface is not None, \
            "The completion criteria type is not recognized by the "\
            "openmm_interface module or any loaded module: "\
            f"type: {completion_criteria.type}, "\
            f"engine: {md_engine.type}."
        
        input_stage_index = stage_output.input_stage_index
        if input_stage_index == 0:
            input_stage_name = "initial"
        else:
            input_stage_name = self.model.stages[input_stage_index-1].name
        if stage_output.scope == "unpartitioned":
            input_stage_parameter \
                = md_engine.unpartitioned_settings\
                    .stage_parameters[input_stage_name]
            input_positions_filenames = md_engine.unpartitioned_settings\
                .get_positions_full_paths(self.model.directory, input_stage_index, 
                                          "dummy", swarm_id)
        elif stage_output.scope == "partitioned":
            assert anchor is not None
            input_stage_parameter = anchor.stage_parameters[input_stage_name]
            input_positions_filenames = anchor.get_positions_full_paths(
                self.model.directory, input_stage_index, stage_output.set_name, 
                swarm_id)
            
        else:
            raise Exception("Invalid scope type: "\
                f"{stage_output.scope}.")
        
        # MYPY: ignore mypy error here - the dictionary is having trouble with typing
        input_stage_directory = input_stage_parameter["directory"]
        self.input_directory = os.path.join(
            self.model.directory, anchor_unpartitioned_dir, input_stage_directory)
        self.running_directory = os.path.join(
            self.model.directory, anchor_unpartitioned_dir, stage_output_directory)
        self.basename = stage_output.sampling_method.get_output_positions_basename_noext()
        assert len(input_positions_filenames) == 1, \
            "Ambiguity detected in input positions files: "\
            f"{input_positions_filenames}. "
        self.input_positions = os.path.basename(input_positions_filenames[0])

        glob_ = stage_output.sampling_method.get_output_positions_glob_list(swarm_id=swarm_id)
        
        if stage_output.scope == "unpartitioned":
            output_files_glob_dir = os.path.join(
                self.running_directory)

        elif stage_output.scope == "partitioned":
            anchor_dirs_glob = os.path.join(
                self.model.directory, "anchor_*", stage_output_directory)
            output_files_glob_dir = os.path.join(
                anchor_dirs_glob)

        else:
            raise Exception("Invalid scope type: "\
                f"{stage_output.scope}.")

        if stage_output.timestep is not None:
            self.timestep = stage_output.timestep
            
        self.swarm_id = swarm_id
        self.swarm_string = ""
        assert md_engine.type == "openmm", \
            "This model was not prepared for OpenMM."
        restart_index = 1
        self.swarm_string = ".swarm_{}".format(swarm_id)
            
        output_files_list = base.glob_or(glob_, output_files_glob_dir)
        checkpoint_information_basename = engines_runner\
            .get_checkpoint_information_basename(swarm_id)
        restart_checkpoint_basename = engines_runner\
            .get_restart_checkpoint_basename(swarm_id)
        self.checkpoint_information_filename = os.path.join(
            self.running_directory, checkpoint_information_basename)
        self.restart_checkpoint_filename = os.path.join(
            self.running_directory, restart_checkpoint_basename)
        if restart:
            assert os.path.exists(self.checkpoint_information_filename), \
                "Previous checkpoint information file does not exist: cannot run "\
                "restart. You will probably need to run with --force_overwrite."
            checkpoint_information = engines_runner.load_checkpoint_information(
                self.checkpoint_information_filename)
            restart_index = checkpoint_information.restart_index + 1
            
        else:
            if len(output_files_list) > 0:
                # if not force_overwrite and not umbrella_restart_mode:
                # TODO: extract something from sampling method to avoid restarts?
                if not force_overwrite:
                    anchor_str = "unpartitioned" \
                        if self.anchor_index is None \
                        else f"anchor {self.anchor_index}"
                    logger.critical(
                        f"{anchor_str} already has existing output files "\
                        "and the entered command would overwrite them. "\
                        "If you desire to overwrite the existing files, "\
                        "then use the --force_overwrite (-f) option, and "\
                        "all outputs will be deleted and replace by a new "\
                        "run.")
                    raise Exception("Cannot overwrite existing outputs.")
                else:
                    for output_file in output_files_list:
                        os.remove(output_file)

        return restart_index
    
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
        system = self.sim_openmm.system
        simulation = self.sim_openmm.simulation
        assert self.sampling_method_interface is not None
        self.sampling_method_interface.assign_system_openmm(system)
        self.sampling_method_interface.assign_sim_openmm(simulation)
        output_stage = self.model.stages[self.stage_output_index]
        assert output_stage.type == "md_stage"
        assert self.completion_criteria_interface is not None
        self.restart_checkpoint_interval \
            = output_stage.md_restart_checkpoint_interval
        # TODO: have multiple kinds of simulation? Like umbrella stage, other stages?
        # This can be handled by making elber umbrella vs. elber reverse vs. elber forward
        if restart:
            simulation.loadCheckpoint(self.restart_checkpoint_filename)
            currentStep = simulation.context.getState().getStepCount()
            self.start_step = currentStep 

            simulation.currentStep = currentStep
            logger.info(
                  "restarting from saved checkpoint:"\
                +f"{self.restart_checkpoint_filename} at step: {currentStep}")
            # see how many restart files have already been created
            reporter_basename = os.path.join(
                self.running_directory, 
                f"{self.basename}{restart_index}{self.swarm_string}"
            )
        else:
            self.start_step = 0
            reporter_basename = os.path.join(
                self.running_directory, 
                f"{self.basename}1{self.swarm_string}")
        
        self.min_steps = output_stage.completion_criteria.get_min_sim_len()
        self.max_steps = output_stage.completion_criteria.get_max_sim_len()
        self.sampling_method_interface.pre_simulation_preparations(
            model=self.model,
            directory=self.running_directory,)
        for i, reporter_interface in enumerate(self.reporters_interfaces):
            reporter_ext = reporter_interface.get_reporter_extension()
            reporter_name = reporter_interface._reporter.name
            reporter_filename = reporter_basename + "." + reporter_name \
                + "." + reporter_ext
            reporter_interface.add_sampling_method_interface(self.sampling_method_interface)
            reporter_interface.add_reporter_to_simulation(
                simulation=simulation, filename=reporter_filename, 
                toy_mode=self.model.using_toy())
        #lock_filename = os.path.join(
        #    self.running_directory, "LOCK")
        lock_filename = os.path.join(
            self.running_directory, f"LOCK.swarm_{self.swarm_id}")
        steps_between_evals_compl_crit = output_stage.completion_criteria\
            .get_evaluation_interval()
        steps_between_evals_sampling = self.sampling_method_interface\
            .get_evaluation_interval()
        if steps_between_evals_compl_crit  == 0:
            if steps_between_evals_sampling == 0:
                steps_between_evals = 0
            else:
                steps_between_evals = steps_between_evals_sampling
        else:
            if steps_between_evals_sampling == 0:
                steps_between_evals = steps_between_evals_compl_crit
            else:
                assert steps_between_evals_compl_crit == steps_between_evals_sampling, \
                    "The evaluation intervals for the sampling method and "\
                    "the completion criteria do not match. Please make them "\
                    "the same, or set one of them to 0."
                steps_between_evals = steps_between_evals_compl_crit
        start_time = time.time()
        num_steps = self.sampling_method_interface.run(
            steps_between_checkpoints=self.restart_checkpoint_interval,
            steps_between_evals=steps_between_evals,
            start_step=self.start_step, min_steps=self.min_steps,
            max_steps=self.max_steps, restart_index=restart_index,
            swarm_id=self.swarm_id,
            checkpoint_filename=self.restart_checkpoint_filename,
            checkpoint_info_filename=self.checkpoint_information_filename,
            lock_filename=lock_filename,
            completion_openmm_interface=self.completion_criteria_interface,
            do_initial_benchmark=True,
            needs_cv_values=self.needs_cv_values,
            run_minimization=self.run_minimization)
        total_time = time.time() - start_time
        ns_per_day = num_steps * self.sim_openmm.timestep * 1e-3 \
            * 86400 / total_time
        logger.info(f"Final benchmark (ns/day): {ns_per_day}")
        # Write final PDB file
        if self.output_positions_basename is not None:
            final_positions_filename = os.path.join(
                self.running_directory, self.output_positions_basename)
            #if self.model.using_toy():
            #    sampling_openmm_interface.write_txt_file_from_simulation(
            #        final_positions_filename, simulation)
            #else:
            #    sampling_openmm_interface.write_pdb_file_from_simulation(
            #        final_positions_filename, simulation)
            sampling_openmm_interface.write_state_file_from_simulation(
                final_positions_filename, simulation)
            
        if self.benchmark:
            # Delete all outputs if this was benchmark mode.
            if self.anchor_index is None:
                anchor = None
            else:
                anchor = self.model.anchors[self.anchor_index]
            if anchor is None:
                md_engine = self.model.get_engine_by_scale_type("molecular_dynamics")
                preparer.cleanse_outputs_stage(
                    self.model, md_engine.unpartitioned_settings, 
                    self.stage_output_index, self.swarm_id)
            else:
                preparer.cleanse_outputs_stage(
                    self.model, anchor, self.stage_output_index, self.swarm_id)
        
        self.sampling_method_interface.post_simulation_procedures(swarm_id=self.swarm_id)
        return
    
def run_simulation(
        model: structures.Seekr_model, 
        stage_index: int, 
        anchor_index: int, 
        restart: bool, 
        swarm_id: int, 
        device_index: str | None = None,
        force_overwrite: bool = False,
        benchmark: bool = False
        ) -> None:
    """
    Run an OpenMM simulation.
    """
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
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

    runner = Runner_openmm(model, stage_index, anchor_index, benchmark=benchmark)
    restart_index = runner.prepare(restart, force_overwrite, swarm_id)
    #assert runner.md_parameters_topology is not None
    assert runner.sampling_method_interface is not None
    assert runner.completion_criteria_interface is not None
    sim_openmm_obj = None
    if sim_openmm_obj is None:
        sim_openmm_obj = sim_openmm.create_sim_openmm(
            model=model, 
            anchor_index=anchor_index,
            positions_directory=runner.input_directory, 
            positions_basename=runner.input_positions, 
            md_parameters_directory=runner.md_parameters_directory,
            md_parameters_topology=runner.md_parameters_topology, 
            running_directory=runner.running_directory,
            ensemble=runner.get_ensemble(),
            sampling_method_interface=runner.sampling_method_interface,
            completion_criteria_interface=runner.completion_criteria_interface, 
            restraints_interfaces=runner.restraints_interfaces,
            swarm_id=swarm_id,
            restart=restart,
            timestep=runner.timestep
            )
            
    runner.run(sim_openmm_obj, restart, restart_index)
    return
