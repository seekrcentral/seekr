from __future__ import annotations
"""
This logistic stage will take the outputs to a reporter and 
will assign them as the starting structures for anchors within a
future stage.

This involves changing the scope of a future stage to be a 
'partitioned' scope from a 'unpartitioned' scope.
"""

import os
import glob
import json
import logging
import typing
from shutil import rmtree

import numpy as np
import mdtraj
from attrs import define, field, validators, Factory

import seekr.modules.base as base
import seekr.modules.scales.logistic.base as logistic_scale_base

logger = logging.getLogger(__name__)

OUTPUT_POSITIONS_BASENAME_NOEXT = "position"

@define
class Stage_anchor_starting_structures_from_reporter(
        logistic_scale_base.Stage_logistic):
    """
    This stage takes the output of a reporter within an input stage and
    assigns starting structures to the anchors of a later stage, changing
    the scope of the scale to 'partitioned'.

    If there's more than one reporter in the stage, assume the first one.
    """
    type: typing.Literal["anchor_starting_structures_from_reporter_stage"] \
        = "anchor_starting_structures_from_reporter_stage"
    scale_type: typing.Literal["logistic"] = "logistic"
    index: int = field(default=0,
                        validator=[validators.instance_of(int),
                                      validators.ge(0)])
    name: str = field(default="anchor_starting_structures_from_reporter", 
                      validator=validators.instance_of(str))
    comment: str = field(default="This is a logistic stage.",
                         validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions", 
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["output_positions*.pdb", 
                                 "output_positions*.txt"]), 
        validator=validators.instance_of(list))
    input_stage_index: int = field(
        default=0,
        validator=[validators.instance_of(int),
                    validators.ge(0)])
    set_name: typing.Optional[str] = field(
        default="main", 
        validator=validators.optional(validators.instance_of(str)))
    reporter_name: str = field(
        default="",
        validator=validators.instance_of(str))
    scope: typing.Literal["partitioned"] = "partitioned"
    
    def get_directory_name(self) -> str:
        """
        Get the directory of this stage.
        """
        directory_name: str = f"stage_{self.index}_{self.name}"
        return directory_name

    def add_stage_parameters_to_anchors(
            self, 
            model: "Seekr_model",
            ) -> None:
        """
        Add the stage directory to all anchors in the model.
        """
        for anchor in model.anchors:
            directory_name: str = self.get_directory_name()
            stage_parameter: dict[str, int | str] = {}
            stage_parameter["index"] = self.index
            stage_parameter["directory"] = directory_name
            stage_parameter["positions"] \
                = self.output_positions_basename_noext
            anchor.stage_parameters[self.name] = stage_parameter
        return
    
    def has_files(
            self, 
            model: "Seekr_model",
            anchor_index: int | None = None,
            ) -> bool:
        """
        Returns true if any of the stage directories contain
        outputs from any of the stages.
        """
        
        stage_output_globs: list[str] = self.output_positions_glob_list
        my_directory = self.get_directory_name()
        anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", self.get_output_set_name())
        for anchor in model.anchors:
            if anchor_index is not None and anchor.index != anchor_index:
                continue
            for stage_parameter in anchor.stage_parameters.values():
                stage_directory = stage_parameter["directory"]
                if stage_directory == my_directory:
                    for stage_output_glob in stage_output_globs:
                        output_files_glob = os.path.join(
                            anchor_set_directory, anchor.directory, 
                            stage_directory, stage_output_glob)
                        if len(glob.glob(output_files_glob)) > 0:
                            return True
        return False
    
    def remove_directory(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Remove entire directory in all anchors and unpartitioned settings for this stage.
        """
        my_directory = self.get_directory_name()
        anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", self.get_output_set_name())
        for anchor in model.anchors:
            full_directory = os.path.join(anchor_set_directory, anchor.directory, my_directory)
            if os.path.exists(full_directory):
                rmtree(full_directory)
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.unpartitioned_settings is not None
        full_directory = os.path.join(
            model.directory, md_engine.unpartitioned_settings.directory, my_directory)
        if os.path.exists(full_directory):
            rmtree(full_directory)
        return
    
    def is_logistic_stage(self) -> bool:
        """
        Returns true since this is a logistic stage.
        """
        return True
    
    def load_sets_into_model(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Load the CVs, selections, and anchor set for this stage into the model.
        """
        model.load_cvs(self.set_name)
        model.load_selections(self.set_name)
        model.load_anchor_set(self.set_name)
        return
    
    def get_output_set_name(self) -> str | None:
        """
        Get the output set name for this stage.
        """
        return self.set_name
    
    def write_to_log(
            self,
            model: "Seekr_model",
            instructions: list[logistic_scale_base.Run_instruction_logistic]
        ) -> None:
        """
        Write the given instructions to every anchor's log file within this 
        stage's directory.
        """
        for instruction in instructions:
            log_filename = model.get_synopsis_filename()
            base.configure_logging(log_filename)
            logger.info(instruction.message)
    
@define
class Input_stage_anchor_starting_structures_from_reporter(logistic_scale_base.Input_stage_logistic):
    """
    This stage assigns the starting structures of the anchors of the model
    according to the output of a reporter stage.
    """
    type: typing.Literal["anchor_starting_structures_from_reporter_stage"] \
        = "anchor_starting_structures_from_reporter_stage"
    name: str = field(default="anchor_starting_structures_from_reporter", 
                      validator=validators.instance_of(str))
    comment: str = field(default="Assign anchor starting structures from a reporter.",
                         validator=validators.instance_of(str))
    # If these two are None, then assume previous/next stage.
    input_stage_name: str | None = field(default=None,
                                 validator=validators.optional(
                                     validators.instance_of(str)))
    set_name: typing.Optional[str] = field(
        default="main", 
        validator=validators.optional(validators.instance_of(str)))
    reporter_name: str = field(
        default="",
        validator=validators.instance_of(str))

    def get_scale_type(self) -> str:
        return "logistic"

    def make_model_stage(
            self, 
            index: int,
            dummy_scope: str,
            dummy_settings: typing.Any = None,
            dummy_running_dir: str = "",
            ) -> typing.Tuple[Stage_anchor_starting_structures_from_reporter, str]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = Stage_anchor_starting_structures_from_reporter()
        stage.index = index
        stage.name = self.name
        stage.comment = self.comment
        stage.set_name = self.set_name
        stage.reporter_name = self.reporter_name
        new_scope = "partitioned"
        return stage, new_scope

def _get_topology_file(
        model: "Seekr_model",
        md_engine: typing.Any,
        input_stage: typing.Any,
        running_dir: str
        ) -> str:
    """
    Get the path to a topology/structure file for loading DCD trajectories,
    mirroring the logic in modules/engines/openmm/runner.py.
    For toy systems, returns toy.pdb from the input stage's running directory.
    For real systems, returns the input positions PDB used by that stage.
    """
    if model.using_toy():
        toy_pdb = os.path.join(running_dir, "toy.pdb")
        assert os.path.exists(toy_pdb), \
            f"toy.pdb not found in the input stage directory: {toy_pdb}"
        return toy_pdb
    else:
        input_positions_filenames = md_engine.unpartitioned_settings\
            .get_positions_full_paths(
                #model.directory, input_stage.input_stage_index, 
                #input_stage.set_name, swarm_id=0)
                model.directory, 0, "main", swarm_id=0)
                
        assert len(input_positions_filenames) > 0, \
            "No input positions/topology file found for DCD loading."
        return input_positions_filenames[0]


def _write_frame(
        traj: typing.Any,
        frame_num: int,
        output_path: str,
        using_toy: bool
        ) -> None:
    """
    Write a single frame from an mdtraj Trajectory to disk.
    Creates the parent directory if it does not already exist.
    Toy systems produce .txt (NumPy savetxt); real systems produce .pdb.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if using_toy:
        np.savetxt(output_path, traj.xyz[frame_num])
    else:
        traj[frame_num].save_pdb(output_path)
    return


def _process_anchor_reporter(
        model: "Seekr_model",
        stage: Stage_anchor_starting_structures_from_reporter,
        traj_json_pairs: list,
        reporter: typing.Any
        ) -> None:
    """
    For an Anchor_reporter output, extract one frame per anchor and write
    starting structures into each anchor's logistic-stage directory.
    Iterates all restart files oldest→newest. For 'save_last_frame' (default)
    later files overwrite earlier ones; for 'save_first_frame' the first
    encounter per anchor wins.
    """
    # anchor_index → (traj, frame_num_within_that_traj)
    anchor_to_traj_frame: dict[int, tuple] = {}
    for traj, json_data in traj_json_pairs:
        file_filled_indices = sorted(
            v["index"] for v in json_data.values() if v.get("filled", False)
        )
        for fn, anchor_idx in enumerate(file_filled_indices):
            if fn >= len(traj):
                break
            if reporter.mode == "save_first_frame" and anchor_idx in anchor_to_traj_frame:
                continue
            # Defense in depth: reject frames whose coordinates are not actually
            # inside the claimed Voronoi cell (e.g. stale-CV mislabels from
            # CustomCVForce / Metadynamics reporters).
            cv_values = _compute_cv_values_from_frame(model, traj, fn)
            if cv_values is not None:
                validated_idx = _find_anchor_index_from_cv_values(model, cv_values)
                if validated_idx != anchor_idx:
                    logger.warning(
                        f"Skipping reporter frame {fn} labeled as anchor "
                        f"{anchor_idx}: coordinates map to anchor "
                        f"{validated_idx} (CV={cv_values}).")
                    continue
            anchor_to_traj_frame[anchor_idx] = (traj, fn)
    output_ext = ".txt" if model.using_toy() else ".pdb"
    stage_dir_name = stage.get_directory_name()
    anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", stage.get_output_set_name())
    for anchor in model.anchors:
        if anchor.index not in anchor_to_traj_frame:
            logger.warning(
                f"Anchor {anchor.index} was not visited by the Anchor_reporter. "
                "No starting structure assigned.")
            continue
        traj, frame_num = anchor_to_traj_frame[anchor.index]
        output_path = os.path.join(
            anchor_set_directory, anchor.directory, stage_dir_name,
            stage.output_positions_basename_noext + ".swarm_0" + output_ext)
        _write_frame(traj, frame_num, output_path, model.using_toy())
        logger.info(
            f"Assigned starting structure for anchor {anchor.index}: {output_path}")
    return


def _compute_cv_values_from_frame(
        model: "Seekr_model",
        traj: typing.Any,
        frame_index: int
        ) -> list[float] | None:
    """
    Compute CV values for a single trajectory frame using each CV's
    get_mdtraj_cv_value() method. Returns None if any CV lacks that method.
    Uses os.chdir(model.directory) around each call, matching the pattern
    in find_current_anchor_index (needed for RMSD reference paths).
    """
    cv_values: list[float] = [0.0] * len(model.collective_variables)
    curdir = os.getcwd()
    try:
        os.chdir(model.directory)
        for cv in model.collective_variables:
            if not hasattr(cv, "get_mdtraj_cv_value"):
                logger.warning(
                    f"CV '{cv.name}' does not support get_mdtraj_cv_value(); "
                    "cannot validate frame against anchor milestones.")
                return None
            cv_values[cv.index] = cv.get_mdtraj_cv_value(model, traj, frame_index)
    finally:
        os.chdir(curdir)
    return cv_values


def _find_anchor_index_from_cv_values(
        model: "Seekr_model",
        cv_values: list[float]
        ) -> int | None:
    """
    Find which anchor a set of CV values belongs to by checking all
    milestone boundaries, mirroring Base_md_openmm_interface.find_current_anchor_index.
    """
    curdir = os.getcwd()
    try:
        os.chdir(model.directory)
        for anchor in model.anchors:
            in_anchor = True
            for milestone in anchor.milestones:
                if not milestone.is_point_within(cv_values):
                    in_anchor = False
                    break
            if in_anchor:
                return anchor.index
    finally:
        os.chdir(curdir)
    return None


def _process_cv_value_reporter(
        model: "Seekr_model",
        stage: Stage_anchor_starting_structures_from_reporter,
        traj_json_pairs: list,
        reporter: typing.Any
        ) -> None:
    """
    For a CV_value_reporter output, assign each anchor's starting structure
    by iterating over every saved frame across all restart files, computing
    CV values from coordinates, and checking milestone boundaries to determine
    which anchor each frame belongs to.
    Iterates oldest→newest; for 'save_last_frame' later files overwrite earlier
    ones; for 'save_first_frame' the first match per anchor is kept.
    """
    # anchor_index → (traj, frame_num_within_that_traj)
    anchor_to_traj_frame: dict[int, tuple] = {}
    for traj, json_data in traj_json_pairs:
        saved_frame_indices = [
            v["frame_index"]
            for v in json_data.values()
            if v.get("frame_index", -1) >= 0
        ]
        for frame_index in saved_frame_indices:
            if frame_index >= len(traj):
                continue
            cv_values = _compute_cv_values_from_frame(model, traj, frame_index)
            if cv_values is None:
                continue
            anchor_index = _find_anchor_index_from_cv_values(model, cv_values)
            if anchor_index is None:
                continue
            if reporter.mode == "save_first_frame" and anchor_index in anchor_to_traj_frame:
                continue
            anchor_to_traj_frame[anchor_index] = (traj, frame_index)
    if not anchor_to_traj_frame:
        logger.warning("CV_value_reporter: no frames could be assigned to any anchor.")
        return
    output_ext = ".txt" if model.using_toy() else ".pdb"
    stage_dir_name = stage.get_directory_name()
    anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", stage.get_output_set_name())
    for anchor in model.anchors:
        if anchor.index not in anchor_to_traj_frame:
            logger.warning(
                f"Anchor {anchor.index} has no frame validated by milestone geometry. "
                "No starting structure assigned.")
            continue
        traj, frame_num = anchor_to_traj_frame[anchor.index]
        output_path = os.path.join(
            anchor_set_directory, anchor.directory, stage_dir_name,
            stage.output_positions_basename_noext + ".swarm_0" + output_ext)
        _write_frame(traj, frame_num, output_path, model.using_toy())
        logger.info(
            f"Assigned starting structure for anchor {anchor.index}: {output_path}")
    return

def is_finished_anchor_starting_structures_from_reporter_logistic_with_message(
        model: "Seekr_model",
        stage: Stage_anchor_starting_structures_from_reporter,
        use_json: bool = True
    ) -> tuple[bool, str | dict]:
    """
    A reassign anchors logistic stage is finished when all of the new anchor
    directories have files in them.
    """
    all_have_files = True
    if use_json:
        final_message: dict[int, dict] = {}
        for anchor in model.anchors:
            if "molecular_dynamics" not in anchor.scale_types or anchor.bulk_anchor:
                continue
            has_files = stage.has_files(model, anchor_index=anchor.index)
            if not has_files:
                all_have_files = False
            final_message[anchor.index] = {
                "anchor_index": anchor.index,
                "has_files": has_files,
            }
        return all_have_files, final_message
    else:
        message = ""
        for anchor in model.anchors:
            if "molecular_dynamics" not in anchor.scale_types or anchor.bulk_anchor:
                continue
            if not stage.has_files(model, anchor_index=anchor.index):
                all_have_files = False
                message += f"    Anchor {anchor.index} has no files.\n"
        if all_have_files:
            message = "    All anchors have files."
        return all_have_files, message

def choose_next_simulation_anchor_starting_structures_from_reporter_logistic(
        model: "Seekr_model",
        stage: Stage_anchor_starting_structures_from_reporter,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    """
    Choose the next simulation to run for a reassign anchors logistic stage.
    """
    run_instructions: list[typing.NamedTuple] = []
    completed, message \
        = is_finished_anchor_starting_structures_from_reporter_logistic_with_message(
            model, stage, use_json)
    if not completed or force_overwrite:
        run_instruction = logistic_scale_base.Run_instruction_logistic(
            anchor_index=None,
            swarm_id=0,
            message=message
        )
        run_instructions.append(run_instruction)
    return run_instructions

def do_run_instruction_anchor_starting_structures_from_reporter_logistic(
        model: "Seekr_model",
        force_overwrite: bool,
        stage_index: int,
        instruction: list
        ) -> None:
    """
    Read reporter output files from the predecessor unpartitioned MD stage,
    extract one representative frame per anchor, and write starting structures
    for the next partitioned stage.
    """
    stage: Stage_anchor_starting_structures_from_reporter = model.stages[stage_index]
    input_stage = model.stages[stage.input_stage_index-1]
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    assert md_engine is not None, "No MD engine found in model."
    assert md_engine.unpartitioned_settings is not None, \
        "No unpartitioned settings found in MD engine."
    unpartitioned_settings = md_engine.unpartitioned_settings
    running_dir = os.path.join(
        model.directory, unpartitioned_settings.directory,
        input_stage.get_directory_name())
    basename = input_stage.sampling_method.get_output_positions_basename_noext()
    swarm_string = ".swarm_0"
    assert stage.reporter_name != "", \
        "reporter_name must be set on the logistic stage."
    reporter = None
    for r in input_stage.reporters:
        if r.name == stage.reporter_name:
            reporter = r
            break
    assert reporter is not None, \
        f"Reporter '{stage.reporter_name}' not found in input stage "\
        f"'{input_stage.name}'. Check that reporter_name matches the reporter's name."
    ext = reporter.get_reporter_extension()
    reporter_glob = os.path.join(
        running_dir, f"{basename}*{swarm_string}.{reporter.name}.{ext}")
    reporter_files = sorted(glob.glob(reporter_glob))
    assert len(reporter_files) > 0, \
        f"No reporter output files found matching: {reporter_glob}"
    topology_file = _get_topology_file(model, md_engine, input_stage, running_dir)
    # Build (traj, json_data) pairs for every restart file that has a JSON sidecar.
    traj_json_pairs = []
    for reporter_file in reporter_files:
        json_sidecar = os.path.splitext(reporter_file)[0] + ".json"
        if not os.path.exists(json_sidecar):
            continue
        with open(json_sidecar, "r") as f:
            json_data = json.load(f)
        if ext == "pdb":
            traj = mdtraj.load(reporter_file)
        elif ext == "dcd":
            traj = mdtraj.load(reporter_file, top=topology_file)
        else:
            raise Exception(f"Unsupported reporter file extension: {ext}")
        traj_json_pairs.append((traj, json_data))
    assert len(traj_json_pairs) > 0, \
        f"No reporter files with JSON sidecars found matching: {reporter_glob}"
    if reporter.type == "anchor":
        _process_anchor_reporter(model, stage, traj_json_pairs, reporter)
    elif reporter.type == "cv_value":
        _process_cv_value_reporter(model, stage, traj_json_pairs, reporter)
    else:
        raise Exception(
            f"Unsupported reporter type '{reporter.type}' for "
            "anchor_starting_structures_from_reporter_logistic stage.")
    return
