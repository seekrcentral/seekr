"""
This logistic stage converts a swarm of ending structures to a 
position reporter.
"""

import os
import glob
import json
import logging
import typing
from shutil import rmtree

from attrs import define, field, validators, Factory
import numpy as np
import mdtraj

import openmm
from openmm import unit
from openmm import app as openmm_app

import seekr.modules.base as base
import seekr.modules.scales.base as scales_base
import seekr.modules.scales.logistic.base as logistic_scale_base

@define
class Stage_swarm_structure_to_reporter(
        logistic_scale_base.Stage_logistic):
    """
    This stage takes the ending structures of the swarms of the 
    previous stage and converts them to a position reporter.
    """
    type: typing.Literal["swarm_structure_to_reporter_stage"] \
        = "swarm_structure_to_reporter_stage"
    scale_type: typing.Literal["logistic"] = "logistic"
    index: int = field(default=0, 
                       validator=[validators.instance_of(int),
                                  validators.ge(0)])
    name: str = field(default="swarm_structure_to_reporter",
                      validator=validators.instance_of(str))
    comment: str = field(default="This is a logistic stage.",
                         validator=validators.instance_of(str))
    output_positions_basename_noext: str = field(
        default="output_positions",
        validator=validators.instance_of(str))
    output_positions_glob_list: list[str] = field(
        default=Factory(lambda: ["output_positions*.pdb",
                                 "output_positions*.dcd",
                                 "output_positions*.xtc"]),
        validator=validators.instance_of(list))
    input_stage_index: int = field(
        default=0,
        validator=[validators.instance_of(int),
                   validators.ge(0)])
    set_name: typing.Optional[str] = field(
        default="main",
        validator=validators.optional(validators.instance_of(str)))
    extension: str = field(default="dcd",
                           validator=validators.in_(["dcd", "pdb", "xtc"]))
    scope: str = field(default="unpartitioned",
                       validator=validators.in_(
                           ["unpartitioned", "partitioned"]))
    
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
    
    def add_stage_parameters_to_unpartitioned_settings(
            self, 
            model: "Seekr_model"
            ) -> None:
        """
        Add the stage parameters to the unpartitioned settings.
        """
        directory_name: str = self.get_directory_name()
        stage_parameter: dict[str, int | str] = {}
        stage_parameter["index"] = self.index
        stage_parameter["directory"] = directory_name
        stage_parameter["positions"] \
            = self.output_positions_basename_noext
        
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.unpartitioned_settings is not None
        md_engine.unpartitioned_settings.stage_parameters[self.name] \
            = stage_parameter
        return
    
    def has_files(
            self, 
            model: "Seekr_model",
            anchor_index: int | None = None,
            ) -> bool:
        """
        Returns true if any of the anchor directories contain
        outputs from the stage.
        """
        
        stage_output_globs: list[str] = self.output_positions_glob_list
        my_directory = self.get_directory_name()
        anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", self.get_output_set_name())

        if anchor_index is None:
            stage_directory = self.get_directory_name()
            for stage_output_glob in stage_output_globs:
                output_files_glob = os.path.join(
                    model.directory, base.UNPARTITIONED_DIRECTORY, 
                    stage_directory, stage_output_glob)
                if len(glob.glob(output_files_glob)) > 0:
                    return True
        else: 
            for anchor in model.anchors:
                if anchor_index is not None \
                        and anchor.index != anchor_index:
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
        Remove entire directory in all anchors and unpartitioned 
        settings for this stage.
        """
        my_directory = self.get_directory_name()
        anchor_set_directory = os.path.join(
            model.directory, "anchor_sets", self.get_output_set_name())
        for anchor in model.anchors:
            full_directory = os.path.join(
                anchor_set_directory, anchor.directory, my_directory)
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
        Load the CVs, selections, and anchor set for this stage 
        into the model.
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
            instructions: list[logistic_scale_base\
                               .Run_instruction_logistic]
        ) -> None:
        """
        Write the given instructions to every anchor's log file 
        within this stage's directory.
        """
        for instruction in instructions:
            log_filename = model.get_synopsis_filename()
            base.configure_logging(
                log_filename, overwrite=False)
            logging.info(instruction.message)

@define
class Input_stage_swarm_structure_to_reporter(
        logistic_scale_base.Input_stage_logistic):
    """
    This input stage takes the ending structures of the swarms of the 
    previous stage and converts them to a position reporter.
    """
    type: typing.Literal["swarm_structure_to_reporter_stage"] \
        = "swarm_structure_to_reporter_stage"
    name: str = field(default="swarm_structure_to_reporter",
                      validator=validators.instance_of(str))
    comment: str = field(default="Assign swarm ending structures to a position reporter.",
                         validator=validators.instance_of(str))
    input_stage_name: str | None = field(default=None,
                                         validator=validators.optional(
                                            validators.instance_of(str)))
    set_name: typing.Optional[str] = field(
        default="main",
        validator=validators.optional(validators.instance_of(str)))
    extension: str = field(default="dcd",
                           validator=validators.in_(["dcd", "pdb", "xtc"]))

    def get_scale_type(self) -> str:
        """
        Get the scale type for this input stage.
        """
        return "logistic"
    
    def make_model_stage(
            self, 
            index: int,
            scope: str,
            dummy_settings: typing.Any = None,
            dummy_directory: str = ""
            ) -> typing.Tuple[Stage_swarm_structure_to_reporter, str]:
        """
        Given this input stage object, create the model's Stage object.
        """
        stage = Stage_swarm_structure_to_reporter()
        stage.index = index
        stage.name = self.name
        stage.comment = self.comment
        stage.set_name = self.set_name
        stage.scope = scope
        stage.extension = self.extension
        stage.output_positions_glob_list = [
            f"{stage.output_positions_basename_noext}*.{stage.extension}"]
        return stage, scope

def is_finished_swarm_structure_to_reporter_logistic_with_message(
        model: "Seekr_model",
        stage: Stage_swarm_structure_to_reporter,
        anchor_or_unpartitioned: "structures.Anchor | None",
        use_json: bool = True
    ) -> tuple[bool, str | dict]:
    """
    A swarm structures from reporter logistic stage is finished when all 
    of the ending structures for the swarm have been written to the reporter.
    """
    if use_json:
        if stage.scope == "partitioned":
            assert anchor_or_unpartitioned is not None
            has_files = stage.has_files(
                model, anchor_index=anchor_or_unpartitioned.index)
            progress = 1.0 if has_files else 0.0
            message = {
                "finished": has_files,
                "progress": progress,
                "num_swarms": 1,
                "swarms": {"0":{
                    "finished": has_files,
                    "progress": progress,
                }}
            }
        else:
            has_files = stage.has_files(model, anchor_index=None)
            progress = 1.0 if has_files else 0.0
            message = {
                "finished": has_files,
                "progress": progress,
                "num_swarms": 1,
                "swarms": {"0":{
                    "finished": has_files,
                    "progress": progress,
                }}
            }
        return has_files, message
    else:
        if stage.scope == "partitioned":
            assert anchor_or_unpartitioned is not None
            has_files = stage.has_files(
                model, anchor_index=anchor_or_unpartitioned.index)
            if has_files:
                message = f"    Anchor {anchor_or_unpartitioned.index} "\
                           "has ending reporter.\n"
            else:
                message = f"    Anchor {anchor_or_unpartitioned.index} "\
                           "has no ending reporter.\n"
        else:
            has_files = stage.has_files(model, anchor_index=None)
            if has_files:
                message = f"    Reporter generated.\n"
            else:
                message = f"    Reporter not generated.\n"
        return has_files, message

def choose_next_simulation_swarm_structure_to_reporter_logistic(
        model: "Seekr_model",
        stage: Stage_swarm_structure_to_reporter,
        force_overwrite: bool,
        use_json: bool = True,
        silent: bool = False
        ) -> list:
    """
    Choose the next process to run for a swarm structures 
    to reporter logistic stage.
    """
    run_instructions: list[typing.NamedTuple] = []
    all_completed = True
    message: str | dict = ""
    anchor_or_unpartitioned_index_list = scales_base\
        .construct_anchor_or_unpartitioned_index_list(model, stage, "any")
    for anchor_or_unpartitioned_index in anchor_or_unpartitioned_index_list:
        if anchor_or_unpartitioned_index is None:
            anchor_or_unpartitioned = None
        else:
            anchor_or_unpartitioned = model.anchors[anchor_or_unpartitioned_index]
            if anchor_or_unpartitioned.bulk_anchor:
                continue
        completed, message \
            = is_finished_swarm_structure_to_reporter_logistic_with_message(
                model, stage, anchor_or_unpartitioned, use_json)
        if not completed:
            all_completed = False
            
    if not all_completed or force_overwrite:
        run_instruction = logistic_scale_base.Run_instruction_logistic(
            anchor_index=None,
            swarm_id=0,
            message=message
        )
        run_instructions.append(run_instruction)
    return run_instructions

def _get_topology_file(
        model: "Seekr_model",
        md_engine: typing.Any,
        input_stage: typing.Any,
        running_dir: str
        ) -> str:
    """
    Get the path to a topology/structure file for loading DCD 
    trajectories, mirroring the logic in 
    modules/engines/openmm/runner.py. For toy systems, returns 
    toy.pdb from the input stage's running directory. For real 
    systems, returns the input positions PDB used by that stage.
    """
    if model.using_toy():
        toy_pdb = os.path.join(running_dir, "toy.pdb")
        assert os.path.exists(toy_pdb), \
            f"toy.pdb not found in the input stage directory: {toy_pdb}"
        return toy_pdb
    else:
        if input_stage.scope == "unpartitioned":
            # Get topology from the initial stage
            input_positions_filenames = md_engine.unpartitioned_settings\
                .get_positions_full_paths(
                    #model.directory, input_stage.input_stage_index, 
                    #input_stage.set_name, swarm_id=0)
                    model.directory, 0, 
                    "main", swarm_id=0)
        else:
            for anchor in model.anchors:
                if anchor.bulk_anchor:
                    continue
                stage_dir_name = input_stage.get_directory_name()
                anchor_set_directory = os.path.join(
                    model.directory, "anchor_sets", 
                    input_stage.get_output_set_name())
                full_stage_dir = os.path.join(
                    anchor_set_directory, stage_dir_name)
                basename_glob = input_stage.sampling_method\
                    .get_output_positions_basename_noext()+"*.pdb"
                input_positions_glob = os.path.join(
                    full_stage_dir, basename_glob)
                input_positions_filenames = glob.glob(
                    input_positions_glob)
        
        assert len(input_positions_filenames) > 0, \
            "No input positions/topology file found for DCD loading."
        return input_positions_filenames[0]

def _box_vectors_nm(box_vectors) -> np.ndarray | None:
    """
    Return periodic box vectors as a (3, 3) array in nm, or None.
    """
    if box_vectors is None:
        return None
    if hasattr(box_vectors, "value_in_unit"):
        arr = np.asarray(
            box_vectors.value_in_unit(unit.nanometer), dtype=float)
    else:
        rows = []
        for vector in box_vectors:
            if hasattr(vector, "value_in_unit"):
                rows.append(np.asarray(
                    vector.value_in_unit(unit.nanometer), dtype=float))
            else:
                rows.append(np.asarray(vector, dtype=float).reshape(3))
        arr = np.asarray(rows, dtype=float)
    if arr.shape != (3, 3):
        arr = np.reshape(arr, (3, 3))
    if not np.all(np.linalg.norm(arr, axis=1) > 0.0):
        return None
    return arr

def load_positions_nm(
        path: str
        ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return xyz (n_atoms, 3) in nm, and box vectors (3, 3) in nm or None.
    """
    ext = os.path.splitext(path)[1]
    if ext == ".txt":
        xyz = np.loadtxt(path, ndmin=2)
        return np.asarray(xyz, dtype=float), None
    if ext == ".pdb":
        pdb = openmm_app.PDBFile(path)
        xyz = np.asarray(
            pdb.positions.value_in_unit(unit.nanometer), dtype=float)
        return xyz, _box_vectors_nm(pdb.topology.getPeriodicBoxVectors())
    if ext == ".state":
        with open(path, "r") as f:
            state = openmm.XmlSerializer.deserialize(f.read())
        xyz = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=float)
        return xyz, _box_vectors_nm(state.getPeriodicBoxVectors())
    raise Exception(
        f"Positions file must be .txt, .pdb, or .state: {path}")

def do_run_instruction_swarm_structure_to_reporter_logistic(
        model: "Seekr_model",
        force_overwrite: bool,
        stage_index: int,
        instruction: list
        ) -> None:
    """
    Read swarm structure files from the predecessor MD stage,
    write the reporter from these.
    """
    stage: Stage_swarm_structure_to_reporter \
        = model.stages[stage_index]
    input_stage = model.stages[stage.input_stage_index-1]
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    assert md_engine is not None, "No MD engine found in model."
    if stage.scope == "unpartitioned":
        assert md_engine.unpartitioned_settings is not None, \
            "No unpartitioned settings found in MD engine."
        unpartitioned_settings = md_engine.unpartitioned_settings
        running_dir = os.path.join(
            model.directory, unpartitioned_settings.directory,
            input_stage.get_directory_name())
    else:
        raise NotImplementedError(
            "swarm_structures_from_reporter_stage with " \
            "partitioned scope is not yet implemented.")
    swarm_ids = list(dict.fromkeys(
        unpartitioned_settings.get_starting_structure_swarm_ids(
            model.directory, input_stage.index, stage.set_name)))
    assert swarm_ids, f"No ending structures for stage {input_stage.name}."
    paths = []
    for swarm_id in swarm_ids:
        matches = unpartitioned_settings.get_positions_full_paths(
            model.directory, input_stage.index, stage.set_name, swarm_id)
        state_matches = [match for match in matches if match.endswith(".state")]
        chosen = state_matches or matches
        assert len(chosen) == 1, (
            f"Expected one ending structure for swarm {swarm_id}, found {matches}.")
        paths.append(chosen[0])
    
    loaded = [load_positions_nm(path) for path in paths]
    xyz = np.stack([positions for positions, _ in loaded])
    boxes = [box for _, box in loaded]
    topology_file = _get_topology_file(model, md_engine, input_stage, running_dir)
    topology = mdtraj.load(topology_file).topology
    traj = mdtraj.Trajectory(xyz, topology)
    if all(b is not None for b in boxes):
        traj.unitcell_vectors = np.stack(boxes)

    out_dir = os.path.join(
        model.directory, unpartitioned_settings.directory,
        stage.get_directory_name())
    os.makedirs(out_dir, exist_ok=True)
    traj.save(os.path.join(
        out_dir, stage.output_positions_basename_noext + "." + stage.extension))
    return
