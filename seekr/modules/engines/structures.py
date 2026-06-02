from __future__ import annotations
"""
modules/engines/structures.py

Import base objects from all implemented engines.
"""

import os
import re
import json
import glob
import typing
import logging
from shutil import copyfile

from attrs import define, field, validators, Factory
import cattrs

import seekr.modules.base as base

# NOTE: this one seems only to be used by modules/structures.py and
# some engines structures.py modules, so it seems best placed here for
# now, even though it's not exclusive to engines.
@define
class Selection:
    """
    A selection of atoms to help with the definition of CVs.
    """
    name: str = field(default="",
                        validator=validators.instance_of(str))
    atom_indices: list[int] = field(
        default=Factory(list), validator=validators.deep_iterable(
            member_validator=[
                validators.instance_of(int),
                validators.ge(0)],
                iterable_validator=validators.instance_of(list),))
    
    def save(
            self, 
            directory: str,
            scale_name: str
            ) -> None:
        """
        Save the selection to a file.
        """
        converter: cattrs.Converter = cattrs.Converter()
        cvs_dict: dict = converter.unstructure(self)
        cvs_dict_noindent: dict = base.wrap_noindent_integers(cvs_dict)
        json_dump: str = json.dumps(cvs_dict_noindent, indent=4, 
                                    cls=base.NoIndentListEncoder)
        
        filename = os.path.join(directory, self.name +  "_" + scale_name + ".json")
        with open(filename, "w") as file:
            file.write(json_dump)
        return
    
    def load(
            self,
            filename: str
            ) -> None:
        """
        Load the selection from a file.
        """
        with open(filename, "r") as file:
            selection_dict = json.load(file)
            self.name = selection_dict["name"]
            self.atom_indices = selection_dict["atom_indices"]
        return

    
@define
class Ion:
    """
    Ion objects used for Browndye and SDA engines.
    """
    name: str | None = field(
        default=None, validator=validators.optional(
            validators.instance_of(str)))
    radius: float = field(
        default=0.9, validator=[validators.instance_of(float), 
                                validators.gt(0)])
    charge: float = field(
        default=1.0, validator=[validators.instance_of(float)])
    concentration: float = field(
        default=0.0, validator=[validators.instance_of(float),
                                validators.ge(0)])

@define
class Checkpoint_information:
    """
    A class to store information about the current state of a SEEKR
    calculation.
    """
    current_step: int = field(
        default=0, validator=[validators.instance_of(int),
                              validators.ge(0)])
    restart_index: int = field(
        default=1, validator=[validators.instance_of(int),
                              validators.ge(1)])
    timestep: float = field(
        default=0.002, validator=[validators.instance_of(float),
                                  validators.gt(0)])
    cv_values: tuple[float] | None = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=[validators.instance_of(float)],
                iterable_validator=validators.instance_of(tuple),
            )
        ))
    
    def save(
            self, 
            filename: str
            ) -> None:
        """
        Write the model object to a file.
        """
        checkpoint_information_dict: dict = cattrs.unstructure(self)
        checkpoint_information_dict_noindent: dict \
            = base.wrap_noindent_integers(checkpoint_information_dict)
        json_dump: str = json.dumps(checkpoint_information_dict_noindent,
                                    indent=4, cls=base.NoIndentListEncoder)
        with open(filename, "w") as file:
            file.write(json_dump)

        return

def write_checkpoint_information_file(
        step: int, 
        restart_index: int,
        timestep: float,
        cv_values: tuple[float] | None,
        filename: str) -> None:
    """
    Write the checkpoint information to a file.
    """
    checkpoint_information = Checkpoint_information(
        step, restart_index, timestep, cv_values)
    checkpoint_information.save(filename)
    return

@define
class Langevin_integrator_settings:
    type: typing.Literal["langevin"] = "langevin"
    timestep: float = field(
        default=0.002, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    friction: float = field(
        default=1.0, validator=[validators.instance_of(float),
                                validators.ge(0)])
    target_temperature: float = field(
        default=298.15, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    random_seed: int = field(
        default=0, validator=[validators.instance_of(int)])
    rigid_tolerance: float = field(
        default=1e-6, validator=[validators.instance_of(float),
                                 validators.gt(0)])

@define
class Langevin_middle_integrator_settings:
    type: typing.Literal["langevinMiddle"] = "langevinMiddle"
    timestep: float = field(
        default=0.002, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    friction: float = field(
        default=1.0, validator=[validators.instance_of(float),
                                validators.ge(0)])
    target_temperature: float = field(
        default=298.15, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    random_seed: int = field(
        default=0, validator=[validators.instance_of(int)])
    rigid_tolerance: float = field(
        default=1e-6, validator=[validators.instance_of(float),
                                 validators.gt(0)])

@define
class Brownian_integrator_settings:
    type: typing.Literal["brownian"] = "brownian"
    timestep: float = field(
        default=0.002, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    friction: float = field(
        default=1.0, validator=[validators.instance_of(float),
                                validators.ge(0)])
    target_temperature: float = field(
        default=298.15, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    random_seed: int = field(
        default=0, validator=[validators.instance_of(int)])
    rigid_tolerance: float = field(
        default=1e-6, validator=[validators.instance_of(float),
                                 validators.gt(0)])

@define
class Verlet_integrator_settings:
    type: typing.Literal["verlet"] = "verlet"
    timestep: float = field(
        default=0.002, validator=[validators.instance_of(float),
                                   validators.gt(0)])
    rigid_tolerance: float = field(
        default=1e-6, validator=[validators.instance_of(float),
                                 validators.gt(0)])

Integrator_settings = typing.Union[Langevin_integrator_settings,
                                   Langevin_middle_integrator_settings,
                                   Brownian_integrator_settings,
                                   Verlet_integrator_settings]

@define
class Cuda_platform_settings:
    type: typing.Literal["cuda"] = "cuda"
    device_index: str = field(
        default="0", validator=[validators.instance_of(str)])
    precision: str = field(
        default="mixed", validator=[
            validators.instance_of(str),
            validators.in_(["single", "mixed", "double"])])

# TODO: Add OpenCL platform settings class

Platform_settings = typing.Union[Cuda_platform_settings]

@define
class Toy_settings:
    potential_energy_expression: str = field(
        default="", validator=[validators.instance_of(str)])
    num_particles: int = field(
        default=0, validator=[validators.instance_of(int), validators.ge(0)])
    masses: list[float] = field(default=Factory(list),)
    
@define
class Amber_parameters_topology:
    type: typing.Literal["Amber"] = "Amber"
    prmtop_filename: str = field(
        default="", validator=validators.instance_of(str))

    def same_parameters(
            self, 
            other: "Parameters_topology"
            ) -> bool:
        assert self.type == other.type, "Type mismatch."
        if self.prmtop_filename == other.prmtop_filename:
            return True
        else:
            return False
        
    def copy_files(
            self,
            dest_directory: str,
            new_md_parameters_topology: "Parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.type == new_md_parameters_topology.type, "Type mismatch."
        assert self.prmtop_filename != "", "prmtop_filename must be defined."
        prmtop_basename = os.path.basename(self.prmtop_filename)
        anchor_prmtop_full_path = os.path.join(dest_directory,
                                            prmtop_basename)
        full_prmtop_filename = os.path.expanduser(self.prmtop_filename)
        logging.info(f"Copying Amber prmtop file: {full_prmtop_filename} "
                        f"to anchor directory: {anchor_prmtop_full_path}.")
        copyfile(full_prmtop_filename, anchor_prmtop_full_path)
        new_md_parameters_topology.prmtop_filename = prmtop_basename
        return
    
@define
class Gromacs_parameters_topology:
    type: typing.Literal["Gromacs"] = "Gromacs"
    top_filename: str = field(default="",
                                validator=validators.instance_of(str))
    gro_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(
            self, 
            other: "Parameters_topology"
            ) -> bool:
        assert self.type == other.type, "Type mismatch."
        if (self.top_filename == other.top_filename) \
                and (self.gro_filename == other.gro_filename):
            return True
        else:
            return False
    
    def copy_files(
            self,
            dest_directory: str,
            new_md_parameters_topology: "Parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.type == new_md_parameters_topology.type, "Type mismatch."
        assert self.top_filename != "", "top_filename must be defined."
        assert self.gro_filename != "", "gro_filename must be defined."
        top_basename = os.path.basename(self.top_filename)
        anchor_top_full_path = os.path.join(dest_directory,
                                            top_basename)
        full_top_filename = os.path.expanduser(self.top_filename)
        logging.info(f"Copying Gromacs top file: {full_top_filename} "
                        f"to anchor directory: {anchor_top_full_path}.")
        copyfile(full_top_filename, anchor_top_full_path)
        new_md_parameters_topology.top_filename = top_basename
        gro_basename = os.path.basename(self.gro_filename)
        anchor_gro_full_path = os.path.join(dest_directory,
                                            gro_basename)
        full_gro_filename = os.path.expanduser(self.gro_filename)
        logging.info(f"Copying Gromacs gro file: {full_gro_filename} "
                        f"to anchor directory: {anchor_gro_full_path}.")
        copyfile(full_gro_filename, anchor_gro_full_path)
        new_md_parameters_topology.gro_filename = gro_basename
        return

@define
class Charmm_parameters_topology:
    type: typing.Literal["Charmm"] = "Charmm"
    psf_filename: str = field(default="",
                                validator=validators.instance_of(str))
    param_filename_list: list[str] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(list),
        ))
    
    def same_parameters(
            self, 
            other: "Parameters_topology"
            ) -> bool:
        assert self.type == other.type, "Type mismatch."
        if (self.psf_filename == other.psf_filename) \
                and (self.param_filename_list == other.param_filename_list):
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.type == new_md_parameters_topology.type, "Type mismatch."
        assert self.psf_filename != "", "psf_filename must be defined."
        assert len(self.param_filename_list) > 0, \
            "param_filename_list must be defined."
        psf_basename = os.path.basename(self.psf_filename)
        anchor_psf_full_path = os.path.join(dest_directory,
                                            psf_basename)
        full_psf_filename = os.path.expanduser(self.psf_filename)
        logging.info(f"Copying Charmm psf file: {full_psf_filename} "
                        f"to anchor directory: {anchor_psf_full_path}.")
        copyfile(full_psf_filename, anchor_psf_full_path)
        new_md_parameters_topology.psf_filename = psf_basename
        new_md_parameters_topology.param_filename_list = []
        for param_filename in self.param_filename_list:
            param_basename = os.path.basename(param_filename)
            anchor_param_full_path = os.path.join(dest_directory,
                                                param_basename)
            full_param_filename = os.path.expanduser(param_filename)
            logging.info(f"Copying Charmm parameter file: {full_param_filename} "
                            f"to anchor directory: {anchor_param_full_path}.")
            copyfile(full_param_filename, anchor_param_full_path)
            new_md_parameters_topology.param_filename_list.append(param_basename)
        return
   
# TODO: remove support for this - users can create their own systems from an
#  OpenMM FF if they want to and save it as a system XML.
@define
class Forcefield_parameters:
    type: typing.Literal["OpenMM_forcefield"] = "OpenMM_forcefield"
    built_in_forcefield_filenames: list[str] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(list),
        ))
    custom_forcefield_filenames: list[str] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(list),
        ))
    
    def same_parameters(
            self, 
            other: "Parameters_topology"
            ) -> bool:
        assert self.type == other.type, "Type mismatch."
        if (self.built_in_forcefield_filenames \
                == other.built_in_forcefield_filenames) \
                and (self.custom_forcefield_filenames \
                     == other.custom_forcefield_filenames):
            return True
        else:
            return False
        
    def copy_files(
            self,
            dest_directory: str,
            new_md_parameters_topology: "Parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.type == new_md_parameters_topology.type, "Type mismatch."
        new_md_parameters_topology.built_in_forcefield_filenames = []
        for forcefield_filename in self.built_in_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(
                dest_directory, forcefield_basename)
            full_forcefield_filename = os.path.expanduser(forcefield_filename)
            logging.info(f"Copying OpenMM forcefield file: {full_forcefield_filename} "
                            f"to anchor directory: {anchor_forcefield_full_path}.")
            copyfile(full_forcefield_filename, anchor_forcefield_full_path)
            new_md_parameters_topology.built_in_forcefield_filenames.append(
                forcefield_basename)

        new_md_parameters_topology.custom_forcefield_filenames = []
        for forcefield_filename in self.custom_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(dest_directory,
                                                    forcefield_basename)
            full_forcefield_filename = os.path.expanduser(forcefield_filename)
            logging.info(f"Copying OpenMM forcefield file: {full_forcefield_filename} "
                            f"to anchor directory: {anchor_forcefield_full_path}.")
            copyfile(full_forcefield_filename, anchor_forcefield_full_path)
            new_md_parameters_topology.custom_forcefield_filenames.append(
                forcefield_basename)

        return
    
@define
class Openmm_system:
    type: typing.Literal["OpenMM_system"] = "OpenMM_system"
    # TODO: add topology file? Make it optional?
    system_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(
            self, 
            other: "Parameters_topology"
            ) -> bool:
        assert self.type == other.type, "Type mismatch."
        if self.system_filename == other.system_filename:
            return True
        else:
            return False
        
    def copy_files(
            self,
            dest_directory: str,
            new_md_parameters_topology: "Parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.type == new_md_parameters_topology.type, "Type mismatch."
        assert self.system_filename != "", "system_filename must be defined."
        system_basename = os.path.basename(self.system_filename)
        anchor_system_full_path = os.path.join(dest_directory,
                                            system_basename)
        full_system_filename = os.path.expanduser(self.system_filename)
        logging.info(f"Copying OpenMM system file: {full_system_filename} "
                        f"to anchor directory: {anchor_system_full_path}.")
        copyfile(full_system_filename, anchor_system_full_path)
        new_md_parameters_topology.system_filename = system_basename
        return

Parameters_topology = typing.Union[
    Amber_parameters_topology, Gromacs_parameters_topology, 
    Charmm_parameters_topology, Forcefield_parameters, Openmm_system]

# This object logically belongs here, but many of the methods are
#  better suited for the preparer module. Move these methods to
#  functions within the preparer module.
@define
class MD_unpartitioned_settings:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=base.UNPARTITIONED_DIRECTORY, 
                           validator=validators.instance_of(str))
    stage_parameters: dict[str, dict | None] = field(default=Factory(dict),)
    md_parameters_topology: Parameters_topology | None = None
    selections: list[Selection] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(Selection),
            iterable_validator=validators.instance_of(list),
        ))
    
    def generate_filetree(
            self, 
            root_directory: str
            ) -> None:
        """
        Generate the filetree for this unpartitioned settings object
        """
        unpartitioned_dict: dict[str, dict] = {self.directory: {}}
        for stage_parameter in self.stage_parameters.values():
            if stage_parameter is not None:
                stage_directory = stage_parameter["directory"]
                unpartitioned_dict[self.directory][stage_directory] = {}
        base.make_tree(root_directory, unpartitioned_dict)
        return

    def get_stage_parameters_by_index(
            self,
            stage_index: int
            ) -> dict:
        """
        Get the stage parameters for a given stage index.
        """
        for stage_parameters in self.stage_parameters.values():
            if stage_parameters["index"] == stage_index:
                return stage_parameters
        raise Exception(f"Stage parameters with index {stage_index} not found.")
    
    def get_positions_full_paths(
            self,
            root_directory: str,
            input_stage_index: int,
            dummy_set_name: str | None = None,
            swarm_id: int | None = None
            ) -> list[str]:
        """
        Construct the full path to the PDB/TXT for this unpartitioned settings object and a 
        given stage.
        """
        # NOTE: set_name isn't used here, but needs to be an argument for
        # anchors, which are ambiguously lumped with unpartitioned settings 
        # in some contexts. So, we just pass it
        positions_base_noext = self.get_stage_parameters_by_index(
            input_stage_index)["positions"]
        assert isinstance(positions_base_noext, str), \
            "Positions must be a filename str."
        input_stage_dir = self.get_stage_parameters_by_index(
            input_stage_index)["directory"]
        if swarm_id is None:
            glob_base = [f"{positions_base_noext}*.pdb", 
                         f"{positions_base_noext}*.txt"]
        else:
            glob_base = [f"{positions_base_noext}*.swarm_{swarm_id}.pdb",
                         f"{positions_base_noext}*.swarm_{swarm_id}.txt"]
        glob_path = os.path.join(
            root_directory, self.directory, input_stage_dir)
        positions_full_paths = base.glob_or(glob_base, glob_path)
        return positions_full_paths

    #def get_starting_structure_num_swarms(
    def get_starting_structure_swarm_ids(
            self,
            root_directory: str,
            input_stage_index: int,
            dummy_set_name: str
            ) -> list[int]:
        """
        Return the number of frames within a starting structure
        for this unpartitioned settings object. AKA: swarm size.
        """
        # NOTE: set_name isn't used here, but needs to be an argument for
        # anchors, which are ambiguously lumped with unpartitioned settings 
        # in some contexts. So, we just pass it
        pdb_full_paths = self.get_positions_full_paths(
            root_directory, input_stage_index)
        #num_frames = len(pdb_full_paths)
        #return num_frames
        swarm_ids = []
        # Loop through the paths, and use the re module to extract
        #  instances where .swarm_123456. is in the filename
        for pdb_full_path in pdb_full_paths:
            match = re.search(r"swarm_(\d+)", pdb_full_path)
            if match:
                swarm_id = int(match.group(1))
                swarm_ids.append(swarm_id)
        swarm_ids.sort()
        return swarm_ids
    
    def get_atom_indices_by_selection_name(
            self,
            name: str
            ) -> list[int]:
        """
        Get the atom indices for a given selection name.
        """
        for selection in self.selections:
            if selection.name == name:
                return selection.atom_indices
        raise Exception(f"Selection with name {name} not found.")

    """
    def get_stage_output_directory(
            self,
            stage_index: int
            ) -> str:
        ""
        Get the output directory for this anchor for a given stage index.
        ""
        stage_output_directory \
            = self.get_stage_parameters_by_index(stage_index)["directory"]
        return stage_output_directory
    """
    def get_stage_output_directory(
            self,
            stage_name: str
            ) -> str:
        """
        Get the output directory for this anchor for a given stage name.
        """
        stage_output_directory \
            = self.stage_parameters[stage_name]["directory"]
        return stage_output_directory

    def save_selections(
            self,
            directory: str,
            ) -> None:
        """
        Save the selections to files.
        """
        for selection in self.selections:
            selection.save(directory, scale_name="mdScale")

    def load_selections(
            self,
            directory: str,
            ) -> None:
        """
        Load the selections from a directory.
        """
        filename_glob = os.path.join(directory, "*_mdScale.json")
        selection_files = glob.glob(filename_glob)
        loaded_selections = []
        for selection_file in selection_files:
            new_selection = Selection()
            new_selection.load(selection_file)
            loaded_selections.append(new_selection)
        self.selections = loaded_selections
            
class Base_engine(typing.Protocol):
    """
    Base class for all engines.
    """
    type: typing.Literal["base"] = "base"
    
    def can_handle_toy(self) -> bool: ...

    def get_random_seed(self) -> int: ...

    def set_random_seed(self, seed: int) -> None: ...

    def save_selections(self, directory: str) -> None: ...

    def load_selections(self, directory: str) -> None: ...

class Base_engine_settings_input(typing.Protocol):
    """
    Base engine settings class for all engines.
    """
    type: typing.Literal["base"] = "base"

