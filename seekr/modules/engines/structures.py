"""
modules/engines/structures.py

Import base objects from all implemented engines.
"""

import os
import json
import typing
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
    atom_indices: list[int] = field(default=Factory(list),
                                    validator=validators.deep_iterable(
                                        member_validator=[validators.instance_of(int),
                                                    validators.ge(0)],
                                        iterable_validator=validators.instance_of(list),
                                    ))

@define
class Checkpoint_information:
    """
    A class to store information about the current state of a SEEKR
    calculation.
    """
    current_step: int = field(default=0, 
                              validator=[validators.instance_of(int),
                                               validators.ge(0)])
    restart_index: int = field(default=1,
                                validator=[validators.instance_of(int),
                                           validators.ge(1)])
    timestep: float = field(default=0.002,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])
    
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
        filename: str) -> None:
    """
    Write the checkpoint information to a file.
    """
    checkpoint_information = Checkpoint_information(step, restart_index, timestep)
    checkpoint_information.save(filename)
    return

@define
class Langevin_integrator_settings:
    type: typing.Literal["langevin"] = "langevin"
    timestep: float = field(default=0.002,
                            validator=[validators.instance_of(float),
                                       validators.gt(0)])
    friction: float = field(default=1.0,
                            validator=[validators.instance_of(float),
                                       validators.ge(0)])
    target_temperature: float = field(default=298.15,
                                      validator=[validators.instance_of(float),
                                                 validators.gt(0)])
    random_seed: int = field(default=0,
                             validator=[validators.instance_of(int)])
    rigid_tolerance: float = field(default=1e-6,
                                   validator=[validators.instance_of(float),
                                              validators.gt(0)])

Integrator_settings = typing.Union[Langevin_integrator_settings]

@define
class Cuda_platform_settings:
    type: typing.Literal["cuda"] = "cuda"
    device_index: str = field(default="0",
                              validator=[validators.instance_of(str)])
    precision: str = field(default="mixed",
                           #converter=str.lower,
                           converter=base.string_lower,
                           validator=[validators.instance_of(str),
                                      validators.in_(["single", 
                                                      "mixed",
                                                      "double"])])

Platform_settings = typing.Union[Cuda_platform_settings]

@define
class Toy_settings:
    potential_energy_expression: str = field(default="",
                                             validator=[validators.instance_of(str)])
    num_particles: int = field(default=0, 
                               validator=[validators.instance_of(int),
                                          validators.ge(0)])
    masses: list[float] = field(default=Factory(list),)
                                #validator=validators.deep_iterable(
                                #    member_validator=[validators.instance_of(float),
                                #                      validators.ge(0)],
                                #    iterable_validator=validators.instance_of(list),
                                #))
    
@define
class Amber_parameters_topology:
    type: typing.Literal["Amber"] = "Amber"
    prmtop_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(
            self, 
            other: "Amber_parameters_topology"
            ) -> bool:
        if self.prmtop_filename == other.prmtop_filename:
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Amber_parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.prmtop_filename != "", "prmtop_filename must be defined."
        prmtop_basename = os.path.basename(self.prmtop_filename)
        anchor_prmtop_full_path = os.path.join(dest_directory,
                                            prmtop_basename)
        copyfile(os.path.expanduser(self.prmtop_filename), anchor_prmtop_full_path)
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
            other: "Gromacs_parameters_topology"
            ) -> bool:
        if (self.top_filename == other.top_filename) \
                and (self.gro_filename == other.gro_filename):
            return True
        else:
            return False
    
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Gromacs_parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.top_filename != "", "top_filename must be defined."
        assert self.gro_filename != "", "gro_filename must be defined."
        top_basename = os.path.basename(self.top_filename)
        anchor_top_full_path = os.path.join(dest_directory,
                                            top_basename)
        copyfile(os.path.expanduser(self.top_filename), anchor_top_full_path)
        new_md_parameters_topology.top_filename = top_basename
        gro_basename = os.path.basename(self.gro_filename)
        anchor_gro_full_path = os.path.join(dest_directory,
                                            gro_basename)
        copyfile(os.path.expanduser(self.gro_filename), anchor_gro_full_path)
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
            other: "Charmm_parameters_topology"
            ) -> bool:
        if (self.psf_filename == other.psf_filename) \
                and (self.param_filename_list == other.param_filename_list):
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Charmm_parameters_topology") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.psf_filename != "", "psf_filename must be defined."
        assert len(self.param_filename_list) > 0, "param_filename_list must be defined."
        psf_basename = os.path.basename(self.psf_filename)
        anchor_psf_full_path = os.path.join(dest_directory,
                                            psf_basename)
        copyfile(os.path.expanduser(self.psf_filename), anchor_psf_full_path)
        new_md_parameters_topology.psf_filename = psf_basename
        new_md_parameters_topology.param_filename_list = []
        for param_filename in self.param_filename_list:
            param_basename = os.path.basename(param_filename)
            anchor_param_full_path = os.path.join(dest_directory,
                                                param_basename)
            copyfile(os.path.expanduser(param_filename), anchor_param_full_path)
            new_md_parameters_topology.param_filename_list.append(param_basename)
        return
    
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
            other: "Forcefield_parameters"
            ) -> bool:
        if (self.built_in_forcefield_filenames == other.built_in_forcefield_filenames) \
                and (self.custom_forcefield_filenames == other.custom_forcefield_filenames):
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Forcefield_parameters") -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        new_md_parameters_topology.build_in_forcefield_filenames = []
        for forcefield_filename in self.build_in_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(dest_directory,
                                                    forcefield_basename)
            copyfile(os.path.expanduser(forcefield_filename), anchor_forcefield_full_path)
            new_md_parameters_topology.build_in_forcefield_filenames.append(forcefield_basename)

        new_md_parameters_topology.custom_forcefield_filenames = []
        for forcefield_filename in self.custom_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(dest_directory,
                                                    forcefield_basename)
            copyfile(os.path.expanduser(forcefield_filename), anchor_forcefield_full_path)
            new_md_parameters_topology.custom_forcefield_filenames.append(forcefield_basename)

        return
    
@define
class Openmm_system:
    type: typing.Literal["OpenMM_system"] = "OpenMM_system"
    system_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(
            self, 
            other: "Openmm_system"
            ) -> bool:
        if self.system_filename == other.system_filename:
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str,
                   new_md_parameters_topology: "Openmm_system") -> None:
          """
          Copy the files for the initial 'stage' from this Anchor_input
          to the actual anchor.
          """
          assert self.system_filename != "", "system_filename must be defined."
          system_basename = os.path.basename(self.system_filename)
          anchor_system_full_path = os.path.join(dest_directory,
                                              system_basename)
          copyfile(os.path.expanduser(self.system_filename), anchor_system_full_path)
          new_md_parameters_topology.system_filename = system_basename
          return

Parameters_topology = typing.Union[
    Amber_parameters_topology, Gromacs_parameters_topology, Charmm_parameters_topology, 
    Forcefield_parameters, Openmm_system]

# This object logically belongs here, but many of the methods are
#  better suited for the preparer module. Move these methods to
#  functions within the preparer module.
@define
class MD_systemwide_settings:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=base.SYSTEMWIDE_DIRECTORY, 
                           validator=validators.instance_of(str))
    stage_parameters: list[dict | None] = field(default=Factory(list),)
                                        #validator=validators.deep_iterable(
                                        #    member_validator=validators.instance_of(dict),
                                        #    iterable_validator=validators.instance_of(list),
                                        #))
    md_parameters_topology: Parameters_topology | None = None
    selections: list[Selection] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(Selection),
            iterable_validator=validators.instance_of(list),
        ))
    
    # TODO: move to preparer module - and combine with anchor generate_filetree()?
    def generate_filetree(
            self, 
            root_directory: str
            ) -> None:
        """
        Generate the filetree for this systemwide settings object
        """
        systemwide_dict: dict[str, dict] = {self.directory: {}}
        for stage_parameter in self.stage_parameters:
            if stage_parameter is not None:
                stage_directory = stage_parameter["directory"]
                systemwide_dict[self.directory][stage_directory] = {}
        base.make_tree(root_directory, systemwide_dict)
        return

    def get_positions_full_paths(
            self,
            root_directory: str,
            input_stage_index: int
            ) -> list[str]:
        """
        Construct the full path to the PDB/TXT for this systemwide and a given stage.
        """
        assert len(self.stage_parameters) > input_stage_index, "Stage index "\
            "provided is out of range."
        assert self.stage_parameters[input_stage_index] is not None
        positions_base_noext = self.stage_parameters[input_stage_index]["positions"]
        assert isinstance(positions_base_noext, str), "Positions must be a filename str."
        input_stage_dir = self.stage_parameters[input_stage_index]["directory"]
        glob_base = [f"{positions_base_noext}*.pdb", f"{positions_base_noext}*.txt"]
        glob_path = os.path.join(
            root_directory, self.directory, input_stage_dir)
        positions_full_paths = base.glob_or(glob_base, glob_path)
        return positions_full_paths

    def get_starting_structure_num_frames(
            self,
            root_directory: str,
            input_stage_index: int
            ) -> int:
        """
        Return the number of frames within a starting structure
        for this systemwide. AKA: swarm size.
        """
        pdb_full_paths = self.get_positions_full_paths(
            root_directory, input_stage_index)
        num_frames = len(pdb_full_paths)
        return num_frames
    
    # TODO: remove? Handled by the input object.
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