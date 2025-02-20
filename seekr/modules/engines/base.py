"""

"""

import os
import typing
from shutil import copyfile
import logging

from attrs import define, field, validators, Factory

import seekr.modules.base as base

if typing.TYPE_CHECKING:
    import seekr.modules.data_structures as seekr_data_struct

SYSTEMWIDE_DIRECTORY = "systemwide"

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
                           converter=str.lower,
                           validator=[validators.instance_of(str),
                                      validators.in_(["single", 
                                                      "mixed",
                                                      "double"])])

# TODO: make a typing.Union here for other types of platforms?
Platform_settings = typing.Union[Cuda_platform_settings]

@define
class Toy_settings:
    potential_energy_expression: str = field(default="",
                                             validator=[validators.instance_of(str)])
    num_particles: int = field(default=0, 
                               validator=[validators.instance_of(int),
                                          validators.ge(0)])
    masses: list[float] = field(default=Factory(list),
                                validator=validators.deep_iterable(
                                    member_validator=[validators.instance_of(float),
                                                      validators.ge(0)],
                                    iterable_validator=validators.instance_of(list),
                                ))
    
@define
class Amber_parameters_topology:
    type: typing.Literal["Amber"] = "Amber"
    prmtop_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(self, other):
        if self.prmtop_filename == other.prmtop_filename:
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        assert self.prmtop_filename != "", "prmtop_filename must be defined."
        prmtop_basename = os.path.basename(self.prmtop_filename)
        anchor_prmtop_full_path = os.path.join(dest_directory,
                                            prmtop_basename)
        copyfile(os.path.expanduser(self.prmtop_filename), anchor_prmtop_full_path)
        return

@define
class Gromacs_parameters_topology:
    type: typing.Literal["Gromacs"] = "Gromacs"
    top_filename: str = field(default="",
                                validator=validators.instance_of(str))
    gro_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(self, other):
        if (self.top_filename == other.top_filename) \
                and (self.gro_filename == other.gro_filename):
            return True
        else:
            return False
    
    def copy_files(self,
                   dest_directory: str) -> None:
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
        gro_basename = os.path.basename(self.gro_filename)
        anchor_gro_full_path = os.path.join(dest_directory,
                                            gro_basename)
        copyfile(os.path.expanduser(self.gro_filename), anchor_gro_full_path)
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
    
    def same_parameters(self, other):
        if (self.psf_filename == other.psf_filename) \
                and (self.param_filename_list == other.param_filename_list):
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str) -> None:
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
        for param_filename in self.param_filename_list:
            param_basename = os.path.basename(param_filename)
            anchor_param_full_path = os.path.join(dest_directory,
                                                param_basename)
            copyfile(os.path.expanduser(param_filename), anchor_param_full_path)
        return
    
@define
class Forcefield_parameters:
    type: typing.Literal["OpenMM_forcefield"] = "OpenMM_forcefield"
    build_in_forcefield_filenames: list[str] = field(
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
    
    def same_parameters(self, other):
        if (self.build_in_forcefield_filenames == other.build_in_forcefield_filenames) \
                and (self.custom_forcefield_filenames == other.custom_forcefield_filenames):
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        for forcefield_filename in self.build_in_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(dest_directory,
                                                    forcefield_basename)
            copyfile(os.path.expanduser(forcefield_filename), anchor_forcefield_full_path)
        for forcefield_filename in self.custom_forcefield_filenames:
            forcefield_basename = os.path.basename(forcefield_filename)
            anchor_forcefield_full_path = os.path.join(dest_directory,
                                                    forcefield_basename)
            copyfile(os.path.expanduser(forcefield_filename), anchor_forcefield_full_path)
        return
    
@define
class Openmm_system:
    type: typing.Literal["OpenMM_system"] = "OpenMM_system"
    system_filename: str = field(default="",
                                validator=validators.instance_of(str))
    
    def same_parameters(self, other):
        if self.system_filename == other.system_filename:
            return True
        else:
            return False
        
    def copy_files(self,
                   dest_directory: str) -> None:
          """
          Copy the files for the initial 'stage' from this Anchor_input
          to the actual anchor.
          """
          assert self.system_filename != "", "system_filename must be defined."
          system_basename = os.path.basename(self.system_filename)
          anchor_system_full_path = os.path.join(dest_directory,
                                              system_basename)
          copyfile(os.path.expanduser(self.system_filename), anchor_system_full_path)
          return

Parameters_topology = typing.Union[
    Amber_parameters_topology, Gromacs_parameters_topology, Charmm_parameters_topology, 
    Forcefield_parameters, Openmm_system]

#Positions_type = typing.Union[list, str, None]
Positions_type = typing.Union[typing.List, str, None]

@define
class MD_systemwide_settings:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=SYSTEMWIDE_DIRECTORY, 
                           validator=validators.instance_of(str))
    md_parameters_topology: Parameters_topology | None = None
    # This can be a position assignment to be automatically placed into an anchor
    positions: Positions_type = None
    stage_directories: list[str] = field(
        default=Factory(list),
            validator=validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            ))
    selections: list[base.Selection] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(base.Selection),
            iterable_validator=validators.instance_of(list),
        ))
    
    def generate_filetree(
            self, 
            root_directory: str
            ) -> None:
        """
        Generate the filetree for this systemwide settings object
        """
        
        systemwide_dict = {self.directory: {}}
        if (self.md_parameters_topology is not None) \
                or (self.positions is not None):
            systemwide_dict[self.directory][base.INITIAL_DIRECTORY] = {}
        for stage_directory in self.stage_directories:
            systemwide_dict[self.directory][stage_directory] = {}
        base.make_tree(root_directory, systemwide_dict)

        return

    def copy_initial_files(
            self, 
            root_directory: str,
            new_systemwide_settings: "MD_systemwide_settings",
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        systemwide_directory = os.path.join(
            root_directory, new_systemwide_settings.directory, base.INITIAL_DIRECTORY)
        if toy_mode:
            new_systemwide_settings.positions = self.positions
        else:
            if self.md_parameters_topology is not None \
                    or (self.positions is not None \
                        and self.positions != ""):
                self.generate_filetree(root_directory)

            if self.md_parameters_topology is not None:
                self.md_parameters_topology.copy_files(systemwide_directory)

            if (self.positions is not None) and (self.positions != ""):
                pdb_basename = os.path.basename(self.positions)
                anchor_pdb_full_path = os.path.join(systemwide_directory,
                                                    pdb_basename)
                copyfile(os.path.expanduser(self.positions), anchor_pdb_full_path)
                new_systemwide_settings.positions = pdb_basename
    
