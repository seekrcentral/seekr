"""

"""

import os
import glob
import typing
from shutil import copyfile

from attrs import define, field, validators, Factory

import seekr.modules.base as base
import seekr.modules.run_base as run_base

SYSTEMWIDE_DIRECTORY = "systemwide"

if typing.TYPE_CHECKING:
    import seekr.modules.data_structures as data_structures

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

# Otherwise Mypy has a problem with the str.lower method.
def string_lower(string: str | None) -> str | None:
    if isinstance(string, str):
        return string.lower()
    elif string is None:
        return None
    else:
        raise TypeError("string_lower only accepts strings or None.")

@define
class Cuda_platform_settings:
    type: typing.Literal["cuda"] = "cuda"
    device_index: str = field(default="0",
                              validator=[validators.instance_of(str)])
    precision: str = field(default="mixed",
                           #converter=str.lower,
                           converter=string_lower,
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
    
    def same_parameters(self, other):
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
    
    def same_parameters(self, other):
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
    
    def same_parameters(self, other):
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
    
    def same_parameters(self, other):
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

#Positions_type = typing.Union[list, str, None]
Positions_type = typing.Union[typing.List, str, None]

@define
class MD_systemwide_settings:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=SYSTEMWIDE_DIRECTORY, 
                           validator=validators.instance_of(str))
    stage_parameters: list[dict] = field(default=Factory(list),
                                        validator=validators.deep_iterable(
                                            member_validator=validators.instance_of(dict),
                                            iterable_validator=validators.instance_of(list),
                                        ))
    md_parameters_topology: Parameters_topology | None = None
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
        
        systemwide_dict: dict[str, dict] = {self.directory: {}}
        #md_parameters_topology = self.stage_parameters[0]["md_parameters_topology"]
        md_parameters_topology = self.md_parameters_topology
        positions = self.stage_parameters[0]["positions"]
        #if (md_parameters_topology is not None) \
        #        or (positions is not None):
        #    systemwide_dict[self.directory][base.INITIAL_DIRECTORY] = {}
        for stage_parameter in self.stage_parameters:
            stage_directory = stage_parameter["directory"]
            systemwide_dict[self.directory][stage_directory] = {}
        base.make_tree(root_directory, systemwide_dict)

        return

    def _get_positions_pdb_full_path(
            self,
            model: "data_structures.Seekr_model",
            input_stage_index: int
            ) -> str:
        """
        Construct the full path to the PDB for this systemwide and stage.
        """
        positions = self.stage_parameters[input_stage_index]["positions"]
        assert isinstance(positions, str), "Positions must be a filename str."
        input_stage_dir = self.stage_parameters[input_stage_index]["directory"]
        positions_full_path: str = os.path.join(
            model.directory, self.directory, input_stage_dir, positions)
        return positions_full_path

    def get_starting_structure_num_frames(
            self,
            model: "data_structures.Seekr_model",
            input_stage_index: int
            ) -> int:
        """
        Return the number of frames within a starting structure
        for this systemwide. AKA: swarm size.
        """
        positions = self.stage_parameters[input_stage_index]["positions"]
        if isinstance(positions, str):
            # Load the PDB file and count the number of frames
            pdb_full_path = self._get_positions_pdb_full_path(model, input_stage_index)
            num_frames = model.md_engine.get_pdb_num_frames(pdb_full_path)
            return num_frames
        elif isinstance(positions, list):
            return len(positions)
        else:
            return 0
    
    def cleanse_outputs(
            self,
            model: "Seekr_model"
            ) -> None:
        """
        Delete all simulation outputs from systemwide.
        """
        for i, stage in enumerate(model.stages):
            full_directory = os.path.join(model.directory, self.directory, 
                                          stage.get_directory_name())
            if os.path.exists(full_directory):
                file_globs = stage.sampling_method.get_output_globs()
                for file_glob in file_globs:
                    full_glob = os.path.join(full_directory, file_glob)
                    for file in glob.glob(full_glob):
                        os.remove(file)

        return

    def get_current_step(
            self, 
            stage_index: int, 
            swarm_index: int | None,
            root_directory: str):
        """
        Return the current step for a given stage index.
        """
        stage_directory = self.stage_parameters[stage_index]["directory"]
        full_directory = os.path.join(root_directory, self.directory, stage_directory)
        checkpoint_info_basename = run_base.get_checkpoint_information_filename(swarm_index)
        checkpoint_information_filename = os.path.join(
            full_directory, checkpoint_info_basename)
        if os.path.exists(checkpoint_information_filename):
            checkpoint_information = run_base.load_checkpoint_information(
                checkpoint_information_filename)
            current_step = checkpoint_information.current_step
        else: 
            current_step = 0
        return current_step
    
    def get_restart_checkpoint_filename(
            self, 
            stage_index: int, 
            swarm_index: int | None,
            root_directory: str):
        """
        Return the restart checkpoint filename for a given stage index.
        """
        stage_directory = self.stage_parameters[stage_index]["directory"]
        full_directory = os.path.join(root_directory, self.directory, stage_directory)
        restart_checkpoint_basename = run_base.get_restart_checkpoint_filename(swarm_index)
        restart_checkpoint_filename = os.path.join(
            full_directory, restart_checkpoint_basename)
        return restart_checkpoint_filename
    
@define
class MD_systemwide_settings_input:
    # To be used for a common set of parameters for all anchors
    # within MD simulations
    directory: str = field(default=SYSTEMWIDE_DIRECTORY, 
                           validator=validators.instance_of(str))
    md_parameters_topology: Parameters_topology | None = None
    # This can be a position assignment to be automatically placed into an anchor
    positions: Positions_type = None
    selections: list[base.Selection] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(base.Selection),
            iterable_validator=validators.instance_of(list),
        ))
    
    def make_systemwide_settings_for_model(self) -> MD_systemwide_settings:
        new_md_systemwide_settings = MD_systemwide_settings()
        new_md_systemwide_settings.directory = self.directory
        new_md_systemwide_settings.selections = self.selections
        if self.positions is not None and self.positions != "":
            stage_parameters = \
                [{"directory": base.INITIAL_DIRECTORY, "positions": self.positions}]
        else:
            stage_parameters = []
        new_md_systemwide_settings.stage_parameters = stage_parameters
        new_md_systemwide_settings.md_parameters_topology = self.md_parameters_topology
        return new_md_systemwide_settings

    def copy_initial_files(
            self, 
            root_directory: str,
            new_systemwide_settings: MD_systemwide_settings,
            toy_mode: bool
            ) -> None:
        """
        Copy the files for the initial 'stage' from this Anchor_input
        to the actual anchor.
        """
        systemwide_initial_directory = os.path.join(
            root_directory, self.directory, base.INITIAL_DIRECTORY)
        if toy_mode:
            # TODO: remove
            #new_systemwide_settings.positions = self.positions
            #new_systemwide_settings.stage_parameters[0]["positions"] = self.positions
            pass

        else:
            if self.md_parameters_topology is not None \
                    or (self.positions is not None \
                        and self.positions != ""):
                new_systemwide_settings.generate_filetree(root_directory)

            if self.md_parameters_topology is not None:
                #new_systemwide_settings.stage_parameters[0]["md_parameters_topology"] \
                #    = self.md_parameters_topology
                new_systemwide_settings.md_parameters_topology = self.md_parameters_topology
                self.md_parameters_topology.copy_files(
                    systemwide_initial_directory,
                    new_systemwide_settings.md_parameters_topology)

            if (self.positions is not None) and (self.positions != ""):
                assert isinstance(self.positions, str), \
                    "MD_systemwide_settings.positions must be a string."
                pdb_basename = os.path.basename(self.positions)
                anchor_pdb_full_path = os.path.join(systemwide_initial_directory,
                                                    pdb_basename)
                copyfile(os.path.expanduser(self.positions), anchor_pdb_full_path)
                new_systemwide_settings.stage_parameters[0]["positions"] = pdb_basename
