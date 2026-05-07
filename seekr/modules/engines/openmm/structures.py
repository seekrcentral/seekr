"""
modules/engines/openmm/structures.py

Structures for the OpenMM engine - anything that might appear in the model.json.
"""

import typing
from attrs import define, field, validators
from attrs.validators import instance_of, optional, in_, gt, and_

if typing.TYPE_CHECKING:
    import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures

@define
class Monte_carlo_barostat_settings:
    target_pressure: float = field(default=1.0,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    target_temperature: float = field(default=298.15,
                                      validator=[validators.instance_of(float),
                                                 validators.gt(0)])
    frequency: int = field(default=25,
                           validator=[validators.instance_of(int),
                                      validators.gt(0)])
    membrane: bool = field(default=False,
                           validator=validators.instance_of(bool))

Barostat_settings = typing.Union[Monte_carlo_barostat_settings]

@define
class Openmm_md_engine(engines_structures.Base_engine):
    type: typing.Literal["openmm"] = "openmm"
    scale_type: typing.Literal["molecular_dynamics"] = "molecular_dynamics"
    nonbonded_method: str | None = field(
        default="pme", converter=base.string_lower)
    nonbonded_cutoff: float | None = field(
        default=0.9)
    constraints: str | None = field(
        default="hbonds", converter=base.string_lower, )
    hydrogen_mass: float | None = field(
        default=None, validator=optional(and_(instance_of(float), gt(0))))
    initial_temperature: float = field(
        default=298.15, validator=[instance_of(float), gt(0)])
    integrator_settings: engines_structures.Integrator_settings | None = None
    # If platform_settings is None, then use Reference platform
    platform_settings: engines_structures.Platform_settings | None = None
    toy_settings: engines_structures.Toy_settings | None = None
    unpartitioned_settings: engines_structures.MD_unpartitioned_settings | None = None

    def get_pdb_object(
            self, 
            pdb_filename: str) -> "openmm_app.PDBFile":
        """
        Using the OpenMM engine, get a PDB object from a PDB file.
        """
        import openmm.app as openmm_app
        pdb_obj = openmm_app.PDBFile(pdb_filename)
        return pdb_obj

    def get_pdb_num_frames(
            self, 
            pdb_filename: str) -> int:
        """
        Using the OpenMM engine, get the number of frames in a PDB file.
        """
        pdb_obj = self.get_pdb_object(pdb_filename)
        num_frames = pdb_obj.getNumFrames()
        return num_frames
    
    def get_random_seed(
            self
            ) -> int:
        """
        Get the random seed for the OpenMM engine.
        """
        if self.integrator_settings is not None:
            if hasattr(self.integrator_settings, "random_seed"):
                return self.integrator_settings.random_seed
            
        return 0

    def set_random_seed(
            self, 
            seed: int
            ) -> None:
        """
        Set the random seed for the OpenMM engine.
        """
        if self.integrator_settings is not None:
            if hasattr(self.integrator_settings, "random_seed"):
                self.integrator_settings.random_seed = seed
        return
    
    def get_timestep(self) -> float:
        """
        Get the timestep for the OpenMM engine.
        """
        assert self.integrator_settings is not None, \
            "integrator settings not provided."
        timestep: float = self.integrator_settings.timestep
        return timestep

    def get_atom_indices_by_selection_name(
            self, 
            name: str) -> typing.List[int]:
        """
        Get the atom indices for a given selection name.
        """
        assert self.unpartitioned_settings is not None, \
            "unpartitioned settings not provided."
        return self.unpartitioned_settings.get_atom_indices_by_selection_name(name)

    def can_handle_toy(
            self
            ) -> bool:
        """
        Can this engine handle toy systems?
        """
        return True
    
    def get_device_index(
            self
            ) -> str | None:
        """
        Get the device index for the OpenMM engine.
        """
        if self.platform_settings is not None:
            if hasattr(self.platform_settings, "device_index"):
                return self.platform_settings.device_index
            
        return None
    
    def set_device_index(
            self, 
            device_index: str
            ) -> None:
        """
        Set the device index for the OpenMM engine.
        """
        if self.platform_settings is not None:
            if hasattr(self.platform_settings, "device_index"):
                self.platform_settings.device_index = device_index
        return
    
    def save_selections(
            self, 
            directory: str, 
            ) -> None:
        """
        Save the selections to files.
        """
        if self.unpartitioned_settings is not None:
            self.unpartitioned_settings.save_selections(directory)

    def load_selections(
            self,
            directory: str,
            ) -> None:
        """
        Load the selections from a directory.
        """
        if self.unpartitioned_settings is not None:
            self.unpartitioned_settings.load_selections(directory)

    def get_selection_names(
            self,
            ) -> list[str]:
        """
        Get the names of all selections.
        """
        selection_names = []
        for selection in self.unpartitioned_settings.selections:
            selection_names.append(selection.name)
        return selection_names