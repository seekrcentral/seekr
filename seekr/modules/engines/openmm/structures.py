"""
modules/engines/openmm/structures.py

Structures for the OpenMM engine - anything that might appear in the model.json.
"""

import typing
from attrs import define, field, validators
from attrs.validators import instance_of, optional, in_, gt, and_
import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures

# TODO: move to OpenMM structures file
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
class Openmm_md_engine:
    type: typing.Literal["openmm"] = "openmm"
    nonbonded_method: str | None = field(
        default="pme", converter=base.string_lower, validator=
            optional(and_(instance_of(str), in_([
                "nocutoff", "cutoffperiodic", "cutoffnonperiodic", 
                "pme", "ljpme", "ewald"]))))
    nonbonded_cutoff: float | None = field(
        default=0.9, validator=optional(and_(instance_of(float), gt(0))))
    constraints: str | None = field(
        default="hbonds", converter=base.string_lower, 
        validator=optional(and_(instance_of(str), in_([
            "none", "hbonds", "allbonds", "hangles"]))))
    hydrogen_mass: float | None = field(
        default=None, validator=optional(and_(instance_of(float), gt(0))))
    initial_temperature: float = field(
        default=298.15, validator=[instance_of(float), gt(0)])
    integrator_settings: engines_structures.Integrator_settings | None = None
    # If platform_settings is None, then use Reference platform
    platform_settings: engines_structures.Platform_settings | None = None
    toy_settings: engines_structures.Toy_settings | None = None
    systemwide_settings: engines_structures.MD_systemwide_settings | None = None

    def get_pdb_object(
            self, 
            pdb_filename: str) -> openmm_app.PDBFile:
        """
        Using the OpenMM engine, get a PDB object from a PDB file.
        """
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
        assert self.systemwide_settings is not None, \
            "systemwide settings not provided."
        return self.systemwide_settings.get_atom_indices_by_selection_name(name)

    