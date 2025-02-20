"""
engines/browndye/input.py

Objects and routines for the input stage of an anchor that
uses Browndye.
"""

import typing
from attrs import define, field, validators, Factory

import numpy as np
import parmed

import seekr.modules.base as base
import seekr.modules.engines.base as engines_base
import seekr.modules.engines.data_structures as engines_data_structures
import seekr.modules.engines.browndye.base as browndye_base

@define
class Browndye_molecule_input:
    name: str = field(default="",
                        validator=validators.instance_of(str))
    pqr_filename: str = field(default="",
                            validator=validators.instance_of(str))
    selections: list[base.Selection] = field(default=Factory(list),
                                        validator=validators.deep_iterable(
                                            member_validator=validators.instance_of(
                                                base.Selection),
                                            iterable_validator=validators.instance_of(
                                                list),
                                        ))
    
    def add_ghost_atoms_to_pqr(
            self,
            new_pqr_filename: str,
            center_molecule: bool = True,
            ) -> typing.Dict[str, int]:
        """
        Add a ghost atom to the PQR file to represent a significant location.
        Write a new PQR file and return the index of the ghost atom.
        """
        pqr_struct = parmed.load_file(self.pqr_filename, skip_bonds=True)
        centers_of_mass = []
        ghost_site_indices: typing.Dict[str, int] = {}
        for selection in self.selections:
            # Compute the center of mass for the selected group of atoms
            center_of_mass = np.array([[0., 0., 0.]])
            total_mass = 0.0
            for atom_index in selection.atom_indices:
                atom_pos = pqr_struct.coordinates[atom_index,:]
                atom_mass = pqr_struct.atoms[atom_index].mass
                if atom_mass == 0.0:
                    atom_mass = 0.0001
                center_of_mass += atom_mass * atom_pos
                total_mass += atom_mass
            center_of_mass = center_of_mass / total_mass
            centers_of_mass.append(center_of_mass)
        
        if center_molecule:
            # Compute the center of mass of the entire molecule to be transposed
            mol_center_of_mass = np.array([[0., 0., 0.]])
            mol_total_mass = 0.0
            for atom_index, atom in enumerate(pqr_struct.atoms):
                atom_pos = pqr_struct.coordinates[atom_index,:]
                atom_mass = atom.mass
                if atom_mass == 0.0:
                    atom_mass = 0.0001
                mol_center_of_mass += atom_mass * atom_pos
                mol_total_mass += atom_mass
            mol_center_of_mass = mol_center_of_mass / mol_total_mass
        
        pqr_complex = pqr_struct
        for selection, center_of_mass in zip(self.selections, centers_of_mass):
            ghost_atom = parmed.Atom(name="GHO", mass=0.0, charge=0.0, solvent_radius=0.0)
            ghost_structure = parmed.Structure()
            ghost_structure.add_atom(ghost_atom, "GHO", 1)
            ghost_structure.coordinates = np.array(center_of_mass)
            pqr_complex = pqr_complex + ghost_structure
            for residue in pqr_complex.residues:
                residue.chain = ""
            ghost_index = len(pqr_complex.atoms)
            ghost_site_indices[selection.name] = ghost_index
        
        if center_molecule:
            new_coordinates = np.zeros(pqr_complex.coordinates.shape)
            for atom_index in range(len(pqr_complex.atoms)):
                new_coordinates[atom_index,:] = pqr_complex.coordinates[atom_index,:] \
                    - mol_center_of_mass[0,:]
                    
            pqr_complex.coordinates = new_coordinates
        
        pqr_complex.save(new_pqr_filename, overwrite=True)
        return ghost_site_indices
    
@define
class Browndye_settings_input:
    type: typing.Literal["browndye"] = "browndye"
    binary_directory: str = field(default="",
                                    validator=validators.instance_of(str))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    
    ions: list[base.Ion] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    base.Ion),
                                iterable_validator=validators.instance_of(list),
                            ))
    molecules: list[Browndye_molecule_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    Browndye_molecule_input),
                                iterable_validator=validators.instance_of(list),
                            ))
    def make_model_engine(self):
        """
        Make the model engine object for this input.
        """
        bd_engine = browndye_base.Browndye_bd_engine()
        bd_engine.binary_directory = self.binary_directory
        bd_engine.apbs_grid_spacing = self.apbs_grid_spacing
        bd_engine.ions = self.ions
        bd_engine.systemwide_settings = engines_data_structures.BD_systemwide_settings()
        bd_engine.systemwide_settings.molecules = []
        return bd_engine

