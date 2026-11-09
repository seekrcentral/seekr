from __future__ import annotations
"""
modules/restraints/preparer.py

Prepare the restraints inputs for simulations.
"""

import os
import shutil
import typing

from attrs import define, field, validators, Factory
import numpy as np

import seekr.modules.restraints.structures as restraints_structures
import seekr.modules.engines.preparer as engines_preparer

class Base_restraint_input(typing.Protocol):
    """
    Base class for restraint inputs.
    """
    type: typing.Literal["base"] = "base"

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input
            ) -> restraints_structures.Base_restraint:
        """
        Make the restraint object for the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

@define
class Positional_restraint_input(Base_restraint_input):
    """
    Input for positional restraints.
    """
    type: typing.Literal["positional"] = "positional"
    selection_name: str = field(default="",
                                 validator=validators.instance_of(str))
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])
    coordinates_filename: typing.Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)))

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Positional_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection_name)
        restraint = restraints_structures.Positional_restraint()
        restraint.atom_indices = atom_indices
        restraint.force_constant = self.force_constant
        if self.coordinates_filename is not None:
            # Check that coordinates and atom_indices align.
            positions = np.loadtxt(self.coordinates_filename, ndmin=2)

            assert positions.shape[0] == len(atom_indices), \
                f"Number of coordinate sets ({positions.shape[0]}) "\
                f"in {self.coordinates_filename} does not "\
                f"match number of atom indices ({len(atom_indices)})"
            coordinates_basename = os.path.basename(self.coordinates_filename)
            coordinates_destination = os.path.join(
                root_directory, coordinates_basename)
            if not os.path.samefile(self.coordinates_filename, 
                                    coordinates_destination):
                shutil.copyfile(self.coordinates_filename, coordinates_destination)
            restraint.coordinates_filename = coordinates_basename

        return restraint

@define
class Z_restraint_input(Base_restraint_input):
    """
    Input for Z restraints.
    """
    type: typing.Literal["z"] = "z"
    selection_name: str = field(default="",
                                 validator=validators.instance_of(str))
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])
    coordinates_filename: typing.Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)))

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Z_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection_name)
        restraint = restraints_structures.Z_restraint()
        restraint.atom_indices = atom_indices
        restraint.force_constant = self.force_constant
        if self.coordinates_filename is not None:
            # Check that coordinates and atom_indices align.
            positions = np.loadtxt(self.coordinates_filename, ndmin=2)
            assert positions.shape[0] == len(atom_indices), \
                f"Number of coordinate sets ({positions.shape[0]}) "\
                f"in {self.coordinates_filename} does not "\
                f"match number of atom indices ({len(atom_indices)})"
            coordinates_basename = os.path.basename(self.coordinates_filename)
            coordinates_destination = os.path.join(root_directory, coordinates_basename)
            if not os.path.samefile(self.coordinates_filename, coordinates_destination):
                shutil.copyfile(self.coordinates_filename, coordinates_destination)
            restraint.coordinates_filename = coordinates_basename

        return restraint

@define
class Torsion_restraint_input(Base_restraint_input):
    """
    Input for torsion restraints.
    """
    type: typing.Literal["torsion"] = "torsion"
    torsion_indices: list[tuple[int, int, int, int]] = field(default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(tuple),
                              validators.min_len(4),
                              validators.max_len(4)],
            iterable_validator=validators.instance_of(list),))
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])
    torsion_values: list[float] = field(default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(float)],
            iterable_validator=validators.instance_of(list),))

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Torsion_restraint:
        """
        Make the restraint object for the model.
        """
        restraint = restraints_structures.Torsion_restraint()
        restraint.torsion_indices = self.torsion_indices
        restraint.force_constant = self.force_constant
        restraint.torsion_values = self.torsion_values
        assert len(restraint.torsion_values) == len(restraint.torsion_indices), \
            "Number of torsion values does not match number of torsion indices."
        for torsion_value in restraint.torsion_values:
            assert -np.pi <= torsion_value <= np.pi, \
                "Torsion value is not within the range of -pi to pi."
        return restraint

@define
class Pairwise_restraint_input(Base_restraint_input):
    """
    Input for pairwise restraints.
    """
    type: typing.Literal["pairwise"] = "pairwise"
    selection_name: str = field(default="",
                                 validator=validators.instance_of(str))
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Pairwise_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection_name)
        # NOTE: not doing it this way - creates too many number pairs in the model
        #  file.
        #atom_index_pairs = []
        #for atom_index1 in atom_indices:
        #    for atom_index2 in atom_indices:
        #        if atom_index1 < atom_index2:
        #            atom_index_pairs.append([atom_index1, atom_index2])
        restraint = restraints_structures.Pairwise_restraint()
        restraint.atom_indices = atom_indices
        restraint.force_constant = self.force_constant
        return restraint
    
@define
class Proximity_restraint_input(Base_restraint_input):
    """
    Input for proximity restraints.
    """
    type: typing.Literal["proximity"] = "proximity"
    selection1_name: str = field(default="",
                                 validator=validators.instance_of(str))
    selection2_name: str = field(default="",
                                 validator=validators.instance_of(str))
    radius: float = field(default=2.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])
    force_constant: float = field(default=1000.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Proximity_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices1 = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection1_name)
        atom_indices2 = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection2_name)
        restraint = restraints_structures.Proximity_restraint()
        restraint.atom_indices1 = atom_indices1
        restraint.atom_indices2 = atom_indices2
        restraint.radius = self.radius
        restraint.force_constant = self.force_constant
        return restraint
    
@define
class Cylinder_restraint_input(Base_restraint_input):
    """
    Input for cylinder restraint.
    """
    type: typing.Literal["cylinder"] = "cylinder"
    selection1_name: str = field(default="",
                                 validator=validators.instance_of(str))
    selection2_name: str = field(default="",
                                 validator=validators.instance_of(str))
    selection3_name: str = field(default="",
                                 validator=validators.instance_of(str))
    radius: float = field(default=2.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])
    force_constant: float = field(default=1000.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])

    def make_restraint(
            self, 
            unpartitioned_settings: engines_preparer.MD_unpartitioned_settings_input,
            root_directory: str
            ) -> restraints_structures.Cylinder_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices1 = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection1_name)
        atom_indices2 = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection2_name)
        atom_indices3 = unpartitioned_settings.get_atom_indices_by_selection_name(
            self.selection3_name)
        restraint = restraints_structures.Cylinder_restraint()
        restraint.atom_indices1 = atom_indices1
        restraint.atom_indices2 = atom_indices2
        restraint.atom_indices3 = atom_indices3
        restraint.radius = self.radius
        restraint.force_constant = self.force_constant
        return restraint