"""
modules/restraints/preparer.py

Prepare the restraints inputs for simulations.
"""

import typing
from attrs import define, field, validators, Factory

import seekr.modules.engines.structures as engines_structures
import seekr.modules.restraints.structures as restraints_structures
import seekr.modules.engines.preparer as engines_preparer

class Base_restraint_input(typing.Protocol):
    """
    Base class for restraint inputs.
    """
    type: typing.Literal["base"] = "base"

    def make_restraint(
            self, 
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
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

    def make_restraint(
            self, 
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            ) -> restraints_structures.Positional_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices = systemwide_settings.get_atom_indices_by_selection_name(
            self.selection_name)
        restraint = restraints_structures.Positional_restraint()
        restraint.atom_indices = atom_indices
        restraint.force_constant = self.force_constant
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
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            ) -> restraints_structures.Pairwise_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices = systemwide_settings.get_atom_indices_by_selection_name(
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
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            ) -> restraints_structures.Proximity_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices1 = systemwide_settings.get_atom_indices_by_selection_name(
            self.selection1_name)
        atom_indices2 = systemwide_settings.get_atom_indices_by_selection_name(
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
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            ) -> restraints_structures.Cylinder_restraint:
        """
        Make the restraint object for the model.
        """
        atom_indices1 = systemwide_settings.get_atom_indices_by_selection_name(
            self.selection1_name)
        atom_indices2 = systemwide_settings.get_atom_indices_by_selection_name(
            self.selection2_name)
        atom_indices3 = systemwide_settings.get_atom_indices_by_selection_name(
            self.selection3_name)
        restraint = restraints_structures.Cylinder_restraint()
        restraint.atom_indices1 = atom_indices1
        restraint.atom_indices2 = atom_indices2
        restraint.atom_indices3 = atom_indices3
        restraint.radius = self.radius
        restraint.force_constant = self.force_constant
        return restraint