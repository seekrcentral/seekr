"""
modules/restraints/preparer.py

Prepare the restraints inputs for simulations.
"""

import typing
from attrs import define, field, validators, Factory

import seekr.modules.engines.structures as engines_structures
import seekr.modules.restraints.structures as restraints_structures
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.structures as structures

@define
class Positional_restraint_input:
    """
    Input for positional restraints.
    """
    type: typing.Literal["positional"] = "positional"
    atom_indices: list[int] = field(
        default=Factory(list),
        validator=validators.deep_iterable(
            member_validator=[validators.instance_of(int), validators.ge(0)],
            iterable_validator=validators.instance_of(list),
        )
    )
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])

    def make_restraint(
            self, 
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            #    | engines_preparer.BD_settings_input
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
class Pairwise_restraint_input:
    """
    Input for pairwise restraints.
    """
    type: typing.Literal["pairwise"] = "pairwise"
    atom_index_pairs: list[tuple[int, int]] = field(
        default=Factory(list))
    force_constant: float = field(default=400.0, validator=[
        validators.instance_of(float), validators.gt(0)
    ])

    def make_restraint(
            self, 
            systemwide_settings: engines_preparer.MD_systemwide_settings_input
            #    | engines_preparer.BD_settings_input
            ) -> restraints_structures.Pairwise_restraint:
        """
        Make the restraint object for the model.
        """
        restraint = restraints_structures.Pairwise_restraint()
        restraint.atom_index_pairs = self.atom_index_pairs
        restraint.force_constant = self.force_constant
        return restraint
    
@define
class Proximity_restraint_input:
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
            #    | engines_preparer.BD_settings_input
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
    
Restraint_input = typing.Union[
    Positional_restraint_input,
    Pairwise_restraint_input,
    Proximity_restraint_input
]