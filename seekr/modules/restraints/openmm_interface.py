"""
modules/restraints/openmm_interface.py

Restraints that interface with OpenMM.
"""

import typing

from attrs import define, field

import openmm
import openmm.unit as unit
import openmm.app as openmm_app

import seekr.modules.structures as structures
import seekr.modules.restraints.structures as restraints_structures

# TODO: choose which force group to assign restraints to. Will it be a 
# variable in the model file? Is group0 good enough?

@define
class Base_restraint_openmm_interface:
    """
    A base class for a restraint using the OpenMM engine.
    """
    _restraint: restraints_structures.Restraint
    def add_forces_to_system(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add force(s) to an OpenMM System object.
        """
        return

@define
class Positional_restraint_openmm_interface:
    """
    A Positional restraint using the OpenMM engine. This keeps
    the set of atoms or groups close to their starting positions.
    This keeps a receptor at the same position and orientation in
    Cartesian space, for example.
    """
    _restraint: restraints_structures.Positional_restraint = field()
    def add_forces_to_system(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add force(s) to an OpenMM System object.
        """
        system: openmm.System = kwargs["system"]
        positions: unit.Quantity = kwargs["positions"]
        periodic_system: bool = kwargs["periodic_system"]
        if periodic_system:
            expr = "0.5*positional_force_const*periodicdistance(x, y, z, x0, y0, z0)^2"
        else:
            expr = "0.5*positional_force_const*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        indices = self._restraint.atom_indices
        restraint_force_constant = self._restraint.force_constant
        force = openmm.CustomExternalForce(expr)
        force.addGlobalParameter("positional_force_const", restraint_force_constant)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        for index in indices:
            position = positions[index]
            force.addParticle(index, [position[0], position[1], position[2]])
            
        system.addForce(force)
        return None

@define
class Pairwise_restraint_openmm_interface:
    """
    A Pairwise restraint using the OpenMM engine. Keep pairs of
    atoms or groups at an approximately constant distance - to
    prevent unfolding, for example.
    """
    _restraint: restraints_structures.Pairwise_restraint = field()
    def add_forces_to_system(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add force(s) to an OpenMM System object.
        """
        system = kwargs["system"]
        positions: unit.Quantity = kwargs["positions"]
        expr = "0.5*pairwise_force_const*(distance(g1, g2) - my_dist)^2"
        #pairs = self._restraint.atom_index_pairs
        atom_indices = self._restraint.atom_indices
        pairwise_force_spring_constant = self._restraint.force_constant
        pairwiseForce= openmm.CustomCentroidBondForce(2, expr)
        pairwiseForce.setForceGroup(0)
        pairwiseForce.addPerBondParameter("pairwise_force_const")
        pairwiseForce.addPerBondParameter("my_dist")
        pairs = []
        for atom_index1 in atom_indices:
            for atom_index2 in atom_indices:
                if atom_index1 < atom_index2:
                    pairs.append([atom_index1, atom_index2])
        
        for pair in pairs:
            index1 = pair[0]
            index2 = pair[1]
            position1 = positions[index1]
            position2 = positions[index2]
            startDistance = unit.norm(position1 - position2)
            pairwiseGroup1 = pairwiseForce.addGroup([index1])
            pairwiseGroup2 = pairwiseForce.addGroup([index2])
            pairwiseForce.addBond([pairwiseGroup1, pairwiseGroup2], 
                                [pairwise_force_spring_constant, startDistance])
        
        system.addForce(pairwiseForce)
        return None

@define
class Proximity_restraint_openmm_interface:
    """
    A Proximity restraint using the OpenMM engine. Keep two groups of
    atoms within a certain distance - to prevent the ligand from escaping too
    far from the receptor, for example.
    """
    _restraint: restraints_structures.Proximity_restraint = field()
    def add_forces_to_system(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add force(s) to an OpenMM System object.
        """
        system = kwargs["system"]
        expr = "0.5*proximity_force_const*step(distance(g1, g2) - radius)*(distance(g1, g2) - radius)^2"
        group1_indices = self._restraint.atom_indices1
        group2_indices = self._restraint.atom_indices2
        proximity_force_spring_constant = self._restraint.force_constant
        proximity_radius = self._restraint.radius
        proximityForce= openmm.CustomCentroidBondForce(2, expr)
        proximityGroup1 = proximityForce.addGroup(group1_indices)
        proximityGroup2 = proximityForce.addGroup(group2_indices)
        proximityForce.setForceGroup(0)
        proximityForce.addPerBondParameter("proximity_force_const")
        proximityForce.addPerBondParameter("radius")
        proximityForce.addBond([proximityGroup1, proximityGroup2], 
                            [proximity_force_spring_constant, proximity_radius])
        system.addForce(proximityForce)
        return None

Restraint_openmm_interface = typing.Union[
    Positional_restraint_openmm_interface,
    Pairwise_restraint_openmm_interface,
    Proximity_restraint_openmm_interface
]

def assign_restraints_interface(
        restraint: restraints_structures.Restraint, 
        md_engine: structures.MD_engine
        ) -> Restraint_openmm_interface:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if restraint.type == "positional":
        return Positional_restraint_openmm_interface(restraint)
    elif restraint.type == "pairwise":
        return Pairwise_restraint_openmm_interface(restraint)
    elif restraint.type == "proximity":
        return Proximity_restraint_openmm_interface(restraint)
    else:
        return None