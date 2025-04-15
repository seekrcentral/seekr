"""
modules/collective_variables/openmm_interface.py

The OpenMM interface for the collective variables.
"""

import typing

from attrs import define, field
import openmm
import openmm.unit as unit

import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.structures as structures

@define
class Distance_cv_openmm_interface:
    collective_variable: distance_cv.Distance_CV = field()
    group1_atom_indices: typing.List[int] = field()
    group2_atom_indices: typing.List[int] = field()
    NUM_GROUPS: int = field(default=2)
    CV_EXPRESSION: str = field(default="distance(g1, g2)")
    BOUNDARY_EXPRESSION: str = field(default="step(k(distance(g1, g2) - value))")
    RESTRAINING_EXPRESSION: str = field(default="0.5*k*(distance(g1, g2) - value)^2")
    _cv_force = field(default=None)
    _boundary_force = field(default=None)
    _restraining_force = field(default=None)

    def make_cv_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        self._cv_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.CV_EXPRESSION)
        return self._cv_force

    def add_cv_groups_and_variables(
            self, 
            #force: openmm.CustomCentroidBondForce,
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        assert self._cv_force is not None
        force = self._cv_force
        group_list = []
        group1 = force.addGroup(self.group1_atom_indices)
        group_list.append(group1)
        group2 = force.addGroup(self.group2_atom_indices)
        group_list.append(group2)
        force.addBond(group_list, [])
        return

    def get_cv_value(self, context: openmm.Context) -> unit.Quantity:
        """
        Get the value of the CV from the OpenMM context.
        """
        return self._cv_force.getCollectiveVariableValues(context)

    def make_boundary_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM boundary force object for this CV.
        """
        self._boundary_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.BOUNDARY_EXPRESSION)
        return self._boundary_force

    def add_boundary_groups_and_variables(
            self, 
            #force: openmm.CustomCentroidBondForce, 
            bitcode: int, 
            k: unit.Quantity, 
            value: unit.Quantity
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used to define a boundary element.
        """
        assert self._boundary_force is not None
        force = self._boundary_force
        group_list = []
        group1 = force.addGroup(self.group1_atom_indices)
        group_list.append(group1)
        group2 = force.addGroup(self.group2_atom_indices)
        group_list.append(group2)
        force.addPerBondParameter("bitcode")
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        variable_values_list = [bitcode, k, value]
        force.addBond(group_list, variable_values_list)
        return

    def make_restraining_force(self):
        """
        Create the OpenMM restraining force object for this CV.
        """
        self._restraining_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.RESTRAINING_EXPRESSION)
        return self._restraining_force
    
    def add_restraining_groups_and_variables(
            self, 
            #force: openmm.CustomCentroidBondForce, 
            k: unit.Quantity, 
            value: unit.Quantity
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used as a restraint.
        """
        assert self._restraining_force is not None
        force = self._restraining_force
        group_list = []
        group1 = force.addGroup(self.group1_atom_indices)
        group_list.append(group1)
        group2 = force.addGroup(self.group2_atom_indices)
        group_list.append(group2)
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        variable_values_list = [k, value]
        force.addBond(group_list, variable_values_list)
        return

CV_openmm_interface = typing.Union[Distance_cv_openmm_interface]

def assign_cv_interface(
        collective_variable: cv_structures.Collective_variable,
        md_engine: structures.MD_engine) -> CV_openmm_interface:
    """
    Given a collective variable object, assign the appropriate OpenMM interface.
    """
    if collective_variable.type == "distance":
        group1_atom_indices = md_engine.get_atom_indices_by_selection_name(
            collective_variable.group1_name)
        group2_atom_indices = md_engine.get_atom_indices_by_selection_name(
            collective_variable.group2_name)
        return Distance_cv_openmm_interface(
            collective_variable=collective_variable, 
            group1_atom_indices=group1_atom_indices,
            group2_atom_indices=group2_atom_indices)
        
    else:
        return None