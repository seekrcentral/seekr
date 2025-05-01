"""
modules/collective_variables/openmm_interface.py

The OpenMM interface for the collective variables.
"""

import typing

from attrs import define, field
import openmm
import openmm.unit as unit

import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.external_cv as external_cv
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.structures as structures

@define
class Distance_cv_openmm_interface:
    collective_variable: distance_cv.Distance_CV = field()
    group1_atom_indices: typing.List[int] = field()
    group2_atom_indices: typing.List[int] = field()
    NUM_GROUPS: int = field(default=2)
    CV_EXPRESSION: str = field(default="distance(g1, g2)")
    BOUNDARY_EXPRESSION: str = field(default="step(k*(distance(g1, g2) - value))")
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

    def add_cv_groups_and_variables(self) -> None:
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

    def make_voronoi_cv_boundary_forces(
            self,
            me_val: float,
            neighbor_val: float
            ) -> tuple[openmm.CustomCentroidBondForce, openmm.CustomCentroidBondForce]:
        """
        Make the force objects to track proximity to current and adjacent anchors.
        """
        me_expr = "(me_val - {})^2".format(self.CV_EXPRESSION)
        me_force = openmm.CustomCentroidBondForce(self.NUM_GROUPS, me_expr)
        megroup1 = me_force.addGroup(self.group1_atom_indices)
        megroup2 = me_force.addGroup(self.group2_atom_indices)
        me_force.addPerBondParameter("me_val")
        me_force.addBond([megroup1, megroup2], [me_val])

        neighbor_expr = "(neighbor_val - {})^2".format(self.CV_EXPRESSION)
        neighbor_force = openmm.CustomCentroidBondForce(self.NUM_GROUPS, neighbor_expr)
        neighborgroup1 = neighbor_force.addGroup(self.group1_atom_indices)
        neighborgroup2 = neighbor_force.addGroup(self.group2_atom_indices)
        neighbor_force.addPerBondParameter("neighbor_val")
        neighbor_force.addBond([neighborgroup1, neighborgroup2], [neighbor_val])
        return me_force, neighbor_force

    def make_boundary_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM boundary force object for this CV.
        """
        self._boundary_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.BOUNDARY_EXPRESSION)
        return self._boundary_force

    def add_boundary_groups_and_variables(
            self,
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
        #force.addPerBondParameter("bitcode")
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        variable_values_list = [k, value]
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

@define
class External_cv_openmm_interface:
    collective_variable: external_cv.External_CV = field()
    groups_atom_indices: typing.List[typing.List[int]] = field()
    _cv_force = field(default=None)
    _boundary_force = field(default=None)
    _restraining_force = field(default=None)

    def make_cv_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        num_groups = len(self.groups_atom_indices)
        cv_expression = self.collective_variable.expression
        self._cv_force = openmm.CustomCentroidBondForce(
            num_groups, cv_expression)
        return self._cv_force

    def add_cv_groups_and_variables(self) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        assert self._cv_force is not None
        force = self._cv_force
        group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            group = force.addGroup(group_atom_indices)
            group_list.append(group)
            
        force.addBond(group_list, [])
        return

    def get_cv_value(self, context: openmm.Context) -> unit.Quantity:
        """
        Get the value of the CV from the OpenMM context.
        """
        return self._cv_force.getCollectiveVariableValues(context)

    def make_voronoi_cv_boundary_forces(
            self,
            me_val: float,
            neighbor_val: float
            ) -> tuple[openmm.CustomCentroidBondForce, openmm.CustomCentroidBondForce]:
        """
        Make the force objects to track proximity to current and adjacent anchors.
        """
        num_groups = len(self.groups_atom_indices)
        cv_expression = self.collective_variable.expression
        me_expr = "(me_val - {})^2".format(cv_expression)
        me_force = openmm.CustomCentroidBondForce(num_groups, me_expr)
        me_group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            me_group = me_force.addGroup(group_atom_indices)
            me_group_list.append(me_group)
        me_force.addPerBondParameter("me_val")
        me_force.addBond(me_group_list, [me_val])

        neighbor_expr = "(neighbor_val - {})^2".format(cv_expression)
        neighbor_force = openmm.CustomCentroidBondForce(num_groups, neighbor_expr)
        neighbor_group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            neighbor_group = neighbor_force.addGroup(group_atom_indices)
            neighbor_group_list.append(neighbor_group)
        neighbor_force.addPerBondParameter("neighbor_val")
        neighbor_force.addBond(neighbor_group_list, [neighbor_val])
        return me_force, neighbor_force
    
    def make_boundary_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        num_groups = len(self.groups_atom_indices)
        boundary_expression = "step(k*("+self.collective_variable.expression+"-value))"
        self._cv_force = openmm.CustomCentroidBondForce(
            num_groups, boundary_expression)
        print("Boundary expression: ", boundary_expression)
        return self._cv_force

    def add_boundary_groups_and_variables(
            self, 
            k: unit.Quantity, 
            value: unit.Quantity
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        assert self._cv_force is not None
        force = self._cv_force
        group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            group = force.addGroup(group_atom_indices)
            group_list.append(group)
        force.addPerBondParameter("k")
        force.addPerBondParameter("value")
        print("k: ", k, " value: ", value)
        variable_values_list = [k, value]
        force.addBond(group_list, variable_values_list)
        return

CV_openmm_interface = typing.Union[Distance_cv_openmm_interface,
                                   External_cv_openmm_interface]

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
    elif collective_variable.type == "external":
        groups_atom_indices = []
        for group_name in collective_variable.group_names:
            groups_atom_indices.append(md_engine.get_atom_indices_by_selection_name(
                group_name))
        return External_cv_openmm_interface(
            collective_variable=collective_variable,
            groups_atom_indices=groups_atom_indices)

    else:
        return None