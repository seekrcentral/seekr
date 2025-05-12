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
    #_boundary_force = field(default=None)
    restraining_force = field(default=None)
    megroup1 = field(default=0)
    megroup2 = field(default=1)

    def get_num_groups(self) -> int:
        """
        Get the number of groups in the CV.
        """
        return self.NUM_GROUPS

    def make_cv_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        self._cv_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.CV_EXPRESSION)
        return self._cv_force

    def add_cv_groups_and_variables(
            self,
            force: openmm.CustomCentroidBondForce
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        group1 = force.addGroup(self.group1_atom_indices)
        group2 = force.addGroup(self.group2_atom_indices)
        group_list = [group1, group2]
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

    def make_boundary_force_expr(
            self,
            index: int
            ) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        boundary_expression = f"step(k_{index}*("+self.CV_EXPRESSION+f"-value_{index}))"
        return boundary_expression

    def make_boundary_force(
            self,
            full_expr: str
            ) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM boundary force object for this CV.
        """
        num_groups = self.get_num_groups()
        force = openmm.CustomCentroidBondForce(num_groups, full_expr)
        return force

    def add_boundary_groups_and_variables(
            self, 
            force: openmm.CustomCentroidBondForce,
            boundary_parameters: dict[str, unit.Quantity]
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        group_list = []
        group1 = force.addGroup(self.group1_atom_indices)
        group_list.append(group1)
        group2 = force.addGroup(self.group2_atom_indices)
        group_list.append(group2)
        variable_values_list = []
        for key, value in boundary_parameters.items():
            force.addPerBondParameter(key)
            variable_values_list.append(value)
        force.addBond(group_list, variable_values_list)
        return

    def make_restraining_force(self):
        """
        Create the OpenMM restraining force object for this CV.
        """
        self.restraining_force = openmm.CustomCentroidBondForce(
            self.NUM_GROUPS, self.RESTRAINING_EXPRESSION)
        return self.restraining_force
    
    def make_voronoi_cv_restraining_forces(
            self,
            #toggle: float,
            #value: float,
            ) -> openmm.CustomCentroidBondForce:
        """
        Make the force objects to track proximity to current and adjacent anchors.
        """
        expr = self.CV_EXPRESSION
        force = openmm.CustomCentroidBondForce(self.NUM_GROUPS, expr)
        self.megroup1 = force.addGroup(self.group1_atom_indices)
        self.megroup2 = force.addGroup(self.group2_atom_indices)
        force.addBond([self.megroup1, self.megroup2], [])
        self.restraining_force = force
        return force
    
    def add_restraining_groups_and_variables(
            self, 
            k: unit.Quantity, 
            value: unit.Quantity
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used as a restraint.
        """
        assert self.restraining_force is not None
        force = self.restraining_force
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
    restraining_force = field(default=None)

    def get_num_groups(self) -> int:
        """
        Get the number of groups in the CV.
        """
        num_groups = len(self.groups_atom_indices)
        return num_groups

    def make_cv_force(self) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        num_groups = self.get_num_groups()
        cv_expression = self.collective_variable.expression
        cv_force = openmm.CustomCentroidBondForce(
            num_groups, cv_expression)
        self._cv_force = cv_force
        return cv_force

    def add_cv_groups_and_variables(
            self,
            force: openmm.CustomCentroidBondForce
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
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
        num_groups = self.get_num_groups()
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
    
    def make_boundary_force_expr(
            self,
            index: int
            ) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM force object for this CV.
        """
        boundary_expression = f"step(k_{index}*("+self.collective_variable.expression+f"-value_{index}))"
        return boundary_expression

    def make_boundary_force(
            self,
            full_expr: str
            ) -> openmm.CustomCentroidBondForce:
        """
        Create the OpenMM boundary force object for this CV.
        """
        num_groups = self.get_num_groups()
        force = openmm.CustomCentroidBondForce(num_groups, full_expr)
        return force

    def add_boundary_groups_and_variables(
            self, 
            force: openmm.CustomCentroidBondForce,
            boundary_parameters: dict[str, unit.Quantity]
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used just as a CV.
        """
        group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            group = force.addGroup(group_atom_indices)
            group_list.append(group)
        variable_values_list = []
        for key, value in boundary_parameters.items():
            print("adding parameter", key, value)
            force.addPerBondParameter(key)
            variable_values_list.append(value)
        force.addBond(group_list, variable_values_list)
        return
    
    def make_restraining_force(self):
        """
        Create the OpenMM restraining force object for this CV.
        """
        num_groups = len(self.groups_atom_indices)
        cv_expression = self.collective_variable.expression
        self.restraining_force = openmm.CustomCentroidBondForce(
            num_groups, cv_expression)
        return self.restraining_force
    
    def make_voronoi_cv_restraining_forces(
            self,
            ) -> openmm.CustomCentroidBondForce:
        """
        Make the force objects to track proximity to current and adjacent anchors.
        """
        num_groups = len(self.groups_atom_indices)
        cv_expression = self.collective_variable.expression
        force = openmm.CustomCentroidBondForce(num_groups, cv_expression)
        group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            group = force.addGroup(group_atom_indices)
            group_list.append(group)
            
        force.addBond(group_list, [])
        self.restraining_force = force
        return force
    
    def add_restraining_groups_and_variables(
            self, 
            k: unit.Quantity, 
            value: unit.Quantity
            ) -> None:
        """
        Add the groups and variables to the OpenMM force object
        for when this is being used as a restraint.
        """
        assert self.restraining_force is not None
        force = self.restraining_force
        assert self._cv_force is not None
        force = self._cv_force
        group_list = []
        for i, group_atom_indices in enumerate(self.groups_atom_indices):
            group = force.addGroup(group_atom_indices)
            group_list.append(group)
            
        force.addBond(group_list, [])
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