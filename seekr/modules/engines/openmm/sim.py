
# A bunch of distance CV functions.

# TODO: move to OpenMM engine module for cvs?
    def make_boundary_force(self, alias_id):
        """
        Create an OpenMM force object which will be used to compute
        the value of the CV's mathematical expression as the simulation
        propagates. Each *milestone* in a cell, not each CV, will have
        a force object created.
        
        In this implementation of MMVT in OpenMM, CustomForce objects
        are used to provide the boundary definitions of the MMVT cell.
        These 'forces' are designed to never affect atomic motions,
        but are merely monitored by the plugin for bouncing event.
        So while they are technically OpenMM force objects, we don't
        refer to them as forces outside of this layer of the code,
        preferring instead the term: boundary definitions.
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm
            
        assert self.num_groups == 2
        expression_w_bitcode = "bitcode*"+self.openmm_expression
        return openmm.CustomCentroidBondForce(
            self.num_groups, expression_w_bitcode)
    
    # TODO: move to OpenMM engine module for cvs?
    def make_restraining_force(self, alias_id):
        """
        Create an OpenMM force object that will restrain the system to
        a given value of this CV.
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm
            
        assert self.num_groups == 2
        return openmm.CustomCentroidBondForce(
            self.num_groups, self.restraining_expression)
    
    # TODO: move to OpenMM engine module for cvs?
    def make_cv_force(self, alias_id):
        """
        Make a force that is composed only of the cv_expression.
        Used for metadynamics and potentially other things.
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm
            
        assert self.num_groups == 2
        return openmm.CustomCentroidBondForce(
            self.num_groups, self.cv_expression)
    
    # TODO: move to OpenMM engine module for cvs?
    def make_voronoi_cv_boundary_forces(self, me_val, neighbor_val, alias_id):
        """
        
        """
        try:
            import openmm
        except ImportError:
            import simtk.openmm as openmm
            
        me_expr = "(me_val - {})^2".format(self.cv_expression)
        me_force = openmm.CustomCentroidBondForce(
            self.num_groups, me_expr)
        megroup1 = me_force.addGroup(self.group1)
        megroup2 = me_force.addGroup(self.group2)
        me_force.addPerBondParameter("me_val")
        me_force.addBond([megroup1, megroup2], [me_val])
        
        neighbor_expr = "(neighbor_val - {})^2".format(self.cv_expression)
        neighbor_force = openmm.CustomCentroidBondForce(
            self.num_groups, neighbor_expr)
        neighborgroup1 = neighbor_force.addGroup(self.group1)
        neighborgroup2 = neighbor_force.addGroup(self.group2)
        neighbor_force.addPerBondParameter("neighbor_val")
        neighbor_force.addBond([neighborgroup1, neighborgroup2], [neighbor_val])
        
        return me_force, neighbor_force
    
    # TODO: move to OpenMM engine module for cvs?
    def update_voronoi_cv_boundary_forces(
            self, me_force, me_val, neighbor_force, neighbor_val, alias_id, 
            context):
        """
        
        """
        # the group list values of [0,1] is hacky, but probably correct
        me_force.setBondParameters(0, [0,1], [me_val])
        neighbor_force.setBondParameters(0, [0,1], [neighbor_val])
        return
    
    # TODO: move to OpenMM engine module for cvs?
    def add_groups(self, force):
        """
        
        """
        self._mygroup_list = []
        mygroup1 = force.addGroup(self.group1)
        self._mygroup_list.append(mygroup1)
        mygroup2 = force.addGroup(self.group2)
        self._mygroup_list.append(mygroup2)
        return
    
    # TODO: move to OpenMM engine module for cvs?
    def add_parameters(self, force):
        """
        An OpenMM custom force object needs a list of variables
        provided to it that will occur within its expression. Both
        the per-dof and global variables are combined within the
        variable_names_list. The numerical values of these variables
        will be provided at a later step.
        """
        self.add_groups(force)
        variable_names_list = []
        variable_names_list.append("bitcode")
        force.addPerBondParameter("bitcode")
        if self.per_dof_variables is not None:
            for per_dof_variable in self.per_dof_variables:
                force.addPerBondParameter(per_dof_variable)
                variable_names_list.append(per_dof_variable)
        
        if self.global_variables is not None:
            for global_variable in self.global_variables:
                force.addGlobalParameter(global_variable)
                variable_names_list.append(global_variable)
        
        return variable_names_list
    
    # TODO: move to OpenMM engine module for cvs?
    def add_groups_and_variables(self, force, variables, alias_id):
        """
        Provide the custom force with additional information it needs,
        which includes a list of the groups of atoms involved with the
        CV, as well as a list of the variables' *values*.
        """
        assert len(self._mygroup_list) == self.num_groups
        # TODO: add this assert to other CVs
        assert force.getNumBonds() == 0, \
            "add_groups_and_variables cannot be called twice for the same "\
            "CV. Use update_groups_and_variables."
        force.addBond(self._mygroup_list, variables)
        return
    
    # TODO: move to OpenMM engine module for cvs?
    def update_groups_and_variables(self, force, variables, alias_id, context):
        """
        Update the force's variables with a list of new values.
        
        A context must have it's reinitialize() method called after 
        this function for the changes to be applied.
        """
        force.setBondParameters(0, self._mygroup_list, variables)
        return
    
    # TODO: move to OpenMM engine module for cvs?
    def get_variable_values_list(self, milestone):
        """
        Create the list of CV variables' values in the proper order
        so they can be provided to the custom force object.
        """
        assert milestone.cv_index == self.index
        values_list = []
        bitcode = 2**(milestone.alias_index-1)
        k = milestone.variables["k"] * unit.kilojoules_per_mole/unit.angstrom**2
        radius = milestone.variables["radius"] * unit.nanometers
        values_list.append(bitcode)
        values_list.append(k)
        values_list.append(radius)
        return values_list

# This should probably be moved to OpenMM engine module
    def get_openmm_context_cv_value(self, context, positions=None, system=None):
        """
        Determine the current CV value for an openmm context.
        """
        if system is None:
            system = context.getSystem()
        if positions is None:
            assert context is not None, \
                "If positions is None, context must not be None"
            state = context.getState(getPositions=True)
            positions = state.getPositions()
        com1 = base.get_openmm_center_of_mass_com(
            system, positions, self.group1)
        com2 = base.get_openmm_center_of_mass_com(
            system, positions, self.group2)
        radius = np.linalg.norm(com2-com1)
        assert np.isfinite(radius), "Non-finite value detected."
        return radius
        
    def check_openmm_context_within_boundary(
            self, context, milestone_variables, positions=None, verbose=False,
            tolerance=0.0):
        """
        Check if an OpenMM context describes a system that remains
        within the expected anchor. Return True if passed, return
        False if failed.
        """
        radius = self.get_openmm_context_cv_value(context, positions)
        result = self.check_value_within_boundary(
            radius, milestone_variables, verbose, tolerance)
        return result