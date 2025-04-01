"""
modules/sampling_methods/metaD.py

Functions to implement the metadynamics sampling method
within available engines.
"""

import os

import openmm.app as openmm_app

# TODO: these are OpenMM-specific functions, so they should be moved to
#  the interface module.
# TODO: add typing to arguments and return values
def make_metaD_force_bias(
        cv, 
        cv_openmm_interface,
        min_value, 
        max_value, 
        bias_width, 
        grid_width):
    """
    Take a Collective_variable object return an OpenMM BiasVariable() 
    object that the plugin can use to run a metadynamics sim.
    
    Parameters
    ----------
    cv : Collective_variable()
        A Collective_variable object which contains all the information
        for the collective variable describing this variable. In fact,
        the boundaries are contours of the function described by cv.
        This variable contains information like the groups of atoms
        involved with the CV, and the expression which describes the
        function.
        
    Returns
    -------
    my_bias_variable : openmm.BiasVariable()
        An OpenMM BiasVariable object used to propagate a metadynamics sim.
    """
    # TODO: implement openmm-CV interface.
    my_metaD_force = cv_openmm_interface.make_cv_force()
    # NOTE: functions to add groups or add variables to force object?
    
    # TODO: Check cv type to see whether periodic should be accepted
        
    myforce_bias = openmm_app.BiasVariable(
        my_metaD_force, minValue=min_value, maxValue=max_value,
        biasWidth=bias_width, periodic=cv.periodic, gridWidth=grid_width)
    
    return myforce_bias

def add_metaD_cvs(model, bias_widths, grid_widths, ignore_cv=None):
    """
    
    """
    value_dict = {}
    if ignore_cv is None:
        ignore_cv = []
        
    metaD_force_biases = []
    for i, variable_key in enumerate(model.anchors[0].variables):
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        
        variable_values = []
        cv = model.collective_variables[var_cv]
        for anchor in model.anchors:
            value = anchor.variables[variable_key]
            variable_values.append(value)
        
        #min_value = min(variable_values)
        if isinstance(cv, mmvt_spherical_cv.MMVT_spherical_CV):
            min_value = 0.0
            max_value = max(variable_values)
        else:
            min_value = min(variable_values)
            max_value = max(variable_values)
        
        curdir = os.getcwd()
        os.chdir(model.anchor_rootdir)
        
        if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
            var_child_cv = int(variable_key.split("_")[2])
            if var_child_cv in ignore_cv:
                continue
            child_cv = cv.child_cvs[var_child_cv]
            myforce_bias = make_metaD_force_bias(
                child_cv, min_value, max_value, bias_widths[i], grid_widths[i])
        else:
            if var_cv in ignore_cv:
                continue
            myforce_bias = make_metaD_force_bias(
                cv, min_value, max_value, bias_widths[i], grid_widths[i])
                
        os.chdir(curdir)
        metaD_force_biases.append(myforce_bias)
    
    return metaD_force_biases

def add_metaD_cvs_cartesian(model, bias_widths, grid_widths, min_x, max_x, 
                              min_y, max_y, min_z, max_z, ligand_indices,
                              receptor_indices):
    """
    
    """
    print("pushing these indices out:", ligand_indices)
    print("min_x:", min_x, "max_x:", max_x)
    print("min_y:", min_y, "max_y:", max_y)
    print("min_z:", min_z, "max_z:", max_z)
    curdir = os.getcwd()
    os.chdir(model.anchor_rootdir)
    assert len(ligand_indices) > 0, "No ligand atoms could be found."
    x_force = openmm.CustomCentroidBondForce(2, "x1 - x2")
    mygroup_x_lig = x_force.addGroup(ligand_indices)
    mygroup_x_rec = x_force.addGroup(receptor_indices)
    x_force.addBond([mygroup_x_lig, mygroup_x_rec], [])
    y_force = openmm.CustomCentroidBondForce(2, "y1 - y2")
    mygroup_y_lig = y_force.addGroup(ligand_indices)
    mygroup_y_rec = y_force.addGroup(receptor_indices)
    y_force.addBond([mygroup_y_lig, mygroup_y_rec], [])
    z_force = openmm.CustomCentroidBondForce(2, "z1 - z2")
    mygroup_z_lig = z_force.addGroup(ligand_indices)
    mygroup_z_rec = z_force.addGroup(receptor_indices)
    z_force.addBond([mygroup_z_lig, mygroup_z_rec], [])
    
    myforce_x_bias = openmm_app.BiasVariable(
        x_force, minValue=min_x, maxValue=max_x,
        biasWidth=bias_widths[0], periodic=False, gridWidth=grid_widths[0])
    myforce_y_bias = openmm_app.BiasVariable(
        y_force, minValue=min_y, maxValue=max_y,
        biasWidth=bias_widths[1], periodic=False, gridWidth=grid_widths[1])
    myforce_z_bias = openmm_app.BiasVariable(
        z_force, minValue=min_z, maxValue=max_z,
        biasWidth=bias_widths[2], periodic=False, gridWidth=grid_widths[2])
    
    meta_force_biases = [myforce_x_bias, myforce_y_bias, myforce_z_bias]
    os.chdir(curdir)
    return meta_force_biases

def save_metaD_3d_info_file(model, metadyn_npoints, min_x, max_x, min_y, max_y, 
        min_z, max_z, com):
    info_filename = os.path.join(model.anchor_rootdir, "metaD_3d_info.txt")
    min_x += com[0]
    max_x += com[0]
    min_y += com[1]
    max_y += com[1]
    min_z += com[2]
    max_z += com[2]
    with open(info_filename, "w") as f:
        f.write(f"n_x: {metadyn_npoints[0]}\n")
        f.write(f"n_y: {metadyn_npoints[1]}\n")
        f.write(f"n_z: {metadyn_npoints[2]}\n")
        f.write(f"min_x: {min_x}\n")
        f.write(f"max_x: {max_x}\n")
        f.write(f"min_y: {min_y}\n")
        f.write(f"max_y: {max_y}\n")
        f.write(f"min_z: {min_z}\n")
        f.write(f"max_z: {max_z}\n")
        
    return
