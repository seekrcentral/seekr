"""
modules/check.py

Run pre- and post-simulation checks on the SEEKR model.
"""

import os
import sys
import glob
import math
import typing
import argparse
import collections

import numpy as np
import mdtraj
import parmed

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.scales.molecular_dynamics.structures as scale_md

# TODO: implement a check for a sequential series of bounces later in 
#  the simulation. Suggest that this could be remedied by using a
#  smaller friction coefficient.

# The charges of common ions
ION_CHARGE_DICT = {"li":1.0, "na":1.0, "k":1.0, "rb":1.0, "cs":1.0, "fr":1.0,
                   "be":2.0, "mg":2.0, "ca":2.0, "sr":2.0, "ba":2.0, "ra":2.0,
                   "f":-1.0, "cl":-1.0, "br":-1.0, "i":-1.0, "at":-1.0,
                   "he":0.0, "ne":0.0, "ar":0.0, "kr":0.0, "xe":0.0, "rn":0.0}

MAX_BD_STUCK = 5
AVOGADROS_NUMBER = 6.022e23
RECURSION_LIMIT = 100000

def is_ion(atom):
    """If a lone atom has no bonds, assume it's an ion."""
    
    if len(atom.bond_partners) == 0:
        return True
    else:
        return False

def box_volume_from_6_vector(box_6_vector):
    """
    Compute the volume of a box from its 6-vector.
    """
    box_6_vector = box_6_vector[0]
    box_6_vector[0] *= 0.1
    box_6_vector[1] *= 0.1
    box_6_vector[2] *= 0.1
    
    ax = float(box_6_vector[0])
    ay = 0.0
    az = 0.0
    bx = float(box_6_vector[1] * math.cos(box_6_vector[5]*math.pi/180.0))
    by = float(box_6_vector[1] * math.sin(box_6_vector[5]*math.pi/180.0))
    bz = 0.0
    cx = (box_6_vector[2] * math.cos(box_6_vector[4]*math.pi/180.0))
    if box_6_vector[5] == 90.0:
        cy = 0.0
    else:
        cy = float(box_6_vector[2] * (
            math.cos(box_6_vector[3]*math.pi/180.0) \
            - math.cos(box_6_vector[4]*math.pi/180.0) \
            * math.cos(box_6_vector[5]*math.pi/180.0)) \
            / math.sin(box_6_vector[5]*math.pi/180.0))
    cz = float(math.sqrt(box_6_vector[2]**2 - cx**2 - cy**2))
    A = np.array([ax, ay, az])
    B = np.array([bx, by, bz])
    C = np.array([cx, cy, cz])
    volume = abs(np.dot(A, np.cross(B, C)))
    return volume

def load_structure_with_parmed(
        model: structures.Seekr_model, 
        anchor_or_unpartitioned: structures.Anchor \
            | engines_structures.MD_unpartitioned_settings,
        stage: scale_md.Stage_md,
        swarm_id: int = 0
        ) -> parmed.Structure | None:
    """
    Given the simulation inputs, load an anchor's structure for one of
    the checks and return the parmed structure object.
    """
    input_positions_filenames = anchor_or_unpartitioned.get_positions_full_paths(
        model.directory, stage.input_stage_index, swarm_id)
    if len(input_positions_filenames) == 0:
        return None
    assert len(input_positions_filenames) == 1, \
        "Only one input structure is allowed per anchor."
    input_positions_filename = input_positions_filenames[0]
    structure = parmed.load_file(input_positions_filename)
    return structure

def load_structure_with_mdtraj(
        model: structures.Seekr_model,
        anchor_or_unpartitioned: structures.Anchor \
            | engines_structures.MD_unpartitioned_settings,
        stage: scale_md.Stage_md,
        swarm_id: int = 0,
        mode: str = "pdb"
        ) -> mdtraj.Trajectory | None:
    """
    Given the simulation inputs, load an anchor's structure for one of
    the checks and return the mdtraj Trajectory() object.
    """
    full_directory = os.path.join(
        model.directory, anchor_or_unpartitioned.directory,
        anchor_or_unpartitioned.stage_parameters[stage.index]["directory"])
    input_positions_filenames \
        = anchor_or_unpartitioned.get_positions_full_paths(
            model.directory, stage.input_stage_index, swarm_id)
    if len(input_positions_filenames) == 0:
        return None
    assert len(input_positions_filenames) == 1, \
        "Only one input structure is allowed per anchor."
    input_positions_filename = input_positions_filenames[0]
    if mode == "pdb":
        pass
    
    elif mode == "traj":
        traj_glob = os.path.join(full_directory, f"*.swarm_{swarm_id}.dcd")
        traj_filenames = glob.glob(traj_glob)
        if len(traj_filenames) == 0:
            return None
        
        # search for and remove any empty trajectory files
        indices_to_pop = []
        for i, mmvt_traj_filename in enumerate(traj_filenames):
            if os.path.getsize(mmvt_traj_filename) == 0:
                indices_to_pop.append(i)
        
        for i in indices_to_pop[::-1]:
            traj_filenames.pop(i)
    else:
        raise Exception("Check mode not implemented: {}".format(mode))
        
    if model.using_toy():
        pdb_filename = os.path.join(full_directory, "toy.pdb")
        if mode == "pdb":
            traj = mdtraj.load(pdb_filename)
            traj.xyz = np.loadtxt(input_positions_filename, ndmin=2)
            
        elif mode == "traj":
            if not len(traj_filenames) > 0:
                # TODO: log this?
                #warnings.warn("Empty mmvt trajectories were found in "\
                #      "anchor: {}.".format(anchor.index))
                return None
                
            traj = mdtraj.load(traj_filenames, top=pdb_filename)
            
        return traj
    
    # TODO: load the model's md_parameters_topology
    else:
        if mode == "pdb":
            traj = mdtraj.load(input_positions_filename)
        elif mode == "traj":
            if not len(traj_filenames) > 0:
                # TODO: log this?
                #warnings.warn("Empty mmvt trajectories were found in "\
                #      "anchor: {}.".format(anchor.index))
                return None
            traj = mdtraj.load(traj_filenames, top=input_positions_filename)
            raise Exception("TODO: must handle problem where MDtraj won't load "\
                            "correct masses for system XML. Ref code below.")
            """
            full_system_filename = os.path.join(
                building_directory, anchor.forcefield_params.system_filename)
            with open(full_system_filename) as input:
                system = openmm.XmlSerializer.deserialize(input.read())
            for i, atom in enumerate(traj.top.atoms):
                sys_particle_mass = system.getParticleMass(i).value_in_unit(unit.daltons)
                md_element_mass = atom.element.mass
                if not np.isclose(sys_particle_mass, 
                                  md_element_mass, atol=ELEMENT_MASS_TOLERANCE):
                    print("Mass mismatch for atom index {}: ".format(i)\
                            +"mdtraj mass is {}, ".format(md_element_mass)\
                            +"system mass is {}".format(sys_particle_mass))
                    # Choose the closest element by mass
                    md_closest_element = mdtraj.element.Element.getByMass(
                        sys_particle_mass)
                    mass_diff = abs(md_closest_element.mass - sys_particle_mass)
                    if mass_diff < ELEMENT_MASS_TOLERANCE:
                        print("Changing atom index {} element from ".format(i)\
                                +f"{atom.element.symbol} to "\
                                +f"{md_closest_element.symbol}")
                        atom.element = md_closest_element
                    else:
                        # Create a new element with the correct mass
                        particle_mass = sys_particle_mass
                        print("Creating new element with symbol:", f"X{i}")
                        new_element = mdtraj.element.Element(symbol=f"X{i}", name=f"X{i}", number=200+i, radius=atom.element.radius, mass=particle_mass.value_in_unit(unit.daltons))
                        atom.element = new_element
            """
        return traj

def get_first_available_anchor_md_stage_structure(
        model: structures.Seekr_model,
        ) -> typing.Tuple[
            structures.Anchor | engines_structures.MD_unpartitioned_settings,
            scale_md.Stage_md]:
    """
    Get the first anchor and stage for a given model where a structure 
    can be loaded.
    """
    for stage_id, stage in enumerate(model.stages):
        if stage.type == "md_stage":
            # First, try unpartitioned
            md_engine = model.get_engine_by_scale_type("molecular_dynamics")
            if md_engine.unpartitioned_settings.stage_parameters[stage_id] \
                    is not None:
                structure_paths = md_engine.unpartitioned_settings\
                    .get_positions_full_paths(
                        model.directory, stage.input_stage_index, 0)
                if len(structure_paths) > 0:
                    return (md_engine.unpartitioned_settings, stage)
            # if that doesn't work, try anchor
            for anchor in model.anchors:
                if "molecular_dynamics" in anchor.scale_types:
                    structure_paths = anchor.get_positions_full_paths(
                        model.directory, stage.input_stage_index, 0)
                    if len(structure_paths) > 0:
                        return (md_engine.unpartitioned_settings, stage)
                    
    raise Exception("No available anchors or structures found in model.")

def check_pre_sim_MD_and_BD_salt_concentration(
        model: structures.Seekr_model
        ) -> bool:
    """
    Users might inadvertently define different salt concentrations 
    between the MD and BD stages. 
    Examine BD settings and count the numbers of ions in MD starting
    structures to ensure that ion concentrations are relatively 
    consistent to avoid situations where different salt concentrations
    exist between the various anchors and scales.
    """
    RELATIVE_TOLERANCE = 0.1
    ABSOLUTE_TOLERANCE = 0.01
    if model.using_scale("brownian_dynamics"):
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        bd_ionic_strength = 0.0
        for ion in bd_engine.ions:
            bd_ionic_strength += 0.5 * ion.concentration * ion.charge**2
    else:
        # No BD to check
        return True
    
    anchor_or_unpartitioned, stage \
        = get_first_available_anchor_md_stage_structure(model)
    md_ionic_strength = 0.0
    structure = load_structure_with_parmed(model, anchor_or_unpartitioned, stage)
    if structure is None:
        return True
    box_6_vector = structure.get_box()
    assert box_6_vector is not None, "Unable to compute box volume for "\
        f"structures in anchor {anchor_or_unpartitioned.index}"
    box_volume = box_volume_from_6_vector(box_6_vector)
    particle_concentration = 1.0e24 / (AVOGADROS_NUMBER * box_volume)
    for index, atom in enumerate(structure.atoms):
        if is_ion(atom):
            found_ion = False
            for key in ION_CHARGE_DICT:
                if atom.name.lower().startswith(key):
                    charge = ION_CHARGE_DICT[key]
                    md_ionic_strength += 0.5 * particle_concentration \
                        * charge**2
                    found_ion = True
                
            if not found_ion:
                print("check_pre_sim_MD_and_BD_salt_concentration - Found "\
                    f"unbonded atom with anchor {anchor_or_unpartitioned.index}, "\
                    +f"index: {index} and name: {atom.name}. "\
                    +"Charge is uncertain. Assuming its contribution to "\
                    "solvent ionic strength is zero.")
    if not np.isclose(md_ionic_strength, bd_ionic_strength, 
                    rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE):
        print("CHECK FAILURE: BD simulation has significantly different "\
        f"ionic strength of {bd_ionic_strength:.3f} M*e^2 than MD simulation "\
        f"ionic strength of {md_ionic_strength:.3f} M*e^2 for anchor "\
        f"{anchor_or_unpartitioned.index}. Please check the ion concentrations in the "\
        "BD simulation settings, and also count the number of ions "\
        "in the MD simulations. Counterions added to neutralize the protein in an "\
        "MD simulation may also trigger this failure. If that is the case, simply "\
        "skip this check.")
        return False
            
    return True

def check_systems_within_Voronoi_cells(
        model: structures.Seekr_model
        ) -> bool:
    """
    Users might provide the wrong starting structure for a given 
    anchor, and the structure may actually belong in a different one.
    When the model defines Voronoi cells (such as in MMVT), check that
    the starting structures lie within the expected Voronoi cells, and
    suggest corrections if the check fails. Otherwise, the SEEKR2
    backend would fail with a non-helpful error message.
    """
    returning_result = True
    curdir = os.getcwd()
    os.chdir(model.directory)
    
    for anchor in model.anchors:
        if "molecular_dynamics" not in anchor.scale_types:
            continue
        
        for stage_id, stage in enumerate(model.stages):
            stage_parameter = anchor.stage_parameters[stage_id]
            if not stage.type == "md_stage":
                continue

            if not stage.sampling_method.type == "mmvt":
                continue
            if stage_parameter is None:
                continue

            full_directory = os.path.join(
                model.directory, anchor.directory,
                anchor.stage_parameters[stage_id]["directory"])
            input_positions_filenames = anchor.get_positions_full_paths(
                model.directory, stage.input_stage_index, 0)
            if len(input_positions_filenames) == 0:
                continue

            full_positions_pathname = os.path.join(full_directory, 
                                                   input_positions_filenames[0])
            if not os.path.exists(full_positions_pathname):
                continue
            
            traj = load_structure_with_mdtraj(model, anchor, stage, mode="pdb")
            if traj is None:
                continue

            for milestone in anchor.milestones:
                result = cv_structures.check_mdtraj_within_boundary(
                    model, traj, milestone.variables, frame_index=0)
                
                if result == False:
                    correct_anchor = None
                    for anchor2 in model.anchors:
                        within_milestones = True
                        for milestone2 in anchor2.milestones:
                            result2 = cv_structures.check_mdtraj_within_boundary(
                                model, traj, milestone2.variables, frame_index=0)
                            if not result2:
                                within_milestones = False
                                break
                        if within_milestones:
                            correct_anchor = anchor2
                            break
                    
                    warnstr = f"CHECK FAILURE: The provided initial starting "\
                    f"structure for anchor {anchor.index} does not lie within the "\
                    "anchor boundaries. The simulation would fail. Please "\
                    "carefully check this anchor's starting structure, as "\
                    "well as the collective variable's (CV) atom selections, "\
                    "and anchor/milestone variables."
                    print(warnstr)
                    if correct_anchor is not None:
                        print("It looks like the failed structure might belong in "\
                            f"anchor {correct_anchor.index}.")
                    returning_result = False
    
    os.chdir(curdir)
    return returning_result

def recurse_atoms(
        atom: parmed.Atom, 
        _visited_indices: typing.Set[int] = set()
        ) -> typing.Set[int]:
    """
    Recursively visit all atoms within a molecule for the purposes
    of determining the molecules (all sets of atoms connected by bonds)
    in the system.
    """
    
    _visited_indices.add(atom.idx)
    for bonded_atom in atom.bond_partners:
        if not bonded_atom.idx in _visited_indices:
            branch_indices = recurse_atoms(bonded_atom, _visited_indices)
            _visited_indices.update(branch_indices)
    return _visited_indices

def check_atom_selections_on_same_molecule(
        model: structures.Seekr_model,
        ) -> bool:
    """
    The user might accidentally define atom selections that span
    multiple molecules. Check this possibility by finding all molecules
    in the system and ensure that atom selections only exist on one
    molecule.
    """
    sys.setrecursionlimit(RECURSION_LIMIT)
    warnstr1 = """CHECK FAILURE: the atom selection for collective variable 
    (CV) number {} is split over multiple molecules. Atom index {}, 
    which has the name {} and serial id {}, was the first atom to 
    be detected in a different molecule than the others. Please
    check the structure for anchor {} to ensure that atom indexing 
    is correct. Keep in mind: SEEKR2 atom indexing starts at 0 (but 
    PDB files often start with a different atom serial index, such
    as 1 or another number). This might also be caused by multimeric 
    sites or unclosed loops."""
    warnstr2 = """CHECK FAILURE: the atom index {} for collective variable
    (CV) number {} does not exist in the structure for anchor {}."""
    anchor_or_unpartitioned, stage = get_first_available_anchor_md_stage_structure(model)
    structure = load_structure_with_parmed(model, anchor_or_unpartitioned, stage)
    if structure is None:
        return True
    molecules = []
    traversed_atoms = set()
    for index, atom in enumerate(structure.atoms):
        if index in traversed_atoms:
            continue
        molecule = recurse_atoms(atom, set())
        traversed_atoms.update(molecule)
        molecules.append(molecule)
    
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    for cv in model.collective_variables:
        atom_group_names = cv.get_atom_group_names()
        for atom_group_name in atom_group_names:
            in_molecule = None
            atom_group = md_engine.get_atom_indices_by_selection_name(atom_group_name)
            for atom_index in atom_group:
                index_found = False
                for mol_id, molecule in enumerate(molecules):
                    if atom_index in molecule :
                        if in_molecule is None:
                            in_molecule = mol_id
                            index_found = True
                        elif in_molecule == mol_id:
                            index_found = True
                        else:
                            print(warnstr1.format(
                                cv.index, atom_index, 
                                structure.atoms[atom_index].name, 
                                structure.atoms[atom_index].number, 
                                anchor_or_unpartitioned.index))
                            return False
                if not index_found:
                    print(warnstr2.format(atom_index, cv.index, 
                                            anchor_or_unpartitioned.index))
                    return False
    return True

def check_atom_selections_MD_BD(
        model: structures.Seekr_model
        ) -> bool:
    """
    Users might accidentally select atoms that are not equivalent
    between the MD and BD stages. Detect whether this is the case.
    """
    warnstr1 = """CHECK FAILURE: The atom selection as defined for BD
    milestone includes atom index {} that does not exist 
    in the file {}. Please check the atom indices in Browndye
    inputs and also check the atom indices in your input PQR
    files."""
    warnstr2 = """CHECK FAILURE: Different atoms are selected between
    the MD and BD stages. The BD selection contains {} while the
    MD selection contains {}. Please check the input molecular
    structure files, as well as the input atom indices (keep in
    mind that SEEKR2 requires atom indexing to start from 0 in 
    each molecular structure input file, but PDB and PQR files
    might start their numbering at 1 or another number."""
    if model.using_scale("brownian_dynamics"):
        # First check to ensure that the selection indices are actually 
        # contained within the BD pqr files.
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        anchor_set_glob = os.path.join(
            model.directory, base.ANCHOR_SETS_DIR, "*", 
            bd_engine.b_surface_directory)
        b_surface_dir = glob.glob(anchor_set_glob)
        if len(b_surface_dir) == 0:
            raise Exception("No Browndye b_surface directories found "
                            "in anchor_sets/.")
        b_surface_dir = b_surface_dir[0]
        pqr_paths = []
        pqr_structures = []
        for bd_molecule in bd_engine.unpartitioned_settings.molecules:
            pqr_path = os.path.join(b_surface_dir, bd_molecule.pqr_filename)
            pqr_paths.append(pqr_path)
            pqr_structure = parmed.load_file(pqr_path)
            pqr_structures.append(pqr_structure)
            
        for indices in bd_molecule.original_indices.values():
            for index in indices:
                if index < 0 or index > len(pqr_structure.atoms):
                    print(warnstr1.format(index, pqr_path))
                    return False
    else:
        # No BD to check
        return True
    
    anchor_or_unpartitioned, stage \
        = get_first_available_anchor_md_stage_structure(model)
    md_structure = load_structure_with_parmed(
        model, anchor_or_unpartitioned, stage)
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    for md_selection in md_engine.unpartitioned_settings.selections:
        for bd_mol_index, bd_molecule in enumerate(
                bd_engine.unpartitioned_settings.molecules):
            for bd_site_name, bd_site_indices in bd_molecule.original_indices.items():
                if md_selection.name == bd_site_name:
                    md_atom_group = md_selection.atom_indices
                    bd_atom_group = bd_site_indices
                    if len(md_atom_group) != len(bd_atom_group):
                        bd_error_str = "{} atom(s)".format(len(bd_atom_group))
                        md_error_str = "{} atom(s)".format(len(md_atom_group))
                        print(warnstr2.format(bd_error_str, md_error_str))
                        return False
                    md_atomic_name_dict = collections.defaultdict(int)
                    bd_atomic_name_dict = collections.defaultdict(int)
                    for md_atom_idx in md_atom_group:
                        md_atom = md_structure.atoms[md_atom_idx]
                        md_atomic_name_dict[md_atom.name] += 1
                    
                    pqr_structure = pqr_structures[bd_mol_index]
                    for bd_atom_idx in bd_atom_group:
                        bd_atom = pqr_structure.atoms[bd_atom_idx]
                        bd_atomic_name_dict[bd_atom.name] += 1
                    
                    if md_atomic_name_dict != bd_atomic_name_dict:
                        bd_err_str = ""
                        for key in bd_atomic_name_dict:
                            this_str = "{} atoms with atomic name {}".format(
                                bd_atomic_name_dict[key], key)
                            if bd_err_str != "":
                                bd_err_str += " and "
                            bd_err_str += this_str
                        md_err_str = ""
                        for key in md_atomic_name_dict:
                            this_str = "{} atoms with atomic name {}".format(
                                md_atomic_name_dict[key], key)
                            if md_err_str != "":
                                md_err_str += " and "
                            md_err_str += this_str
                        print(warnstr2.format(bd_err_str, md_err_str))
                        return False
    return True

def check_pqr_residues(
        model: structures.Seekr_model
        ) -> bool:
    """
    Browndye will create lumped charges (test charges) out of entire
    residues. This check will suggest that users split their PQR
    atoms into individual residues to increase accuracy.
    """
    if model.using_scale("brownian_dynamics"):
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        anchor_set_glob = os.path.join(
            model.directory, base.ANCHOR_SETS_DIR, "*", bd_engine.b_surface_directory)
        b_surface_dir = glob.glob(anchor_set_glob)
        if len(b_surface_dir) == 0:
            raise Exception("No Browndye b_surface directories found "
                            "in anchor_sets/.")
        b_surface_dir = b_surface_dir[0]
        pqr_paths = []
        pqr_structures = []
        list_of_residue_lists = []
        for bd_molecule in bd_engine.unpartitioned_settings.molecules:
            pqr_path = os.path.join(b_surface_dir, bd_molecule.pqr_filename)
            pqr_paths.append(pqr_path)
            pqr_structure = parmed.load_file(pqr_path)
            pqr_structures.append(pqr_structure)
            residue_list = []
            for residue in pqr_structure.residues:
                if residue.name != "GHO":
                    residue_list.append(residue)

            list_of_residue_lists.append(residue_list)
        
    else:
        # No BD to check
        return True
    
    warnstr1 = """CHECK FAILURE: The PQR file {} is comprised of a single
    residue, named {}. By default, Browndye2 will lump all the charges in
    a residue into a single "test charge". This feature can be overridden by
    numbering every atom in your PQR file(s) as a different residue. If you
    don't mind this, and wish to proceed, then run this script again, skipping
    all checks."""
    for residue_list, pqr_path in zip(list_of_residue_lists, pqr_paths):
        assert len(residue_list) != 0, "Empty PQR file: {}".format(
            pqr_path)
        if len(residue_list) == 1:
            print(warnstr1.format(pqr_path, residue_list[0].name))
            return False
    
    return True

def check_mutual_neighbor_anchors(
        model: structures.Seekr_model
        ) -> bool:
    """
    Make sure that every anchor's neighbor recognizes the anchor itself
    as a neighbor
    """
    anchors_missing_neighbors = set()
    double_booked_neighbors = set()
    for anchor in model.anchors:
        for milestone in anchor.milestones:
            if not milestone.boundary:
                continue
            anchor2 = model.anchors[milestone.neighbor_anchor_index]
            # now check this anchor to ensure that it has me as a neighbor
            has_me_as_neighbor = False
            for milestone2 in anchor2.milestones:
                if not milestone2.boundary:
                    continue
                if anchor.index == milestone2.neighbor_anchor_index:
                    if has_me_as_neighbor:
                        double_booked_neighbors.add(anchor.index)
                    has_me_as_neighbor = True
                    
            if not has_me_as_neighbor:
                anchors_missing_neighbors.add(anchor.index)
    
    success = True
    if len(anchors_missing_neighbors) > 0:
        warnstr = """CHECK FAILURE: Anchors {} were found to have mismatched
        neighbor anchors. This is likely a bug. Please contact the 
        developers."""
        print(warnstr.format(anchors_missing_neighbors))
        success = False
        
    if len(double_booked_neighbors) > 0:
        warnstr = """CHECK FAILURE: Anchors {} were found to have double-
        booked neighbor anchors. This is likely a bug. Please contact the 
        developers."""
        print(warnstr.format(double_booked_neighbors))
        success = False
        
    return success

def check_mmvt_in_Voronoi_cell(
        model: structures.Seekr_model
        ) -> bool:
    """
    For some reason, a MMVT system may drift out of a Voronoi cell.
    This would most likely indicate a bug, so detect this situation.
    """
    
    for stage in model.stages:
        if stage.sampling_method.type != "mmvt":
            continue
        if model.using_scale("molecular_dynamics") is not None:
            md_engine = model.get_engine_by_scale_type("molecular_dynamics")
            if md_engine.hydrogen_mass is not None:
                return True
        for anchor in model.anchors:
            traj = load_structure_with_mdtraj(model, anchor, stage, mode="traj")
            if traj is None:
                continue
            for frame_index in range(traj.n_frames):
                for milestone in anchor.milestones:
                    result = cv_structures.check_mdtraj_within_boundary(
                        model, traj, milestone.variables, frame_index=frame_index,
                        tolerance=0.001)
                    if result == False:
                        warnstr = "CHECK FAILURE: The MMVT trajectory(ies) "\
                        f"for anchor {anchor.index} do not lie within the "\
                        "anchor boundaries. This could be caused by incorrect "\
                        "modification of files or the model.xml file, or possibly a bug"
                        print(warnstr)
                        return False
    
    return True

def check_for_stuck_BD(
        model: structures.Seekr_model
        ) -> bool:
    """
    Examine BD output to ensure that too many BD trajectories weren't
    stuck.
    """
    if model.using_scale("brownian_dynamics"):
        bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
        bd_transition_counts = bd_engine.get_bd_transition_counts(model.directory)
        if "stuck" in bd_transition_counts:
            if bd_transition_counts["stuck"] > MAX_BD_STUCK:
                warnstr = """CHECK FAILURE: A large number of BD
                trajectories stuck. {} stuck BD trajectories found."""
                print(warnstr.format(bd_transition_counts["stuck"]))
                return False
            else:
                return True
    return True

def check_pre_simulation_all(model):
    """
    After the completion of the prepare stage, check inputs for some
    of the most common problems and mistakes a user is likely to 
    encounter. If a check fails, raise an Exception.
    
    Parameters:
    -----------
    model : Model()
        The SEEKR2 model object containing all calculation information.
        
    Returns:
    None
    """
    curdir = os.getcwd()
    check_passed_list = []
    #check_passed_list.append(check_pre_sim_bubbles(model))
    if not model.using_toy():
        check_passed_list.append(check_pre_sim_MD_and_BD_salt_concentration(model))
        check_passed_list.append(check_atom_selections_on_same_molecule(model))
        check_passed_list.append(check_atom_selections_MD_BD(model))
        check_passed_list.append(check_pqr_residues(model))
    else:
        # Checks for toy systems
        pass

    check_passed_list.append(check_systems_within_Voronoi_cells(model))
    # NOTE: removing this check because I want to implement multiple bulk anchors
    #  in the prepare and run stages (will be lumped into a single bulk for analyze).
    #check_passed_list.append(check_for_one_bulk_anchor(model))
    check_passed_list.append(check_mutual_neighbor_anchors(model))

    no_failures = True
    for check_passed in check_passed_list:
        if not check_passed:
            no_failures = False
    
    os.chdir(curdir)
    if no_failures:
        print("All pre-simulation checks passed.")
        return
    else:
        check_fail_str = "One or more fatal pre-simulation checks failed. It "\
        "is highly recommended that you address and correct each of these "\
        "problems. However, you can force SEEKR to skip these checks by using "\
        "the --skip_checks (-s) argument on prepare.py. If these checks were "\
        "performed using prepare.py, the model was saved, and though it is "\
        "not recommended, you may choose to proceed with the calculation "\
        "regardless of these failed checks."
        print(check_fail_str)
        raise Exception("The SEEKR2 calculation should not proceed due "\
                        "to failed pre-simulation checks.")
    

def check_post_simulation_all(
        model: structures.Seekr_model,
        ) -> None:
    """
    After the completion of the run stage, check simulation files
    for some of the most common problems and mistakes a user is likely 
    to encounter. If a check fails, raise an Exception.
    
    Parameters:
    -----------
    model : Seekr_model()
        The seekr model object containing all calculation information.
    
    long_check : bool, Default False
        Whether to conduct a detailed check of post-simulation outputs.
        If set to True, the checks are likely to take a significantly
        longer amount of time, but is more likely to detect problems.
    
    Returns:
    None
    """
    
    curdir = os.getcwd()
    check_passed_list = []
    if model.using_toy():
        pass
    else:
        check_passed_list.append(check_mmvt_in_Voronoi_cell(model))
    
    check_passed_list.append(check_for_stuck_BD(model))
    
    no_failures = True
    for check_passed in check_passed_list:
        if not check_passed:
            no_failures = False
    
    os.chdir(curdir)
    if no_failures:
        print("All post-simulation checks passed.")
        return
    else:
        check_fail_str = "One or more fatal post-simulation checks failed. It "\
        "is highly recommended that you address and correct each of these "\
        "problems. However, you can force SEEKR to skip these checks by using "\
        "the --skip_checks (-s) argument on analyze.py."
        print(check_fail_str)
        raise Exception("The SEEKR2 calculation can not proceed due "\
                        "to failed post-simulation checks.")
    
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Check SEEKR2 calculations for common problems both "\
        "before and after simulations.")
    
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="name of model file for OpenMMVT calculation. This would be the "\
        "XML file generated in the prepare stage.")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    model = structures.load_model(model_file)
    check_pre_simulation_all(model)
    check_post_simulation_all(model)
