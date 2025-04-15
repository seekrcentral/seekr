"""
modules/engines/browndye/input.py

Objects and routines for the input stage of an anchor that
uses Browndye.
"""

import os
import re
import glob
import logging
import typing
from attrs import define, field, validators, Factory

import numpy as np
import parmed

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.engines.browndye.xml_structures as xml_structures
import seekr.modules.engines.browndye.structures as browndye_structures
import seekr.modules.collective_variables.structures as cv_structures

@define
class Browndye_molecule_input:
    name: str = field(default="",
                        validator=validators.instance_of(str))
    pqr_filename: str = field(default="",
                            validator=validators.instance_of(str))
    selections: list[engines_structures.Selection] = field(default=Factory(list),
                                        validator=validators.deep_iterable(
                                            member_validator=validators.instance_of(
                                                engines_structures.Selection),
                                            iterable_validator=validators.instance_of(
                                                list),
                                        ))
    
    def add_ghost_atoms_to_pqr(
            self,
            new_pqr_filename: str,
            center_molecule: bool = True,
            ) -> typing.Dict[str, int]:
        """
        Add a ghost atom to the PQR file to represent a significant location.
        Write a new PQR file and return the index of the ghost atom.
        """
        pqr_struct = parmed.load_file(self.pqr_filename, skip_bonds=True)
        centers_of_mass = []
        ghost_site_indices: typing.Dict[str, int] = {}
        for selection in self.selections:
            # Compute the center of mass for the selected group of atoms
            center_of_mass = np.array([[0., 0., 0.]])
            total_mass = 0.0
            for atom_index in selection.atom_indices:
                atom_pos = pqr_struct.coordinates[atom_index,:]
                atom_mass = pqr_struct.atoms[atom_index].mass
                if atom_mass == 0.0:
                    atom_mass = 0.0001
                center_of_mass += atom_mass * atom_pos
                total_mass += atom_mass
            center_of_mass = center_of_mass / total_mass
            centers_of_mass.append(center_of_mass)
        
        if center_molecule:
            # Compute the center of mass of the entire molecule to be transposed
            mol_center_of_mass = np.array([[0., 0., 0.]])
            mol_total_mass = 0.0
            for atom_index, atom in enumerate(pqr_struct.atoms):
                atom_pos = pqr_struct.coordinates[atom_index,:]
                atom_mass = atom.mass
                if atom_mass == 0.0:
                    atom_mass = 0.0001
                mol_center_of_mass += atom_mass * atom_pos
                mol_total_mass += atom_mass
            mol_center_of_mass = mol_center_of_mass / mol_total_mass
        
        pqr_complex = pqr_struct
        for selection, center_of_mass in zip(self.selections, centers_of_mass):
            ghost_atom = parmed.Atom(name="GHO", mass=0.0, charge=0.0, solvent_radius=0.0)
            ghost_structure = parmed.Structure()
            ghost_structure.add_atom(ghost_atom, "GHO", 1)
            ghost_structure.coordinates = np.array(center_of_mass)
            pqr_complex = pqr_complex + ghost_structure
            for residue in pqr_complex.residues:
                residue.chain = ""
            ghost_index = len(pqr_complex.atoms)
            ghost_site_indices[selection.name] = ghost_index
        
        if center_molecule:
            new_coordinates = np.zeros(pqr_complex.coordinates.shape)
            for atom_index in range(len(pqr_complex.atoms)):
                new_coordinates[atom_index,:] = pqr_complex.coordinates[atom_index,:] \
                    - mol_center_of_mass[0,:]
                    
            pqr_complex.coordinates = new_coordinates
        
        pqr_complex.save(new_pqr_filename, overwrite=True)
        return ghost_site_indices
    
@define
class Browndye_settings_input:
    type: typing.Literal["browndye"] = "browndye"
    binary_directory: str = field(default="",
                                    validator=validators.instance_of(str))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    
    ions: list[browndye_structures.Ion] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    browndye_structures.Ion),
                                iterable_validator=validators.instance_of(list),
                            ))
    molecules: list[Browndye_molecule_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    Browndye_molecule_input),
                                iterable_validator=validators.instance_of(list),
                            ))
    def make_model_engine(self):
        """
        Make the model engine object for this input.
        """
        bd_engine = browndye_structures.Browndye_bd_engine()
        bd_engine.binary_directory = self.binary_directory
        bd_engine.apbs_grid_spacing = self.apbs_grid_spacing
        bd_engine.ions = self.ions
        bd_engine.systemwide_settings = browndye_structures.Browndye_systemwide_settings()
        bd_engine.systemwide_settings.molecules = []
        return bd_engine

def make_and_run_apbs(
        root: xml_structures.Root, 
        input_apbs_xml: str, 
        browndye_bin: str = "", 
        new_input_xml_base: str = "input.xml") -> float:
    """
    Given an entire BD calculation Root() object, prepare and run
    APBS calculations.
    """
    assert root.system.solvent.ions is not None, \
        "Ions must be included for APBS calculations"
    mol_name_list = []
    for group in root.system.group_list:
        mol_name = group.name
        mol_name_list.append(mol_name)
    
    bd_dir = os.path.dirname(input_apbs_xml)
    os.chdir(bd_dir)
    new_input_xml = os.path.join(bd_dir, new_input_xml_base)
    make_apbs_binary = os.path.join(browndye_bin, "make_apbs_inputs")
    run_apbs_binary = os.path.join(browndye_bin, "run_apbs_inputs")
    make_apbs_command = make_apbs_binary + " " + input_apbs_xml + " > " \
        + new_input_xml
    logging.info("running command: " + make_apbs_command)
    os.system(make_apbs_command)
    if not os.path.exists(new_input_xml):
        logging.critical("Problem running make_apbs_input - Output file not found: "\
                              + new_input_xml)
        raise AssertionError("Problem running make_apbs_input - Output file not found: "\
                              + new_input_xml)
    
    for mol_name in mol_name_list:
        apbs_input_glob = mol_name + "*.in"
        if len(glob.glob(apbs_input_glob)) < 1:
            logging.critical("Problem running make_apbs_input - APBS input "\
                             "files not found: " + apbs_input_glob)
            raise AssertionError("Problem running make_apbs_input - APBS input "\
                                "files not found: " + apbs_input_glob)
    
    debye_length = -1.0
    with open(new_input_xml, "r") as f:
        for line in f.readlines():
            result = re.search("<debye_length> (.+?) </debye_length>", line)
            if result:
                debye_length = float(result.group(1))
                
    if debye_length <= 0.0:
        logging.critical("Problem: debye_length not generated.")
        raise AssertionError("Problem: debye_length not generated.")
    
    run_apbs_command = run_apbs_binary + " " + new_input_xml \
        + " > run_apbs_input.out"
    logging.info("running command: "+ run_apbs_command)
    os.system(run_apbs_command)
    for mol_name in mol_name_list:
        apbs_dx_glob = mol_name + "*.dx"
        if len(glob.glob(apbs_dx_glob)) < 1:
            logging.critical("Problem running run_apbs_input - APBS DX file(s) "\
                             "not found: " + apbs_dx_glob)
            raise AssertionError("Problem running run_apbs_input - APBS DX file(s) "\
                                "not found: " + apbs_dx_glob)
    
    return debye_length

def make_browndye_input_xml(
        bd_engine: browndye_structures.Browndye_bd_engine,
        temperature: float,
        root_directory: str,
        make_apbs_mode: bool = True,
        n_trajectories: int = 1,
        n_threads: int = 1,
        n_steps_per_output: int = 100000
        ) -> typing.Tuple[float, str]:
    """
    Make the Browndye input XML file for the system.
    """
    root = xml_structures.Root()
    bd_directory = os.path.join(
        root_directory, bd_engine.b_surface_directory)
    root.system.start_at_site = "False"
    root.n_trajectories = n_trajectories
    root.n_steps_per_output = n_steps_per_output
    root.seed = bd_engine.get_random_seed()
    # TODO: set default for these values and check them at the run stage
    #  or at least before input generation.
    #if root.n_trajectories_per_output > root.n_trajectories:
    #    root.n_trajectories_per_output = root.n_trajectories
    root.n_threads = n_threads
    root.system.reaction_file = browndye_base.REACTION_FILENAME
    reaction_filename = root.system.reaction_file
    if make_apbs_mode:
        root.system.solvent.debye_length = 9.9e9
        input_xml_filename = os.path.join(bd_directory, "apbs_input.xml")
    else:
        root.system.solvent.debye_length = bd_engine.debye_length
        assert float(root.system.solvent.debye_length) > 0.0, \
            "The Debye length must be set if make_apbs_mode=False"
        input_xml_filename = os.path.join(bd_directory, "input.xml")
        
    root.system.solvent.kT = temperature / 298.0
    root.system.time_step_tolerances.minimum_core_dt = 0.2
    root.system.time_step_tolerances.minimum_core_reaction_dt = 0.05
    
    root.system.solvent.ions = []
    for model_ion in bd_engine.ions:
        bd_ion = xml_structures.Ion()
        bd_ion.radius = model_ion.radius
        bd_ion.charge = model_ion.charge
        bd_ion.conc = model_ion.concentration
        root.system.solvent.ions.append(bd_ion)
    
    assert bd_engine.systemwide_settings is not None, \
        "Systemwide settings must be set for BD"
    for molecule in bd_engine.systemwide_settings.molecules:
        molecule_group = xml_structures.Group()
        molecule_group.name = molecule.name
        molecule_core = xml_structures.Core()
        molecule_core.name = molecule.name
        assert molecule.pqrxml_filename is not None, \
            "PQRXML file must be generated before BD input generation"
        molecule_pqrxml_filename: str = molecule.pqrxml_filename
        molecule_core.atoms = molecule_pqrxml_filename
        molecule_core.grid_spacing = bd_engine.apbs_grid_spacing
        molecule_group.core_list.append(molecule_core)
        root.system.group_list.append(molecule_group)

    if make_apbs_mode:
        root.write(input_xml_filename)
        debye_length = make_and_run_apbs(
            root, input_xml_filename, browndye_bin=bd_engine.binary_directory,
            new_input_xml_base=browndye_base.BROWNDYE_INPUT_FILENAME)
    else:
        root.write(input_xml_filename)
        debye_length = bd_engine.debye_length
                
    return debye_length, reaction_filename

def make_browndye_reaction_xml(
        model: structures.Seekr_model | None,
        abs_reaction_path: str
        ) -> None:
    """
    Creates Browndye reaction file.
    """
    rxnroot = xml_structures.Reaction_root()
    rxnroot.first_state = "b_surface"
    if model is None:
        rxnroot.write(abs_reaction_path)
        return

    bd_engine = model.bd_engine
    cvs = model.collective_variables
    bulk_anchor: structures.Anchor
    for anchor in model.anchors:
        if anchor.bd and not anchor.md:
            bulk_anchor = anchor

    assert bulk_anchor is not None, "No BD anchor found"
    for neighbor_anchor_index in bulk_anchor.neighbor_anchor_indices:
        neighbor_anchor = model.anchors[neighbor_anchor_index]
        spherical_milestones_in_this_anchor = []
        this_anchors_bd_cv_index = None
        for milestone in neighbor_anchor.milestones:
            found_variable_in_this_milestone = False
            for variable in milestone.variables:
                if variable.startswith("me"):
                    cv_index = milestone.get_cv_index_from_variable(variable)
                    if cvs[cv_index].type == "distance":
                        spherical_milestones_in_this_anchor.append(
                            milestone)
                        if this_anchors_bd_cv_index is None:
                            this_anchors_bd_cv_index = cv_index
                        else:
                            assert this_anchors_bd_cv_index == cv_index, \
                                "BD anchor is attempting to use multiple cvs to "\
                                "define reaction criteria - not implemented."
                        if not found_variable_in_this_milestone:
                            found_variable_in_this_milestone = True
                        else:
                            raise AssertionError("Milestone has multiple distance CVs"\
                                                 " - not implemented")
                        
        assert len(spherical_milestones_in_this_anchor) == 2, \
            "BD anchor must have exactly 2 spherical milestones - "\
            "other arrangements not yet implemented"
        # Sort by the average between the positions of the neighboring anchors
        spherical_milestones_in_this_anchor.sort(
            key=lambda x: x.variables["avg_value"], reverse=True)
        assert this_anchors_bd_cv_index is not None
        this_cv: cv_structures.Distance_CV = cvs[this_anchors_bd_cv_index]
        this_cv_group1_name: str = this_cv.group1_name
        this_cv_group2_name: str = this_cv.group2_name
        ghost_index_1: int = -1
        ghost_index_2: int = -1
        molecule1_name: str = ""
        molecule2_name: str = ""
        assert bd_engine is not None, "BD engine must be set to make reaction files"
        assert bd_engine.systemwide_settings is not None, \
            "Systemwide settings must be set to make reaction files"
        for molecule in bd_engine.systemwide_settings.molecules:
            assert molecule.ghost_site_indices is not None, \
                "Ghost site indices must be set for all molecules"
            if this_cv_group1_name in molecule.ghost_site_indices:
                ghost_index_1 = molecule.ghost_site_indices[this_cv_group1_name]
                molecule1_name = molecule.name
            if this_cv_group2_name in molecule.ghost_site_indices:
                ghost_index_2 = molecule.ghost_site_indices[this_cv_group2_name]
                molecule2_name = molecule.name
        assert ghost_index_1 is not None, f"Ghost index named {this_cv_group1_name} not found"
        assert ghost_index_2 is not None, f"Ghost index named {this_cv_group2_name} not found"
        
        # Next identify which is the inner and the outer milestone
        rxn_outer = xml_structures.Reaction()
        pair1 = xml_structures.Pair()
        from_state: str = "b"
        outer_milestone: structures.Milestone = spherical_milestones_in_this_anchor[0]
        to_state: str = str(outer_milestone.index)
        rxn_outer.name = f"{from_state}_{to_state}"
        rxn_outer.state_after = str(outer_milestone.index)
        pair1.distance = outer_milestone.variables['avg_value'] * 10.0
        rxn_outer.state_before = rxnroot.first_state
        rxn_outer.molecule0_group = molecule1_name
        rxn_outer.molecule0_core = molecule1_name
        rxn_outer.molecule1_group = molecule2_name
        rxn_outer.molecule1_core = molecule2_name
        rxn_outer.n_needed = 1
        pair1.atom1_index = ghost_index_1
        pair1.atom2_index = ghost_index_2
        rxn_outer.pair_list.append(pair1)
        rxnroot.reaction_list.append(rxn_outer)
        
        rxn_inner = xml_structures.Reaction()
        pair2 = xml_structures.Pair()
        from_state = str(outer_milestone.index)
        inner_milestone = spherical_milestones_in_this_anchor[1]
        to_state = str(inner_milestone.index)
        rxn_inner.name = "{}_{}".format(from_state, to_state)
        rxn_inner.state_after = str(inner_milestone.index)
        pair2.distance = inner_milestone.variables['avg_value'] * 10.0
        rxn_inner.state_before = str(outer_milestone.index)
        rxn_inner.molecule0_group = molecule1_name
        rxn_inner.molecule0_core = molecule1_name
        rxn_inner.molecule1_group = molecule2_name
        rxn_inner.molecule1_core = molecule2_name
        rxn_inner.n_needed = 1
        pair2.atom1_index = ghost_index_1
        pair2.atom2_index = ghost_index_2
        rxn_inner.pair_list.append(pair2)
        rxnroot.reaction_list.append(rxn_inner)
        
        """ # TODO: return to this later - cross-site reactions.
        for j, bd_milestone3 in enumerate(model.k_on_info.bd_milestones):
            if bd_milestone2.index == bd_milestone3.index:
                continue
            rxn_other = sim_browndye2.Reaction()
            pair3 = sim_browndye2.Pair()
            from_state = str(bd_milestone2.outer_milestone.index)
            to_state = str(bd_milestone3.outer_milestone.index)
            rxn_other.name = "{}_{}".format(from_state, to_state)
            rxn_other.state_after = str(bd_milestone3.outer_milestone.index)
            pair3.distance = bd_milestone3.outer_milestone.variables['radius'] * 10.0
            rxn_other.state_before = str(bd_milestone2.outer_milestone.index)
            rxn_other.molecule0_group = sim_browndye2.BROWNDYE_RECEPTOR
            rxn_other.molecule0_core = sim_browndye2.BROWNDYE_RECEPTOR
            rxn_other.molecule1_group = sim_browndye2.BROWNDYE_LIGAND
            rxn_other.molecule1_core = sim_browndye2.BROWNDYE_LIGAND
            rxn_other.n_needed = 1
            other_ghost_index_rec = ghost_indices_rec[j] # comes from the model?
            other_ghost_index_lig = ghost_indices_lig[j]
            pair3.atom1_index = other_ghost_index_rec
            pair3.atom2_index = other_ghost_index_lig
            rxn_other.pair_list.append(pair3)
            rxnroot.reaction_list.append(rxn_other)
        """
        
    rxnroot.write(abs_reaction_path)
    return

def generate_bd_files(
        bd_engine: browndye_structures.Browndye_bd_engine, 
        model: structures.Seekr_model,
        root_directory: str
        ) -> None:
    b_surface_full_path = os.path.join(
        root_directory, bd_engine.b_surface_directory)
    assert bd_engine.systemwide_settings is not None, \
        "Systemwide settings must be set to generate BD files."
    for molecule in bd_engine.systemwide_settings.molecules:
        molecule.generate_pqrxml_file(b_surface_full_path, bd_engine.binary_directory)
    debye_length, reaction_filename = make_browndye_input_xml(bd_engine,
        temperature=model.temperature, root_directory=root_directory)
    bd_engine.debye_length = float(debye_length)
    abs_reaction_path = os.path.join(b_surface_full_path, reaction_filename)
    make_browndye_reaction_xml(model, abs_reaction_path)
    return