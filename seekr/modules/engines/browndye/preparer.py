from __future__ import annotations
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
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.engines.browndye.base as browndye_base
import seekr.modules.engines.browndye.xml_structures as xml_structures
import seekr.modules.engines.browndye.structures as browndye_structures
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.scales.brownian_dynamics.structures as bd_scale

logger = logging.getLogger(__name__)

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
        # ParmEd rebuilds the full Nx3 array on every `.coordinates` access.
        # Cache it once; indexing inside a per-atom loop is otherwise O(N^2).
        coords = pqr_struct.coordinates
        assert coords is not None, (
            f"PQR '{self.pqr_filename}' has no coordinates.")
        centers_of_mass = []
        original_indices: typing.Dict[str, typing.List[int]] = {}
        ghost_site_indices: typing.Dict[str, int] = {}
        for selection in self.selections:
            # Compute the center of mass for the selected group of atoms
            center_of_mass = np.array([[0., 0., 0.]])
            total_mass = 0.0
            for atom_index in selection.atom_indices:
                atom_pos = coords[atom_index, :]
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
                atom_pos = coords[atom_index, :]
                atom_mass = atom.mass
                if atom_mass == 0.0:
                    atom_mass = 0.0001
                mol_center_of_mass += atom_mass * atom_pos
                mol_total_mass += atom_mass
            mol_center_of_mass = mol_center_of_mass / mol_total_mass
        
        pqr_complex = pqr_struct
        for selection, center_of_mass in zip(self.selections, centers_of_mass):
            # 4-char name so ParmEd's PQR writer does not skip the leading
            # space (3-char names with Z=0/"EP" misalign later columns).
            # Keep atomic_number=0 (extra point), not a real element.
            ghost_atom = parmed.Atom(
                name="GHOS", mass=0.0, charge=0.0, solvent_radius=0.0)
            ghost_structure = parmed.Structure()
            ghost_structure.add_atom(ghost_atom, "GHO", 1)
            ghost_structure.coordinates = np.array(center_of_mass)
            pqr_complex = pqr_complex + ghost_structure
            for residue in pqr_complex.residues:
                residue.chain = ""
            ghost_index = len(pqr_complex.atoms)
            original_indices[selection.name] = selection.atom_indices
            ghost_site_indices[selection.name] = ghost_index
        
        if center_molecule:
            complex_coords = pqr_complex.coordinates
            pqr_complex.coordinates = (
                complex_coords - mol_center_of_mass[0, :])
        
        logger.info(f"Saving PQR file with ghost atoms added: {new_pqr_filename}")
        pqr_complex.save(new_pqr_filename, overwrite=True)
        # ParmEd emits HETATM for nonstandard residues; Browndye cannot
        # parse HETATM glued to 5-digit serials (e.g. HETATM10000).
        with open(new_pqr_filename) as f:
            content = f.read()
        with open(new_pqr_filename, "w") as f:
            f.write(content.replace("HETATM", "ATOM  "))
        return ghost_site_indices, original_indices
    
@define
class Browndye_settings_input(engines_structures.Base_engine_settings_input):
    type: typing.Literal["browndye"] = "browndye"
    scale_type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    binary_directory: str = field(default="",
                                    validator=validators.instance_of(str))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=[validators.instance_of(float),
                                               validators.gt(0)])
    
    ions: list[engines_structures.Ion] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    engines_structures.Ion),
                                iterable_validator=validators.instance_of(list),
                            ))
    molecules: list[Browndye_molecule_input] = field(default=Factory(list),
                            validator=validators.deep_iterable(
                                member_validator=validators.instance_of(
                                    Browndye_molecule_input),
                                iterable_validator=validators.instance_of(list),
                            ))
    def make_model_scale(
            self
            ) -> bd_scale.Scale_bd:
        scale = bd_scale.Scale_bd()
        scale.engine = "browndye"
        return scale

    def make_model_engine(
            self,
            dummy_temperature: float
            ) -> browndye_structures.Browndye_bd_engine:
        """
        Make the model engine object for this input.
        """
        bd_engine = browndye_structures.Browndye_bd_engine()
        bd_engine.binary_directory = self.binary_directory
        bd_engine.apbs_grid_spacing = self.apbs_grid_spacing
        bd_engine.ions = self.ions
        bd_engine.unpartitioned_settings = browndye_structures.Browndye_unpartitioned_settings()
        bd_engine.unpartitioned_settings.molecules = []
        return bd_engine
    
    def start_from_unpartitioned(
            self,
            using_toy: bool
        ) -> bool:
        """
        Whether this engine uses unpartitioned settings to start from.
        It does - because BD simulations don't really have anchor-by-anchor
        parameters. Simulations start from the bulk anchor.
        """
        return True
    
    def copy_initial_files(
            self,
            model: structures.Seekr_model, 
            root_directory: str,
            anchor_set_name: str = "main",
            using_toy: bool = False # dummy parameter
            ) -> None:
        """
        Copy the BD files into the anchor directory.
        """
        bd_engine: browndye_structures.Browndye_bd_engine \
            = model.get_engine_by_scale_type("brownian_dynamics")
        assert bd_engine.unpartitioned_settings is not None, \
            "BD unpartitioned settings must be defined - anchor-by-anchor " \
            "not yet implemented."
        for anchor in model.anchors:
            if anchor.bulk_anchor:
                # If anchor.bd is True, while anchor.md is False, then this
                # must be the b-surface directory - "bulk anchor"
                b_surface_directory = os.path.join(
                    root_directory, base.ANCHOR_SETS_DIR, anchor_set_name, anchor.directory)
                for molecule in self.molecules:
                    pqr_filename = os.path.basename(molecule.pqr_filename)
                    new_bd_molecule = browndye_structures.Browndye_molecule()
                    new_bd_molecule.name = os.path.splitext(pqr_filename)[0]
                    new_bd_molecule.pqr_filename = pqr_filename
                    pqr_new_filename = os.path.join(b_surface_directory, pqr_filename)
                    ghost_site_indices, original_indices = molecule.add_ghost_atoms_to_pqr(
                        pqr_new_filename)
                    new_bd_molecule.original_indices = original_indices
                    new_bd_molecule.ghost_site_indices = ghost_site_indices
                    bd_engine.unpartitioned_settings.molecules.append(
                        new_bd_molecule)
                    
        return
    
    def get_atom_indices_by_selection_name(
            self,
            name: str
            ) -> list[int]:
        """
        Get the atom indices for a given selection name.
        """
        for molecule in self.molecules:
            for selection in molecule.selections:
                if selection.name == name:
                    return selection.atom_indices
        raise ValueError(f"Selection name {name} not found in any molecule.")

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
    logger.info("running command: " + make_apbs_command)
    os.system(make_apbs_command)
    if not os.path.exists(new_input_xml):
        logger.critical("Problem running make_apbs_input - Output file not found: "\
                              + new_input_xml)
        raise AssertionError("Problem running make_apbs_input - Output file not found: "\
                              + new_input_xml)
    
    for mol_name in mol_name_list:
        apbs_input_glob = mol_name + "*.in"
        if len(glob.glob(apbs_input_glob)) < 1:
            logger.critical("Problem running make_apbs_input - APBS input "\
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
        logger.critical("Problem: debye_length not generated.")
        raise AssertionError("Problem: debye_length not generated.")
    
    run_apbs_command = run_apbs_binary + " " + new_input_xml \
        + " > run_apbs_input.out"
    logger.info("running command: "+ run_apbs_command)
    os.system(run_apbs_command)
    for mol_name in mol_name_list:
        apbs_dx_glob = mol_name + "*.dx"
        if len(glob.glob(apbs_dx_glob)) < 1:
            logger.critical("Problem running run_apbs_input - APBS DX file(s) "\
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
        n_steps_per_output: int = 100000,
        desolvation_parameter: float = 1.0
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
    # TODO: pass this variable from the model object, not a constant - sometimes
    # It's helpful to have an alternative possibility for a Reaction filename.
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
    root.system.solvent.desolvation_parameter = desolvation_parameter
    root.system.time_step_tolerances.minimum_core_dt = 0.2
    root.system.time_step_tolerances.minimum_core_reaction_dt = 0.05
    
    root.system.solvent.ions = []
    for model_ion in bd_engine.ions:
        bd_ion = xml_structures.Ion()
        bd_ion.radius = model_ion.radius
        bd_ion.charge = model_ion.charge
        bd_ion.conc = model_ion.concentration
        root.system.solvent.ions.append(bd_ion)
    
    assert bd_engine.unpartitioned_settings is not None, \
        "Unpartitioned settings must be set for BD"
    for molecule in bd_engine.unpartitioned_settings.molecules:
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

    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    cvs = model.collective_variables
    bulk_anchors: list[structures.Anchor] = []
    for anchor in model.anchors:
        if anchor.bulk_anchor:
            bulk_anchors.append(anchor)

    cross_site_information: dict = {}

    assert len(bulk_anchors) > 0, "No BD anchor found"
    for bulk_anchor in bulk_anchors:
        for neighbor_anchor_index in bulk_anchor.neighbor_anchor_indices:
            neighbor_anchor = model.anchors[neighbor_anchor_index]
            spherical_milestones_in_neighbor_anchor, this_anchors_bd_cv_index \
                = engines_preparer.get_spherical_milestones_and_cv_index_in_neighbor_anchor(
                    model, neighbor_anchor)
            assert this_anchors_bd_cv_index is not None
            this_cv: cv_structures.Collective_variable = cvs[this_anchors_bd_cv_index]
            assert isinstance(this_cv, distance_cv.Distance_CV), \
                "BD anchor must use distance CV - other CVs not yet implemented"
            this_cv_group1_name: str = this_cv.group1_name
            this_cv_group2_name: str = this_cv.group2_name
            ghost_index_1: int = -1
            ghost_index_2: int = -1
            molecule1_name: str = ""
            molecule2_name: str = ""
            assert bd_engine is not None, "BD engine must be set to make reaction files"
            assert bd_engine.unpartitioned_settings is not None, \
                "Unpartitioned settings must be set to make reaction files"
            for molecule in bd_engine.unpartitioned_settings.molecules:
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
            outer_milestone: structures.Milestone = spherical_milestones_in_neighbor_anchor[0]
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

            cross_site_information[outer_milestone.index] \
                = {"distance": pair1.distance,
                   "state": rxn_outer.state_after,
                   "molecule0_group": rxn_outer.molecule0_group,
                   "molecule0_core": rxn_outer.molecule0_core,
                   "molecule1_group": rxn_outer.molecule1_group,
                   "molecule1_core": rxn_outer.molecule1_core,
                   "n_needed": rxn_outer.n_needed,
                   "ghost_index_1": ghost_index_1,
                   "ghost_index_2": ghost_index_2,}
            
            rxn_inner = xml_structures.Reaction()
            pair2 = xml_structures.Pair()
            from_state = str(outer_milestone.index)
            inner_milestone = spherical_milestones_in_neighbor_anchor[1]
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

            # TODO: remove this block?
            for other_neighbor_anchor_index in bulk_anchor.neighbor_anchor_indices:
                if other_neighbor_anchor_index == neighbor_anchor_index:
                    continue
                other_neighbor_anchor = model.anchors[other_neighbor_anchor_index]
                other_spherical_milestones_in_this_anchor = []
                other_anchors_bd_cv_index = None
                for other_milestone in other_neighbor_anchor.milestones:
                    found_variable_in_this_other_milestone = False
                    for other_variable in other_milestone.variables:
                        if other_variable.startswith("me"):
                            other_cv_index = other_milestone.get_cv_index_from_variable(other_variable)
                            if cvs[other_cv_index].type == "distance":
                                other_spherical_milestones_in_this_anchor.append(
                                    other_milestone)
                                if other_anchors_bd_cv_index is None:
                                    other_anchors_bd_cv_index = other_cv_index
                                else:
                                    assert other_anchors_bd_cv_index == other_cv_index, \
                                        "BD anchor is attempting to use multiple cvs to "\
                                        "define reaction criteria - not implemented."
                                if not found_variable_in_this_other_milestone:
                                    found_variable_in_this_other_milestone = True
                                else:
                                    raise AssertionError("Milestone has multiple distance CVs"\
                                                        " - not implemented")
                                
                # Sort by the average between the positions of the neighboring anchors
                #other_spherical_milestones_in_this_anchor.sort(
                #    key=lambda x: x.variables["avg_value"], reverse=True)

        for milestone1_index, info1 in cross_site_information.items():
            for milestone2_index, info2 in cross_site_information.items():
                if milestone1_index == milestone2_index:
                    continue
                rxn = xml_structures.Reaction()
                pair = xml_structures.Pair()
                from_state: str = info1["state"]
                to_state: str = info2["state"]
                rxn.name = f"{from_state}_{to_state}"
                rxn.state_after = to_state
                pair.distance = info2["distance"]
                rxn.state_before = from_state
                rxn.molecule0_group = info2["molecule0_group"]
                rxn.molecule0_core = info2["molecule0_core"]
                rxn.molecule1_group = info2["molecule1_group"]
                rxn.molecule1_core = info2["molecule1_core"]
                rxn.n_needed = info2["n_needed"]
                pair.atom1_index = info2["ghost_index_1"]
                pair.atom2_index = info2["ghost_index_2"]
                rxn.pair_list.append(pair)
                rxnroot.reaction_list.append(rxn)

    rxnroot.write(abs_reaction_path)
    return

def generate_bd_files(
        model: structures.Seekr_model,
        root_directory: str,
        ) -> None:
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert bd_engine is not None
    b_surface_full_path = os.path.join(
        root_directory, bd_engine.b_surface_directory)
    assert bd_engine.unpartitioned_settings is not None, \
        "Unpartitioned settings must be set to generate BD files."
    for molecule in bd_engine.unpartitioned_settings.molecules:
        molecule.generate_pqrxml_file(b_surface_full_path, bd_engine.binary_directory)
    debye_length, reaction_filename = make_browndye_input_xml(bd_engine,
        temperature=model.temperature, root_directory=root_directory)
    bd_engine.debye_length = float(debye_length)
    abs_reaction_path = os.path.join(b_surface_full_path, reaction_filename)
    make_browndye_reaction_xml(model, abs_reaction_path)
    return
