"""
modules/engines/sda/input.py

Objects and routines for the input stage of an anchor that
uses SDA.
"""

import os
import typing

from attrs import define, field, Factory
from attrs.validators import and_, instance_of, in_, deep_iterable, gt, ge
import numpy as np
import parmed

import seekr.modules.structures as structures
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.structures as cv_structures
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.sda.base as sda_base
import seekr.modules.engines.sda.structures as sda_structures
import seekr.modules.engines.sda.sda_structures as sda_sda_structures
import seekr.modules.scales.brownian_dynamics.structures as bd_scale
import seekr.modules.engines.preparer as engines_preparer

@define
class SDA_molecule_input:
    name: str = field(default="",
                        validator=instance_of(str))
    type: str = field(default="protein",
                    validator=and_(instance_of(str), in_(
                        ["protein", "non_protein"])))
    pqr_filename: str = field(default="",
                            validator=instance_of(str))
    selections: list[engines_structures.Selection] = field(default=Factory(list),
                                        validator=deep_iterable(
                                            member_validator=instance_of(
                                                engines_structures.Selection),
                                            iterable_validator=instance_of(
                                                list),
                                        ))
    # TODO: Abraham had to change this. Why?
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
        original_indices: typing.Dict[str, typing.List[int]] = {}
        ghost_site_indices: typing.Dict[str, int] = {}
        ghost_atom_lines: typing.Dict[str, str] = {}
        ghost_atom_line_list: typing.List[str] = []
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
            original_indices[selection.name] = selection.atom_indices
            ghost_site_indices[selection.name] = ghost_index
            
        
        if center_molecule:
            new_coordinates = np.zeros(pqr_complex.coordinates.shape)
            for atom_index in range(len(pqr_complex.atoms)):
                new_coordinates[atom_index,:] = pqr_complex.coordinates[atom_index,:] \
                    - mol_center_of_mass[0,:]
                    
            pqr_complex.coordinates = new_coordinates
        
        with open(new_pqr_filename, "w") as pqr_file:
            for i, atom in enumerate(pqr_complex.atoms):
                x, y, z = pqr_complex.coordinates[i][:]
                if self.type == "protein":
                    atom_str = "ATOM{:>7d} {:4s} {:3s}  {:>4s}{:>12.3f}{:>8.3f}{:>8.3f} {:>8.4f} {:>8.4f}\n".format(
                        i+1, atom.name, atom.residue.name, str(atom.residue.idx+1), 
                        x, y, z, atom.charge, atom.solvent_radius)
                elif self.type == "non_protein":
                    atom_str = "HETATM{:>5d} {:4s} {:3s}  {:>4s}{:>12.3f}{:>8.3f}{:>8.3f} {:>8.4f} {:>8.4f}\n".format(
                        i+1, atom.name, atom.residue.name, str(atom.residue.idx+1), 
                        x, y, z, atom.charge, atom.solvent_radius)
                pqr_file.write(atom_str)
        
        for selection in self.selections:
            ghost_atom = pqr_complex.atoms[ghost_site_indices[selection.name]-1]
            x, y, z = pqr_complex.coordinates[i][:]
            ghost_atom_line = "ATOM{:>7s} {:4s} {:3s}  {:>4s}{:>12.3f}{:>8.3f}{:>8.3f}".format(
                str(ghost_index), ghost_atom.name, ghost_atom.residue.name, str(ghost_atom.residue.idx), 
                x, y, z)
            ghost_atom_lines[selection.name] = ghost_atom_line
            ghost_atom_line_list.append(ghost_atom_line)
        
        return ghost_site_indices, original_indices, ghost_atom_lines
    
@define
class SDA_settings_input:
    type: typing.Literal["sda"] = "sda"
    scale_type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
    geometry: str = field(default="sphere",
                            validator=in_(
                                ["sphere", "box", "nambox"]))
    binary_directory: str = field(default="",
                                    validator=instance_of(str))
    aux_directory: str = field(default="",
                                    validator=instance_of(str))
    hydropro_directory: str = field(default="",
                                    validator=instance_of(str))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=[instance_of(float),
                                               gt(0)])
    
    ions: list[engines_structures.Ion] = field(default=Factory(list),
                            validator=deep_iterable(
                                member_validator=instance_of(
                                    engines_structures.Ion),
                                iterable_validator=instance_of(list),
                            ))
    molecules: list[SDA_molecule_input] = field(default=Factory(list),
                            validator=deep_iterable(
                                member_validator=instance_of(
                                    SDA_molecule_input),
                                iterable_validator=instance_of(list),
                         ))
    # atoms list?

    def make_model_scale(
            self
            ) -> bd_scale.Scale_bd:
        scale = bd_scale.Scale_bd()
        scale.engine = "sda"
        return scale
    
    def make_model_engine(
            self,
            dummy_temperature: float
            ) -> sda_structures.SDA_bd_engine:
        """
        Make the model engine object for this input.
        """
        bd_engine = sda_structures.SDA_bd_engine()
        bd_engine.binary_directory = self.binary_directory
        bd_engine.aux_directory = self.aux_directory
        bd_engine.hydropro_directory = self.hydropro_directory
        bd_engine.apbs_grid_spacing = self.apbs_grid_spacing
        bd_engine.ions = self.ions
        bd_engine.systemwide_settings = sda_structures.SDA_systemwide_settings()
        bd_engine.systemwide_settings.molecules = []
        bd_engine.systemwide_settings.atomic_parameters = []
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="H*", resname="*", vdw=1.2, test_charge=None))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="I", resname="*", vdw=1.98, test_charge=None))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="BR", resname="*", vdw=1.85, test_charge=None))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="CL", resname="*", vdw=1.75, test_charge=None))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="F", resname="*", vdw=1.47, test_charge=None))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="N*", resname="NTR", vdw=None, test_charge=1.0))
        bd_engine.systemwide_settings.atomic_parameters.append(
            sda_structures.SDA_atomic_parameter(
                type="O*", resname="CTR", vdw=None, test_charge=-0.5))
        return bd_engine
    
    def start_from_systemwide(
            self,
            using_toy: bool
        ) -> bool:
        """
        Whether this engine uses systemwide settings to start from.
        It does - because BD simulations don't really have anchor-by-anchor
        parameters. Simulations start from the bulk anchor.
        """
        return True
    
    def copy_initial_files(
            self,
            model: structures.Seekr_model, 
            root_directory: str,
            using_toy: bool = False  # dummy parameter
    ) -> None:
        """
        Copy the BD files into the anchor directory.
        """
        bd_engine: sda_structures.SDA_bd_engine \
            = model.get_engine_by_scale_type("brownian_dynamics")
        assert bd_engine.systemwide_settings is not None, \
            "BD systemwide settings must be defined - anchor-by-anchor " \
            "not yet implemented."
        for anchor in model.anchors:
            if anchor.bulk_anchor:
                # If anchor.bd is True, while anchor.md is False, then this
                # must be the b-surface directory - "bulk anchor"
                b_surface_directory = os.path.join(
                    root_directory, anchor.directory)
                for molecule in self.molecules:
                    pqr_filename = os.path.basename(molecule.pqr_filename)
                    new_bd_molecule = sda_structures.SDA_molecule()
                    new_bd_molecule.name = os.path.splitext(pqr_filename)[0]
                    new_bd_molecule.type = molecule.type
                    new_bd_molecule.pqr_filename = pqr_filename
                    pqr_new_filename = os.path.join(b_surface_directory, pqr_filename)
                    ghost_site_indices, original_indices, ghost_atom_lines \
                          = molecule.add_ghost_atoms_to_pqr(
                                pqr_new_filename)
                    new_bd_molecule.original_indices = original_indices
                    new_bd_molecule.ghost_site_indices = ghost_site_indices
                    new_bd_molecule.ghost_lines = ghost_atom_lines
                    bd_engine.systemwide_settings.molecules.append(new_bd_molecule)
                    
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

def make_sda_reaction_file(
        model: structures.Seekr_model,
        root_directory: str
        ) -> structures.Anchor:
    """
    Creates SDA reaction file.
    
    Parameters
    ----------
    model : Model()
        The model object contains all information needed for a SEEKR2
        calculation.

    rootdir : str
        The model's root directory.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    wrote_reaction_file = False
    for anchor in model.anchors:
        if anchor.bulk_anchor:
            b_surface_directory = os.path.join(
                root_directory, anchor.directory)
            cvs = model.collective_variables
            bulk_anchor = anchor

            assert bulk_anchor is not None, "No BD anchor found"
            for neighbor_anchor_index in bulk_anchor.neighbor_anchor_indices:
                # TODO: enable the ability to have multiple sites accessible from BD
                if wrote_reaction_file:
                    raise AssertionError("Multiple BD anchors not yet implemented - "\
                                        "only one BD site is supported")

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
                this_cv: cv_structures.Collective_variable = cvs[this_anchors_bd_cv_index]
                assert isinstance(this_cv, distance_cv.Distance_CV), \
                    "BD anchor must use distance CV - other CVs not yet implemented"
                this_cv_group1_name: str = this_cv.group1_name
                this_cv_group2_name: str = this_cv.group2_name
                ghost_index_1: int = -1
                ghost_index_2: int = -1
                ghost_line_1: str = ""
                ghost_line_2: str = ""
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
                        assert molecule.ghost_lines is not None, \
                            f"Ghost lines not set for molecule {molecule.name}"
                        ghost_line_1 = molecule.ghost_lines[this_cv_group1_name]
                        molecule1_name = molecule.name
                    if this_cv_group2_name in molecule.ghost_site_indices:
                        ghost_index_2 = molecule.ghost_site_indices[this_cv_group2_name]
                        assert molecule.ghost_lines is not None, \
                            f"Ghost lines not set for molecule {molecule.name}"
                        ghost_line_2 = molecule.ghost_lines[this_cv_group2_name]
                        molecule2_name = molecule.name
                assert ghost_index_1 is not None, f"Ghost index named {this_cv_group1_name} not found"
                assert ghost_index_2 is not None, f"Ghost index named {this_cv_group2_name} not found"
                
                abs_reaction_path = os.path.join(b_surface_directory, sda_base.REACTION_BASENAME)
                sda_reaction = sda_sda_structures.Reaction([ghost_line_1], [ghost_line_2])
                sda_reaction.write(abs_reaction_path)
                wrote_reaction_file = True

            break

    return anchor

def make_sda_pdb_noh(
        model: structures.Seekr_model,
        bd_anchor: structures.Anchor,
        root_directory: str
        ) -> None:
    """
    Make a PDB file without hydrogens for the anchor.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    anchor = bd_anchor
    b_surface_directory = os.path.join(
        root_directory, anchor.directory)
    curdir = os.getcwd()
    os.chdir(b_surface_directory)
    for molecule in bd_engine.systemwide_settings.molecules:
        pqr_filename = os.path.join(b_surface_directory, molecule.pqr_filename)
        pdb_noh_filename = os.path.splitext(pqr_filename)[0] + "_noh.pdb"
        full_structure = parmed.load_file(pqr_filename, skip_bonds=True)
        noh_structure = full_structure["!@H*"]
        noh_structure.save(pdb_noh_filename, overwrite=True)
        molecule.pdb_noh_filename = os.path.basename(pdb_noh_filename)

    os.chdir(curdir)
    
    return

def run_hydropro(
        model: structures.Seekr_model,
        bd_anchor: structures.Anchor,
        root_directory: str,
        hydropro_directory: str,
        ) -> None:
    """
    Creates hydropro's input files and run hydropro to calculate
    translational and diffussional coefficients of the solutes

    Parameters
    ----------
    model : Model()
        The SEEKR2 model.

    rootdir : str
        The model's root directory.
           
    hydropro_dir : str
        Directory where hydropro program is stored.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    b_surface_directory = os.path.join(
        root_directory, bd_anchor.directory)
    curdir = os.getcwd()
    os.chdir(b_surface_directory)
    hydropro_command = f"{hydropro_directory}/hydropro10-lnx.exe"
    for molecule in bd_engine.systemwide_settings.molecules:
        molecule.run_hydropro(
            temperature=model.temperature,
            hydropro_command=hydropro_command
            )
    
    os.chdir(curdir)
    return

def make_sda_grids(
        bd_engine: sda_structures.SDA_bd_engine,
        anchor: structures.Anchor,
        root_directory: str,
        temperature: float,
        solvent_dielectric: float
        ) -> None:
    """
    Creates SDA grids and effective charge models.
    
    Parameters
    ----------
    model : Model()
        The model object contains all information needed for a SEEKR2
        calculation.

    rootdir : str
        The model's root directory.
    
    sda_bin_dir : str
        Path to the SDA binary files.

    sda_auxi_dir : str
        Path to the SDA binary files.
    """
    b_surface_directory = os.path.join(
        root_directory, anchor.directory)
    curdir = os.getcwd()
    os.chdir(b_surface_directory)
    binary_directory = os.path.expanduser(bd_engine.binary_directory)
    aux_directory = os.path.expanduser(bd_engine.aux_directory)
    for molecule in bd_engine.systemwide_settings.molecules:
        molecule.compute_grid_dime(bd_engine.apbs_grid_spacing)
        molecule.make_sda_grids(
            temperature=temperature,
            binary_directory=binary_directory,
            aux_directory=aux_directory,
            apbs_grid_spacing=bd_engine.apbs_grid_spacing,
            ions=bd_engine.ions,
            solvent_dielectric=solvent_dielectric,
        )
    
    os.chdir(curdir)
    return

def add_atoms_sda(
        bd_engine: sda_structures.SDA_bd_engine,
        bd_anchor: structures.Anchor,
        root_directory: str
        ) -> None:
    """
    Creates add_atoms input file, which contains specific parameters
    for special atoms such as halogen's VdW or NTR and CTR's test charges
    
    Parameters
    ----------
    model : Model()
        The SEEKR2 model.

    rootdir : str
        The model's root directory.
    
    """
    bd_directory = os.path.join(
        root_directory, bd_anchor.directory)
    curdir = os.getcwd()
    os.chdir(bd_directory)
    add_atoms_filename = os.path.join(bd_directory, "add_atoms")

    vdws = []
    test_charges = []
    
    for atom in bd_engine.systemwide_settings.atomic_parameters:
        if atom.resname is None:
            atom.resname = "*"
        if atom.vdw is not None:
            string = "{:5s}{:>3s} {:>6.2f}\n".format(atom.type, atom.resname, atom.vdw)
            vdws.append(string)
        if atom.test_charge is not None:
            string = "{:5s}{:>3s} {:>6.2f}\n".format(atom.type, atom.resname, atom.test_charge)
            test_charges.append(string)

    with open(add_atoms_filename, "w") as add_atoms:
        
        add_atoms.write("GROUP = VdW\n")
        for vdw in vdws:
            add_atoms.write(vdw)
        add_atoms.write("END GROUP\n")

        add_atoms.write("\n")

        add_atoms.write("GROUP = TestCharges\n")
        for test_charge in test_charges:
            add_atoms.write(test_charge)    
        add_atoms.write("END GROUP\n")

    os.chdir(curdir)
    return

def make_sda_input(
        model: structures.Seekr_model,
        bd_anchor: structures.Anchor,
        root_directory: str,
        n_trajectories: int = 1,
        ) -> None:
    """
    Creates SDA input file with all input parameters set.
    If bd_directory is None, then b_surface is assumed.
    
    Parameters
    ----------
    model : Model()
        The SEEKR2 model.
        
    rootdir : str
        The model's root directory.
    """
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    bd_directory = os.path.join(
        root_directory, bd_anchor.directory)
    curdir = os.getcwd()
    os.chdir(bd_directory)
    sda_input = sda_sda_structures.Input()
    sda_input.nrun = n_trajectories
    sda_input.dseed = bd_engine.random_seed
    sda_input.solutes = []
    for molecule in bd_engine.systemwide_settings.molecules:
        sda_input.solutes.append(molecule.make_solute_object(bd_engine.apbs_grid_spacing))
    sda_input.total_grids = len(sda_input.solutes)

    total_solutes = 0
    for solute in sda_input.solutes:
        total_solutes += solute.solute_grid.nb_solute
    sda_input.total_solutes = total_solutes
    sda_input.solutes_path = "."

    sda_input.solvent_ions = []
    for model_ion in bd_engine.ions:
        bd_ion = sda_sda_structures.Ion()
        bd_ion.radius = model_ion.radius
        bd_ion.charge = model_ion.charge
        bd_ion.conc = model_ion.concentration
        sda_input.solvent_ions.append(bd_ion)

    # Make reaction windows
    neighbor_anchor_to_bd_index = bd_anchor.neighbor_anchor_indices[0]
    neighbor_anchor = model.anchors[neighbor_anchor_to_bd_index]
    assert len(neighbor_anchor.milestones) == 2, \
        "BD anchor must have exactly 2 milestones - other arrangements not yet implemented"
    milestone_values = []
    for milestone in neighbor_anchor.milestones:
        milestone_values.append(milestone.variables["avg_value"])
    lower_milestone_value = min(milestone_values)
    upper_milestone_value = max(milestone_values)
    sda_input.win0 = lower_milestone_value * 10.0  # Convert to Angstroms
    sda_input.dwin = (upper_milestone_value - lower_milestone_value) * 10.0  # Convert to Angstroms
    sda_input.nwin = 2  # 2 windows for BD anchor

    sda_input.write(os.path.join(bd_directory, sda_base.SDA_INPUT_BASENAME))
    os.chdir(curdir)
    return

def generate_bd_files(
        model: structures.Seekr_model,
        root_directory: str
        ) -> None:
    bd_engine = model.get_engine_by_scale_type("brownian_dynamics")
    assert isinstance(bd_engine, sda_structures.SDA_bd_engine), \
        "BD engine must be of type 'sda' to use this function."
    hydropro_directory = os.path.expanduser(bd_engine.hydropro_directory)
    assert bd_engine.systemwide_settings is not None, \
        "Systemwide settings must be set to generate BD files."
    bd_anchor = make_sda_reaction_file(model, root_directory)
    add_atoms_sda(bd_engine, bd_anchor, root_directory)
    make_sda_pdb_noh(model, bd_anchor, root_directory)
    run_hydropro(model, bd_anchor, root_directory, hydropro_directory)
    make_sda_grids(bd_engine, bd_anchor, root_directory,
                   model.temperature,
                   bd_engine.systemwide_settings.solvent_dielectric)
    make_sda_input(model, bd_anchor, root_directory)
    return