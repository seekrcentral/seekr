from __future__ import annotations
"""
modules/engines/sda/structures.py

Structures for the SDA engine - anything that might appear in the model.json.
"""

import os
import glob
import typing
import logging

from attrs import define, field, validators, Factory
from attrs.validators import instance_of, optional, gt, ge, and_, in_
import parmed

import seekr.modules.base as base
import seekr.modules.engines.sda.base as sda_base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.sda.sda_structures as sda_structures

logger = logging.getLogger(__name__)

@define
class SDA_molecule:
    """
    Refer to sda_structures.py for some good reference on documentation.
    """
    name: str = field(default="", 
                    validator=instance_of(str))
    type: str = field(default="protein",
                    validator=and_(instance_of(str), in_(
                        ["protein", "non_protein"])))
    pqr_filename: str = field(default="",
                            validator=instance_of(str))
    # Replace with alternative structure file
    #pqrxml_filename: str | None = field(default=None,
    #                        validator=optional(instance_of(str)))
    dime: int | None = field(
        default=None,
        validator=optional(and_(instance_of(int), ge(1))))
    pdb_noh_filename: str | None = field(
        default=None,
        validator=optional(instance_of(str)))
    original_indices: typing.Dict[str, typing.List[int]] = field(
        default=Factory(dict),)
    ghost_site_indices: typing.Dict[str, int] | None = field(
        default=None, validator=optional(instance_of(dict)))
    ghost_lines: typing.Dict[str, str] | None = field(
        default=None,validator=optional(instance_of(dict)))
    dielectric_constant: float = field(
        default=2.0, 
        validator=[instance_of(float), ge(1.0)])
    diffusion_trans: float = field(
        default=0.0, 
        validator=[instance_of(float), ge(0.0)])
    diffusion_rotat: float = field(
        default=0.0, 
        validator=[instance_of(float), ge(0.0)])
    real_net_charge: float = field(
        default=0.0, 
        validator=[instance_of(float)])
    surface_charge_dens: float = field(
        default=0.0, 
        validator=[instance_of(float)])
    rotate: bool = field(
        default=True, 
        validator=instance_of(bool))
    surface: bool = field(
        default=False, 
        validator=instance_of(bool))
    # Whether a conformational ensemble is used for the molecule
    #flex: bool = field(
    #    default=False, 
    #    validator=instance_of(bool))
    # Used when a surface is a metal
    image_charge: bool = field(
        default=False, 
        validator=instance_of(bool))
    dh_radius: float | None = field(
        default=None, 
        validator=optional(and_(instance_of(float), ge(0.0))))
    vol_radius: float | None = field(
        default=None, 
        validator=optional(and_(instance_of(float), ge(0.0))))
    # NOTE: These concern conformational changes and ensembles
    # list_conformation
    # total_conf
    # conformation_choice_method -> "method" in Solute_Grid() in 
    #     sda_structures.py
    # initial_conf
    # frequency
    #std_frequency
    epf: str | None = field(
        default=None, 
        validator=optional(instance_of(str)))
    qef: str | None = field(
        default=None, 
        validator=optional(instance_of(str)))
    edf: str | None = field(
        default=None, 
        validator=optional(instance_of(str)))
    hdf: str | None = field(
        default=None, 
        validator=optional(instance_of(str)))
    lj_repf: str | None = field(
        default=None, 
        validator=optional(instance_of(str)))
    
    def compute_grid_dime(
            self,
            apbs_grid_spacing: float = 0.5,
            ) -> None:
        """
        Compute the grid dimensions for the APBS grid based on the
        apbs_grid_spacing and the size of the molecule.
        """
        if self.pqr_filename == "":
            raise ValueError("PQR filename is not set for the molecule.")
        # Get the size of the molecule from the PQR file
        pqr_struct = parmed.load_file(self.pqr_filename, skip_bonds=True)
        min_coords = pqr_struct.coordinates.min(axis=0)
        max_coords = pqr_struct.coordinates.max(axis=0)
        min_grid = min_coords - 10.0  # 10 Angstroms padding
        max_grid = max_coords + 10.0  # 10 Angstroms padding
        grid_size = max_grid - min_grid
        max_grid_size = max(grid_size)
        exact_dime = max_grid_size / apbs_grid_spacing
        c = 1
        dime = 0
        MAX_ITER = 100
        while dime < exact_dime:
            # According to APBS manual, the arguments of dime are dependent 
            # on the choice of "nlev" by the formula n = c*2**(nlev+1) + 1, 
            # where the most common values are compatible with an nlev of 4. 
            # Hence the equation below.
            dime = c * 32 + 1
            c += 1
            if c > MAX_ITER:
                raise ValueError("Could not compute grid dimensions for the "
                                 "molecule. Maximum iterations reached.")
        self.dime = dime
        return

    def make_sda_grids(
            self,
            temperature: float,
            binary_directory: str,
            aux_directory: str,
            apbs_grid_spacing: float = 0.5,
            ions: typing.List[engines_structures.Ion] | None = None,
            solvent_dielectric: float = 78.5,
            ) -> None:
        """
        Create the grids used for this molecule in the SDA engine.
        """
        
        ionic_strength = 0.0
        bd_ions = []
        for model_ion in ions:
            bd_ion = sda_structures.Ion()
            bd_ion.radius = model_ion.radius
            bd_ion.charge = model_ion.charge
            bd_ion.conc = model_ion.concentration
            bd_ions.append(bd_ion)
            ionic_strength += model_ion.concentration*model_ion.charge**2 / 2

        # Make APBS grid
        apbs_input = sda_structures.APBS_Input(
            temperature=temperature,
            solute_pqr=self.pqr_filename,
            solute_name=self.name,
            solute_dielectric=self.dielectric_constant,
            solvent_dielectric=solvent_dielectric,
            apbs_grid_spacing=apbs_grid_spacing,
            apbs_grid_dime=self.dime,
            ions=bd_ions,
        )
        apbs_input_filename, apbs_output_grid_filename = apbs_input.write()
        apbs_output_filename = os.path.splitext(apbs_input_filename)[0] + ".out"
        run_apbs_command = f"apbs {apbs_input_filename} > {apbs_output_filename}"
        logger.info("Running APBS with command: " + run_apbs_command)
        os.system(run_apbs_command)
        assert os.path.exists(apbs_output_grid_filename), "Problem running " \
            "apbs - APBS UHBD file not found: " + apbs_output_grid_filename
        
        # Convert electrostatic potential energy to kcal
        electric_potential_filename = f"{self.name}_ep.grd"
        self.epf = electric_potential_filename
        sda_convert_bin = os.path.join(binary_directory, "convert_grid")
        run_sda_convert_command \
            = f"{sda_convert_bin} {apbs_output_grid_filename} " \
              f"{electric_potential_filename}  -scale 0.6 -convert > " \
              f"apbs_grid_conversion_{self.name}.out"
        logger.info("Running SDA convert grid with command: " \
                     + run_sda_convert_command)
        os.system(run_sda_convert_command)
        assert os.path.exists(electric_potential_filename), \
            "Problem running convert_grid - electric potential file not found: " \
            + electric_potential_filename
        # Make SDA grids and effective charges
        effective_charge_filename = f"{self.name}.echa"
        self.qef = effective_charge_filename
        effective_charge_input_filename = f"{self.name}_ecm.in"
        test_charge_filename = f"{self.name}.tcha"
        electric_desolvation_filename = f"{self.name}_ed.grd"
        self.edf = electric_desolvation_filename
        electric_desolvation_input_filename = f"{self.name}_ed.in"
        hydrophobic_desolvation_filename = f"{self.name}_hd.grd"
        self.hdf = hydrophobic_desolvation_filename
        hydrophobic_desolvation_input_filename = f"{self.name}_hd.in"
        if self.lj_repf is None:
            lennard_jones_repulsion_filename = ""
            lennard_jones_repulsion_input_filename = ""
        else:
            lennard_jones_repulsion_filename = f"{self.name}_ljrep.grd"
            lennard_jones_repulsion_input_filename = f"{self.name}_ljrep.in"

        sda_grid_inputs = sda_structures.SDA_grid_inputs(
            temperature=temperature,
            ionic_strength=ionic_strength,
            solute_name=self.name,
            solute_pdb=self.pdb_noh_filename,
            solvent_dielectric=solvent_dielectric,
            ecm_input=effective_charge_input_filename,
            tcha=test_charge_filename,
            echa=effective_charge_filename,
            epf=electric_potential_filename,
            edf_input=electric_desolvation_input_filename,
            hdf_input=hydrophobic_desolvation_input_filename,
            lj_repf_input=lennard_jones_repulsion_input_filename,
            edf=electric_desolvation_filename,
            hdf=hydrophobic_desolvation_filename,
            lj_repf=lennard_jones_repulsion_filename,
            dime=self.dime,
            grid_spacing=apbs_grid_spacing,
        )
        sda_grid_inputs.write()

        protein_surface_test_charge_command = None
        protein_surface_test_charge_command = None
        surface_test_charge_command = None
        test_charge_command = None
        print(f"HERE! self.name: {self.name}, self.type: {self.type}")
        if self.type == "non_protein":
            if self.surface:
                surface_test_charge_bin = os.path.join(
                    aux_directory,
                    "Kon-rates-SmallMolecule/Generate-ECMSites-SmallMol"\
                        "/ECM_surface.py")

                if os.path.exists(surface_test_charge_bin):
                    protein_surface_test_charge_command \
                        = f"python {protein_surface_test_charge_bin} "\
                          f"{self.pqr_filename} -threshold 3 "\
                          f"> tcha_{self.name}.out"
                else:
                    surface_test_charge_bin = os.path.join(
                        aux_directory, 
                        "Kon-rates-SmallMolecule/Generate-ECMSites-SmallMol/"\
                            "ECM_ligand.py")
                    surface_test_charge_command \
                        = f"python {protein_surface_test_charge_bin} "\
                          f"{self.pqr_filename} > tcha_{self.name}.out"
            else:
                test_charge_bin = os.path.join(
                    aux_directory, 
                    "Kon-rates-SmallMolecule/Generate-ECMSites-SmallMol/"\
                        "ECM_ligand.py")

                test_charge_command \
                    = f"python {test_charge_bin} {self.pqr_filename} "\
                      f"> tcha_{self.name}.out"
        elif self.type == "protein":
            test_charge_bin = os.path.join(binary_directory, "ecm_mksites")
            test_charge_command = f"{test_charge_bin} {self.pdb_noh_filename} "\
                +f"{test_charge_filename} > tcha_{self.name}.out"
            if self.surface:
                protein_surface_test_charge_bin = os.path.join(
                    aux_directory, 
                    "Kon-rates-SmallMolecule/Generate-ECMSites-SmallMol/"\
                        "ECM_surface.py")
                if os.path.exists(protein_surface_test_charge_bin):
                    protein_surface_test_charge_command \
                        = f"python {protein_surface_test_charge_bin} "\
                          f"{self.pqr_filename} -threshold 3"\
                         +f" > tcha_{self.name}.out"
                else:
                    protein_surface_test_charge_bin = os.path.join(
                        aux_directory, "Kon-rates-SmallMolecule/"\
                            "Generate-ECMSites-SmallMol/ECM_surface.py")
                    protein_surface_test_charge_command \
                        = f"python {protein_surface_test_charge_bin} "\
                          f"{self.pqr_filename} > tcha_{self.name}.out"
            
        if test_charge_command != None:
            logger.info("running command: " + test_charge_command)
            os.system(test_charge_command)
            # Check test charges
            if not os.path.exists(test_charge_filename):
                raise ValueError(
                    f"Problem running test charge command - "\
                    f"{test_charge_filename} not found. Check previous "\
                        "command output.")
            with open(test_charge_filename, 'r') as f:
                total_noncomment_lines = 0
                for line in f:
                    if not line.startswith("REMARK"):
                        total_noncomment_lines += 1
            if total_noncomment_lines == 0:
                raise ValueError(
                    "Problem running test charge command - No test charges "\
                    f"generated in {test_charge_filename}. Check that the "\
                    f"molecule named {self.name} is the correct type. Should "\
                    "it be 'non_protein'?")

        if surface_test_charge_command != None:
            logger.info("running command: " + surface_test_charge_command)
            os.system(surface_test_charge_command)
        if protein_surface_test_charge_command != None:
            logger.info("running command: " \
                         + protein_surface_test_charge_command)
            os.system(f"mv {test_charge_filename} tmp.tcha")
            os.system(protein_surface_test_charge_command)
            os.system(f"cat {self.pqr_filename.split('.')[0]}.tcha "\
                      ">> tmp.tcha")
            os.system(f"mv tmp.tcha {test_charge_filename}")
        effective_charge_bin = os.path.join(binary_directory, \
                                            "ecm_expand")
        effective_charge_command \
            = f"{effective_charge_bin} {effective_charge_input_filename}"\
             +f" > ecm_{self.name}.out"
        logger.info("running command: " + effective_charge_command)
        os.system(effective_charge_command)
        
        edhdlj_bin = os.path.join(binary_directory, "make_edhdlj_grid")
        if self.lj_repf is not None:
            edhdlj_command \
                = f"{edhdlj_bin} -ed {electric_desolvation_input_filename} "\
                 +f"-hd {hydrophobic_desolvation_input_filename} "\
                 +f"-ljrep {lennard_jones_repulsion_input_filename} "\
                 +f"> edhdlj_{self.name}.out"
        else:
            edhdlj_command \
                = f"{edhdlj_bin} -ed {electric_desolvation_input_filename} "\
                 +f"-hd {hydrophobic_desolvation_input_filename} "\
                 +f"> edhdlj_{self.name}.out"
        logger.info("running command: " + edhdlj_command)
        os.system(edhdlj_command)
        
        return

    def run_hydropro(
            self,
            temperature: float,
            hydropro_command: str,
        ) -> None:
        """
        Run hydropro for this molecule.
        """
        hydropro = sda_structures.Hydropro(
            temperature=temperature,
            pdbfile=self.pdb_noh_filename)
        hydropro_filename = hydropro_filename = "hydropro.dat"
        if self.surface:
            self.diffusion_trans = 0.0
            self.diffusion_rotat = 0.0
        else:
            if self.type == "non_protein":
                hydropro.write_input(hydropro_filename, small=True)
            else:
                hydropro.write_input(hydropro_filename, small=False)
        
            full_hydropro_command \
                = f"{hydropro_command} {hydropro_filename} {self.pdb_noh_filename}"\
                +f" > {self.name}_noh-hydropro.out"
            logger.info("Running hydropro with command: " + full_hydropro_command)
            os.system(full_hydropro_command)

            # Find and read hydropro output
            output_file = None
            for _root, _dirs, files in os.walk("."):
                for filename in files:
                    if self.name in filename and "res.txt" in filename:
                        output_file = filename

            assert output_file is not None, "Problem running hydropro - "\
                "output file not found."
            trans_diffusion, rot_diffusion = hydropro.read_output(output_file)
            self.diffusion_trans = trans_diffusion
            self.diffusion_rotat = rot_diffusion

        return
    
    def make_solute_object(
            self,
            apbs_grid_spacing: float,
            ) -> sda_structures.Solute:
        """
        Make a SDA Solute() object for this molecule.
        """
        solute_grid = sda_structures.Solute_Grid(
            pdb_filename=self.pdb_noh_filename,
            trans_diffusion=self.diffusion_trans,
            rot_diffusion=self.diffusion_rotat,
            real_net_charge=self.real_net_charge,
            surface_charge_dens=self.surface_charge_dens,
            rotate=int(self.rotate),
            surface=int(self.surface),
            image_charge=int(self.image_charge),
            dh_radius=self.dh_radius,
            vol_radius=self.vol_radius,
            epf=self.epf,
            qef=self.qef,
            edf=self.edf,
            hdf=self.hdf,
            lj_repf=self.lj_repf,
        )

        solute = sda_structures.Solute(
            type=self.type,
            apbs_grid_dime=self.dime,
            apbs_grid_spacing=apbs_grid_spacing,
            dielectric= self.dielectric_constant,
            solute_grid=solute_grid,
            pqr_filename=self.pqr_filename)
        
        return solute
        

@define
class SDA_atomic_parameter():
    """
    An SDA atomic descriptions

    Attributes:
    --------------------
    vdw: float or None
        VdW radius of the atom.
    test_charge: float or None
        Test charge of the
    resname: string or None, Default None
        Name of the residue where the atom belongs
    type: string or none, Default None
        Atom name
    """
    vdw: typing.Optional[float] = field(default=None,
                      validator=optional(and_(instance_of(float), ge(0.0))))
    test_charge: typing.Optional[float] = field(default=None,
                               validator=optional(instance_of(float)))
    resname: typing.Optional[str] = field(default=None,
                                          validator=optional(instance_of(str)))
    type: typing.Optional[str] = field(default=None,
                                          validator=optional(instance_of(str)))

@define
class SDA_unpartitioned_settings:
    # To be used for a common set of parameters for all anchors
    # including bulk anchor, within BD simulations
    molecules: list[SDA_molecule] = field(default=Factory(list),
                    validator=validators.deep_iterable(
                        member_validator=instance_of(
                            SDA_molecule),
                        iterable_validator=instance_of(list),
                    ))
    atomic_parameters: list[SDA_atomic_parameter] = field(
        default=Factory(list),)
    solvent_dielectric: float = field(
        default=78.5, 
        validator=[instance_of(float), ge(1.0)])
    stage_parameters: dict[str, dict | None] = field(default=Factory(dict),)

@define
class SDA_bd_engine(engines_structures.Base_engine):
    type: typing.Literal["sda"] = "sda"
    scale_type: typing.Literal["brownian_dynamics"] = "brownian_dynamics"
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
                            validator=validators.deep_iterable(
                                member_validator=instance_of(
                                    engines_structures.Ion),
                                iterable_validator=instance_of(list),
                            ))
    b_surface_directory: str = field(default="b_surface",
                                    validator=instance_of(str))
    results_glob: str = field(default=sda_base.SDA_RESULTS_GLOB,
                            validator=instance_of(str))
    unpartitioned_settings: SDA_unpartitioned_settings | None = None
    random_seed: int = field(default=0,
                             validator=[instance_of(int), ge(0)])
    geom_type: str = field(default="sphere",
                            validator=in_(
                                ["sphere", "box", "nambox"]))
    xmin: float | None = field(default=None,
                            validator=optional(instance_of(float)))
    xmax: float | None = field(default=None,
                            validator=optional(instance_of(float)))
    ymin: float | None = field(default=None,
                            validator=optional(instance_of(float)))
    ymax: float | None = field(default=None,
                            validator=optional(instance_of(float)))
    zmin: float | None = field(default=None,
                            validator=optional(instance_of(float)))
    zmax: float | None = field(default=None,
                            validator=optional(instance_of(float)))

    
    def get_bd_transition_counts(
            self, 
            root_directory: str,
            set_name: str
            ) -> dict:
        """
        Get the transition counts from the BD results files.
        """
        output_file_glob = os.path.join(
            root_directory, "anchor_sets", set_name, 
            self.b_surface_directory, sda_base.SDA_RESULTS_GLOB)
        output_file_list = glob.glob(output_file_glob)
        output_file_list = base.order_files_numerically(output_file_list)
        bd_transition_counts = {}
        if len(output_file_list) > 0:
            transition_counts = sda_base.compute_transition_counts(output_file_list)
            bd_transition_counts["b_surface"] = transition_counts
        return bd_transition_counts
    
    def modify_bd_transition_counts(
            self, 
            bd_transition_counts: dict, 
            lower_milestone_index: int,
            upper_milestone_index: int,
            ) -> None:
        """
        Modify the BD transition counts in the model.
        """
        new_bd_transition_counts = {}
        bsurface_bd_transition_counts = {}
        milestone_values = []
        if "b_surface" not in bd_transition_counts:
            return new_bd_transition_counts

        for key, value in bd_transition_counts["b_surface"].items():
            if isinstance(key, float):
                milestone_values.append(key)

            else:
                bsurface_bd_transition_counts[key] = value

        lower_milestone_value = min(milestone_values)
        upper_milestone_value = max(milestone_values)
        bsurface_bd_transition_counts[lower_milestone_index] \
            = bd_transition_counts["b_surface"][lower_milestone_value]
        bsurface_bd_transition_counts[upper_milestone_index] \
            = bd_transition_counts["b_surface"][upper_milestone_value]
        
        new_bd_transition_counts["b_surface"] = bsurface_bd_transition_counts

        return new_bd_transition_counts
    
    def get_random_seed(
            self
            ) -> int:
        """
        Get the random seed for the Browndye engine.
        """
        return self.random_seed

    def set_random_seed(
            self, 
            seed: int
            ) -> None:
        """
        Set the random seed for the Browndye engine.
        """
        self.random_seed = seed
        return
    
    def can_handle_toy(
            self
            ) -> bool:
        """
        Can this engine handle toy systems?
        """
        return False
    
    def save_selections(
            self, 
            directory: str, 
            ) -> None:
        """
        Save the selections to files.
        """
        # Does nothing for now - there are no selections to save for SDA
        return
    
    def load_selections(
            self,
            directory: str,
            ) -> None:
        """
        Load the selections from a directory.
        """
        # Does nothing for now - there are no selections to load for SDA
        return
    
    def get_selection_names(
            self,
            ) -> list[str]:
        """
        Get the names of all selections.
        """
        selection_names = []
        # Does nothing for now
        return selection_names
