"""
create_seekr_input.py

Create SEEKR input objects for use in tests.
"""

import os

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.structures as structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.browndye.preparer as browdye_preparer
import seekr.modules.sampling_methods.preparer as sampling_preparer
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.preparer as preparer


TEST_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.abspath(os.path.join(TEST_DIRECTORY, "..", "data"))
HOSTGUEST_DIRECTORY = os.path.join(DATA_DIRECTORY, "hostguest_files")

def assign_amber_params(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | preparer.Anchor_input, 
        prmtop_filename: str, 
        pdb_filename: str
        ) -> None:
    anchor_or_systemwide.md_parameters_topology\
        = engines_structures.Amber_parameters_topology()
    anchor_or_systemwide.md_parameters_topology.prmtop_filename \
        = prmtop_filename
    anchor_or_systemwide.positions = pdb_filename
    return

def assign_forcefield_params(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | preparer.Anchor_input, 
        built_in_ff_list: list[str], 
        custom_ff_list: list[str], 
        pdb_filename: str
        ) -> None:
    anchor_or_systemwide.md_parameters_topology\
        = engines_structures.Forcefield_parameters()
    anchor_or_systemwide.md_parameters_topology.built_in_forcefield_filenames \
        = built_in_ff_list
    anchor_or_systemwide.md_parameters_topology.custom_forcefield_filenames \
        = custom_ff_list
    anchor_or_systemwide.positions = pdb_filename = pdb_filename
    return

def assign_system_params(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | preparer.Anchor_input, 
        system_filename: str, 
        pdb_filename: str
        ) -> None:
    anchor_or_systemwide.md_parameters_topology\
        = engines_structures.Openmm_system()
    anchor_or_systemwide.md_parameters_topology.system_filename \
        = system_filename
    anchor_or_systemwide.positions = pdb_filename = pdb_filename
    return

def assign_charmm_params(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings | preparer.Anchor_input, 
        psf_filename: str, 
        charmm_ff_filenames: list[str], 
        pdb_filename: str
        ) -> None:
    anchor_or_systemwide.md_parameters_topology\
        = engines_structures.Charmm_parameters_topology()
    anchor_or_systemwide.md_parameters_topology.psf_filename = psf_filename
    anchor_or_systemwide.md_parameters_topology.param_filename_list = charmm_ff_filenames
    anchor_or_systemwide.positions = pdb_filename = pdb_filename
    return

def create_host_guest_example_input(
        root_dir: str, 
        bd: bool = True, 
        ff: str = "amber",
        start_from_systemwide: bool = False,
        parameters_to_all_anchors: bool = False
        ) -> preparer.Seekr_input:
    """
    Create an example input object for loading and testing in SEEKR.
    """
    os.chdir(TEST_DIRECTORY)
    amber_prmtop_filename = os.path.join(HOSTGUEST_DIRECTORY, "hostguest.parm7")
    forcefield_built_in_ff_list = ["amber14/tip3pfb.xml"]
    forcefield_custom_ff_list = [os.path.join(HOSTGUEST_DIRECTORY, "hostguest.xml")]
    system_filename = os.path.join(HOSTGUEST_DIRECTORY, "hostguest_system.xml")
    pdb_filenames = ["hostguest_at0.5.pdb",
                     "hostguest_at1.5.pdb",
                     "hostguest_at2.5.pdb",
                     "hostguest_at3.5.pdb",
                     "hostguest_at4.5.pdb",
                     "hostguest_at5.5.pdb",
                     "hostguest_at6.5.pdb",
                     "hostguest_at7.5.pdb",
                     "hostguest_at8.5.pdb",
                     "hostguest_at9.5.pdb",
                     "hostguest_at10.5.pdb",
                     "hostguest_at11.5.pdb",
                     "hostguest_at12.5.pdb",
                     ""]
    pdb_full_filenames = []
    for pdb_filename in pdb_filenames:
        if pdb_filename:
            pdb_full_filenames.append(os.path.join(HOSTGUEST_DIRECTORY, pdb_filename))
        else:
            pdb_full_filenames.append("")

    values_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                   1.05, 1.15, 1.25, 1.35]
    receptor_indices_md = list(range(147))
    ligand_indices_md = list(range(147, 162))
    receptor_indices_bd = list(range(147))
    ligand_indices_bd = list(range(15))
    
    seekr_input = preparer.Seekr_input()
    seekr_input.directory = root_dir
    seekr_input.temperature = 298.15
    seekr_input.md_settings = openmm_preparer.Openmm_settings_input()
    seekr_input.md_settings.systemwide_settings \
        = engines_structures.MD_systemwide_settings()
    if start_from_systemwide:
        if ff == "amber":
            assign_amber_params(seekr_input.md_settings.systemwide_settings, 
                                amber_prmtop_filename, pdb_filename)
        elif ff == "forcefield":
            assign_forcefield_params(seekr_input.md_settings.systemwide_settings, 
                                     forcefield_built_in_ff_list, forcefield_custom_ff_list, 
                                     pdb_filename)
        elif ff == "system":
            assign_system_params(seekr_input.md_settings.systemwide_settings, 
                                 system_filename, pdb_filename)
        else:
            raise Exception(f"ff option not implemented: {ff}.")
    
    receptor_selection_md = engines_structures.Selection()
    receptor_selection_md.name = "host"
    receptor_selection_md.atom_indices = receptor_indices_md
    ligand_selection_md = engines_structures.Selection()
    ligand_selection_md.name = "guest"
    ligand_selection_md.atom_indices = ligand_indices_md
    seekr_input.md_settings.systemwide_settings.selections\
        = [receptor_selection_md, ligand_selection_md]
    if bd:
        seekr_input.bd_settings = browdye_preparer.Browndye_settings_input()
        seekr_input.bd_settings.apbs_grid_spacing = 2.0
        ion1 = engines_structures.Ion()
        ion1.name = "Cl-"
        ion1.radius = 1.2
        ion1.charge = -1.0
        ion1.concentration = 0.0
        ion2 = base.Ion()
        ion2.name = "Na+"
        ion2.radius = 0.9
        ion2.charge = 1.0
        ion2.concentration = 0.0
        seekr_input.bd_settings.ions = [ion1, ion2]
        receptor_selection_bd = base.Selection()
        receptor_selection_bd.name = "host"
        receptor_selection_bd.atom_indices = receptor_indices_bd
        ligand_selection_bd = base.Selection()
        ligand_selection_bd.name = "guest"
        ligand_selection_bd.atom_indices = ligand_indices_bd
        receptor_molecule = browdye_preparer.Browndye_molecule_input()
        receptor_molecule.name = "beta-cyclodextrin"
        receptor_molecule.pqr_filename = os.path.join(
            HOSTGUEST_DIRECTORY, "hostguest_receptor.pqr")
        receptor_molecule.selections = [receptor_selection_bd]
        ligand_molecule = browdye_preparer.Browndye_molecule_input()
        ligand_molecule.name = "1-butanol"
        ligand_molecule.pqr_filename = os.path.join(
            HOSTGUEST_DIRECTORY, "hostguest_ligand.pqr")
        ligand_molecule.selections = [ligand_selection_bd]
        seekr_input.bd_settings.molecules = [receptor_molecule, ligand_molecule]
    else:
        seekr_input.bd_settings = None
    
    stage1 = preparer.Input_stage_md()
    stage1.name = "equilibration"
    stage1.run_minimization = True
    stage1.checkpoint_interval = 1000
    stage1.ensemble = structures.NVT_ensemble()
    stage1.sampling_method = sampling_preparer.Conventional_md_input()
    stage1.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage1.completion_criteria.number_of_steps = 10000
    
    stage2 = preparer.Input_stage_md()
    stage2.name = "metadynamics"
    stage2.run_minimization = False
    stage2.checkpoint_interval = 1000
    stage2.ensemble = structures.NVT_ensemble()
    stage2.sampling_method = sampling_preparer.Metadynamics_md_input()
    stage2.sampling_method.gaussian_height = 1.0
    stage2.completion_criteria = compl_crit_struct.All_anchors_filled_completion_criteria()

    stage3 = preparer.Input_stage_md()
    stage3.name = "MMVT"
    stage3.run_minimization = False
    stage3.checkpoint_interval = 1000
    stage3.ensemble = structures.NVT_ensemble()
    stage3.sampling_method = sampling_preparer.MMVT_md_input()
    stage3.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage3.completion_criteria.number_of_steps = 10000

    stage4 = preparer.Input_stage_bd()
    stage4.name = "bd"
    stage4.n_trajectories_per_output = 1000
    stage4.n_steps_per_output = 100000
    stage4.sampling_method = sampling_preparer.NAM_bd_input()
    stage4.completion_criteria = compl_crit_struct.Surface_crossed_completion_criteria()
    
    seekr_input.stages = [stage1, stage2, stage3, stage4]
    seekr_input.partitions = [preparer.Partition_input()]
    distance_cv_input = distance_cv.Distance_cv_input()
    distance_cv_input.group1_name = "host"
    distance_cv_input.group2_name = "guest"
    seekr_input.partitions[0].cvs = [distance_cv_input]
    
    bound_state_point = preparer.State_point()
    bound_state_point.name = "bound"
    bound_state_point.value = [0.05]
    bulk_state_point = preparer.State_point()
    bulk_state_point.name = "bulk"
    bulk_state_point.value = [1.35]
    seekr_input.partitions[0].state_points = [bound_state_point, bulk_state_point]
    
    seekr_input.partitions[0].anchors = []
    for i, (value, pdb_filename) in enumerate(zip(values_list, pdb_filenames)):
        input_anchor = preparer.Anchor_input()
        input_anchor.value = [value]
        if pdb_filename == "":
            pdb_filename_full = ""
        else:
            pdb_filename_full = os.path.join(HOSTGUEST_DIRECTORY, pdb_filename)
        
        if not start_from_systemwide:
            if parameters_to_all_anchors:
                if ff == "amber":
                    assign_amber_params(input_anchor, amber_prmtop_filename, pdb_filename_full)
                elif ff == "forcefield":
                    assign_forcefield_params(input_anchor, forcefield_built_in_ff_list, 
                                             forcefield_custom_ff_list, pdb_filename_full)
                elif ff == "system":
                    assign_system_params(input_anchor, system_filename, pdb_filename_full)
                else:
                    raise Exception(f"ff option not implemented: {ff}.")
            else:
                input_anchor.positions = pdb_filename_full
        input_anchor.positions = ""
        seekr_input.partitions[0].anchors.append(input_anchor)
    
    if bd:
        seekr_input.partitions[0].anchors[-1].bulk_anchor = True

    return seekr_input