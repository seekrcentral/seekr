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
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.sampling_methods.preparer as sampling_preparer
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.scales.molecular_dynamics.preparer as md_preparer
import seekr.modules.scales.brownian_dynamics.preparer as bd_preparer
import seekr.modules.anchor_schemes.preparer as anchor_scheme_preparer
import seekr.modules.preparer as preparer


TEST_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.abspath(os.path.join(TEST_DIRECTORY, "..", "data"))
HOSTGUEST_DIRECTORY = os.path.join(DATA_DIRECTORY, "hostguest_files")

def assign_amber_params(
        anchor_or_systemwide: engines_structures.MD_systemwide_settings, 
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
        anchor_or_systemwide: engines_structures.MD_systemwide_settings, 
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
        anchor_or_systemwide: engines_structures.MD_systemwide_settings, 
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
        anchor_or_systemwide: engines_structures.MD_systemwide_settings, 
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
    seekr_input.random_seed = 111
    md_settings = openmm_preparer.Openmm_settings_input()
    md_settings.systemwide_settings \
        = engines_preparer.MD_systemwide_settings_input()
    if ff == "amber":
        md_settings.systemwide_settings.md_parameters_topology\
            = engines_structures.Amber_parameters_topology()
        md_settings.systemwide_settings.md_parameters_topology\
            .prmtop_filename = amber_prmtop_filename
        # To test starting from a single position
        md_settings.systemwide_settings.positions\
            = os.path.join(HOSTGUEST_DIRECTORY, "hostguest_at0.5.pdb")
    else:
        raise Exception(f"ff option not implemented: {ff}.")
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
    solute_selection_md = engines_structures.Selection()
    solute_selection_md.name = "solute"
    solute_selection_md.atom_indices = receptor_indices_md + ligand_indices_md
    md_settings.systemwide_settings.selections\
        = [receptor_selection_md, ligand_selection_md, solute_selection_md]
    seekr_input.settings_list = [md_settings]
    if bd:
        ion1 = engines_structures.Ion()
        ion1.name = "Cl-"
        ion1.radius = 1.2
        ion1.charge = -1.0
        ion1.concentration = 0.0
        ion2 = engines_structures.Ion()
        ion2.name = "Na+"
        ion2.radius = 0.9
        ion2.charge = 1.0
        ion2.concentration = 0.0
        receptor_selection_bd = engines_structures.Selection()
        receptor_selection_bd.name = "host"
        receptor_selection_bd.atom_indices = receptor_indices_bd
        ligand_selection_bd = engines_structures.Selection()
        ligand_selection_bd.name = "guest"
        ligand_selection_bd.atom_indices = ligand_indices_bd
        bd_settings = browndye_preparer.Browndye_settings_input()
        bd_settings.apbs_grid_spacing = 2.0
        bd_settings.ions = [ion1, ion2]
        receptor_molecule = browndye_preparer.Browndye_molecule_input()
        receptor_molecule.name = "beta-cyclodextrin"
        receptor_molecule.pqr_filename = os.path.join(
            HOSTGUEST_DIRECTORY, "hostguest_receptor.pqr")
        receptor_molecule.selections = [receptor_selection_bd]
        ligand_molecule = browndye_preparer.Browndye_molecule_input()
        ligand_molecule.name = "1-butanol"
        ligand_molecule.pqr_filename = os.path.join(
            HOSTGUEST_DIRECTORY, "hostguest_ligand.pqr")
        ligand_molecule.selections = [ligand_selection_bd]
        bd_settings.molecules = [receptor_molecule, ligand_molecule]
        seekr_input.settings_list.append(bd_settings)
    
    stage1 = md_preparer.Input_stage_md()
    stage1.name = "equilibration"
    stage1.run_minimization = True
    stage1.checkpoint_interval = 1000
    stage1.ensemble = base.NVT_ensemble()
    stage1.sampling_method = sampling_preparer.Conventional_md_input()
    stage1.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage1.completion_criteria.number_of_steps = 100000
    stage1_state_reporter = reporters_preparer.State_reporter_input()
    stage1_state_reporter.to_stdout = True
    stage1_state_reporter.interval = 1000
    stage1_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage1_dcd_reporter.interval = 1000
    stage1_dcd_reporter.atom_subset = None
    stage1.reporters = [stage1_state_reporter, stage1_dcd_reporter]
    
    stage2 = md_preparer.Input_stage_md()
    stage2.name = "metadynamics"
    stage2.sampling_method = sampling_preparer.Metadynamics_md_input()
    stage2.sampling_method.gaussian_height = 1.0
    stage2.sampling_method.cv_names = ["distance_host_guest"]
    stage2.run_minimization = False
    stage2.checkpoint_interval = 1000
    stage2.ensemble = base.NVT_ensemble()
    stage2.completion_criteria = compl_crit_struct.Anchors_filled_completion_criteria()
    stage2_state_reporter = reporters_preparer.State_reporter_input()
    stage2_state_reporter.to_stdout = True
    stage2_state_reporter.interval = 500
    stage2_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage2_dcd_reporter.interval = 500
    stage2_dcd_reporter.atom_subset = "solute"
    stage2.reporters = [stage2_state_reporter, stage2_dcd_reporter]

    stage3 = md_preparer.Input_stage_md()
    stage3.name = "MMVT"
    stage3.run_minimization = False
    stage3.checkpoint_interval = 1000
    stage3.ensemble = base.NVT_ensemble()
    stage3.sampling_method = sampling_preparer.MMVT_md_input()
    stage3.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage3.completion_criteria.number_of_steps = 100000
    #positional_restraint = restraints_preparer.Positional_restraint_input()
    #positional_restraint.selection_name = "host"
    #stage3.restraints = [positional_restraint]
    stage3_state_reporter = reporters_preparer.State_reporter_input()
    stage3_state_reporter.to_stdout = True
    stage3_state_reporter.interval = 250
    stage3_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage3_dcd_reporter.interval = 250
    stage3_dcd_reporter.atom_subset = "solute"
    stage3.reporters = [stage3_state_reporter, stage3_dcd_reporter]

    stage4 = bd_preparer.Input_stage_bd()
    stage4.name = "bd"
    stage4.n_trajectories_per_output = 1000
    stage4.n_steps_per_output = 100000
    stage4.sampling_method = sampling_preparer.NAM_bd_input()
    stage4.completion_criteria = compl_crit_struct.Number_of_trajectories_completion_criteria()
    stage4.completion_criteria.number_of_trajectories = 10000
    
    seekr_input.stages = [stage1, stage2, stage3, stage4]

    seekr_input.partitions = [preparer.Partition_input()]
    distance_cv_input = distance_cv.Distance_cv_input()
    distance_cv_input.name = "distance_host_guest"
    distance_cv_input.group1_name = "host"
    distance_cv_input.group2_name = "guest"
    seekr_input.partitions[0].cvs = [distance_cv_input]
    
    bound_state_point = preparer.State_point()
    bound_state_point.name = "bound"
    bound_state_point.value = [0.05]
    bound_state_point.end_state = True
    bulk_state_point = preparer.State_point()
    bulk_state_point.name = "bulk"
    bulk_state_point.value = [1.35]
    bulk_state_point.end_state = True
    seekr_input.partitions[0].state_points = [bound_state_point, bulk_state_point]
    
    seekr_input.partitions[0].anchor_scheme \
        = anchor_scheme_preparer.Anchors_as_voronoi_cells_anchor_scheme()
    seekr_input.partitions[0].anchor_scheme.voronoi_cells = []
    for i, (value, pdb_filename) in enumerate(zip(values_list, pdb_filenames)):
        voronoi_cell = anchor_scheme_preparer.Voronoi_cell()
        voronoi_cell.value = [value]
        if pdb_filename == "":
            pdb_filename_full = ""
        else:
            pdb_filename_full = os.path.join(HOSTGUEST_DIRECTORY, pdb_filename)
        voronoi_cell.positions = ""
        seekr_input.partitions[0].anchor_scheme.voronoi_cells.append(voronoi_cell)
    
    if bd:
        seekr_input.partitions[0].anchor_scheme.voronoi_cells[-1].bulk_anchor = True

    return seekr_input