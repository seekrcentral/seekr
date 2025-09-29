"""
This script produces input objects that will be used to
make in input file that can be read and used to create
a SEEKR calculation model.
"""

import os
import json
import argparse

from cattrs import unstructure

import seekr.modules.base as base
import seekr.modules.collective_variables.distance_cv as distance_cv
import seekr.modules.collective_variables.external_cv as external_cv
import seekr.modules.collective_variables.rmsd_cv as rmsd_cv
import seekr.modules.engines.structures as engines_structures
import seekr.modules.completion_criteria.structures as compl_crit_struct
import seekr.modules.restraints.preparer as restraints_preparer
import seekr.modules.reporters.preparer as reporters_preparer
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.browndye.preparer as browndye_preparer
import seekr.modules.engines.sda.preparer as sda_preparer
import seekr.modules.sampling_methods.preparer as sampling_preparer
import seekr.modules.preparer as preparer
import seekr.modules.scales.molecular_dynamics.preparer as md_preparer
import seekr.modules.scales.brownian_dynamics.preparer as bd_preparer

TOOLS_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.abspath(os.path.join(TOOLS_DIRECTORY, "..", "data"))
HOSTGUEST_DIRECTORY = os.path.join(DATA_DIRECTORY, "hostguest_files")

def create_toy_model_input(
        root_dir: str,
        use_smd: bool = False,
        use_2cvs: bool = False
    ) -> preparer.Seekr_input:
    """
    Create a toy model input.
    
    This is the Entropy barrier system.
    """
    if use_2cvs:
        values_list = [[0.1, -0.7], [-0.1, -0.5], [0.1, -0.3], [-0.1, -0.1], [0.1, 0.1], [-0.1, 0.3], [0.1, 0.5], [-0.1, 0.7]]
        positions_list = [[[0.0, value[1], 0.0]] for value in values_list]

    else:
        values_list = [[-0.7], [-0.5], [-0.3], [-0.1], [0.1], [0.3], [0.5], [0.7]]
        positions_list = [[[0.0, value[0], 0.0]] for value in values_list]

    seekr_input = preparer.Seekr_input()
    seekr_input.directory = root_dir
    seekr_input.temperature = 298.15
    seekr_input.random_seed = 111
    md_settings = openmm_preparer.Openmm_settings_input()
    md_settings.platform_type = "reference"
    md_settings.systemwide_settings\
        = engines_preparer.MD_systemwide_settings_input()
    selection = engines_structures.Selection()
    selection.name = "particle0"
    selection.atom_indices = [0]
    md_settings.systemwide_settings.selections\
        = [selection]
    md_settings.systemwide_settings.positions\
        = [[[0.0, -0.7, 0.0]]]
    seekr_input.settings_list = [md_settings]

    stage1 = md_preparer.Input_stage_md()
    stage1.name = "equilibration"
    stage1.run_minimization = False
    stage1.checkpoint_interval = 10
    stage1.ensemble = base.NVT_ensemble()
    stage1.sampling_method = sampling_preparer.Conventional_md_input()
    stage1.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage1.completion_criteria.number_of_steps = 10000
    stage1_state_reporter = reporters_preparer.State_reporter_input()
    stage1_state_reporter.to_stdout = True
    stage1_state_reporter.interval = 100
    stage1_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage1_dcd_reporter.interval = 100
    stage1_dcd_reporter.enforce_periodic_box = False
    stage1.reporters = [stage1_state_reporter, stage1_dcd_reporter]

    stage2 = md_preparer.Input_stage_md()
    if use_smd:
        stage2.name = "steered"
        stage2.sampling_method = sampling_preparer.Steered_md_input()
        if use_2cvs:
            stage2.sampling_method.cv_names = ["progress_coordinate"]
        else:
            stage2.sampling_method.cv_names = ["progress_coordinate"]
        stage2.sampling_method.force_constant = 100000.0
        stage2.sampling_method.velocity = 1.0
    else:
        stage2.name = "metadynamics"
        stage2.sampling_method = sampling_preparer.Metadynamics_md_input()
        stage2.sampling_method.gaussian_height = 1.0
        if use_2cvs:
            stage2.sampling_method.cv_names = ["progress_coordinate"]
        else:
            stage2.sampling_method.cv_names = ["progress_coordinate"]
    
    stage2.run_minimization = False
    stage2.checkpoint_interval = 10
    stage2.ensemble = base.NVT_ensemble()
    stage2.completion_criteria = compl_crit_struct.Anchors_filled_completion_criteria()
    stage2.completion_criteria.exclude = [2]
    stage2.completion_criteria.eval_interval = 10
    stage2_state_reporter = reporters_preparer.State_reporter_input()
    stage2_state_reporter.to_stdout = True
    stage2_state_reporter.interval = 50
    stage2_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage2_dcd_reporter.interval = 500
    stage2_dcd_reporter.enforce_periodic_box = False
    stage2.reporters = [stage2_dcd_reporter] #[stage2_state_reporter, stage2_dcd_reporter]

    stage3 = md_preparer.Input_stage_md()
    stage3.name = "MMVT"
    stage3.run_minimization = False
    stage3.checkpoint_interval = 5000
    stage3.ensemble = base.NVT_ensemble()
    stage3.sampling_method = sampling_preparer.MMVT_md_input()
    stage3.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage3.completion_criteria.number_of_steps = 2000000
    stage3_state_reporter = reporters_preparer.State_reporter_input()
    stage3_state_reporter.to_stdout = True
    stage3_state_reporter.interval = 5000
    stage3_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage3_dcd_reporter.interval = 5000
    stage3_dcd_reporter.enforce_periodic_box = False
    stage3.reporters = [stage3_state_reporter, stage3_dcd_reporter]

    seekr_input.stages = [stage1, stage2, stage3]
    seekr_input.partitions = [preparer.Partition_input()]
    external_cv_input = external_cv.External_cv_input()
    external_cv_input.name = "progress_coordinate"
    external_cv_input.expression = "y1"
    external_cv_input.group_names = ["particle0"]
    external_cv_input.min_value = -1.0
    external_cv_input.max_value = 1.0
    
    
    if use_2cvs:
        external_cv_input2 = external_cv.External_cv_input()
        external_cv_input2.name = "additional_coordinate"
        external_cv_input2.expression = "x1"
        external_cv_input2.group_names = ["particle0"]
        external_cv_input2.min_value = -1.0
        external_cv_input2.max_value = 1.0
        seekr_input.partitions[0].cvs = [external_cv_input2, external_cv_input]
        
    else:
        seekr_input.partitions[0].cvs = [external_cv_input]
    
    state_point1 = preparer.State_point()
    state_point1.name = "stateA"
    if use_2cvs:
        state_point1.value = [0.0, -0.7]
    else:
        state_point1.value = [-0.7]
    state_point1.end_state = True
    state_point2 = preparer.State_point()
    state_point2.name = "stateB"
    if use_2cvs:
        state_point2.value = [0.0, 0.7]
    else:
        state_point2.value = [0.7]
    state_point2.end_state = True
    seekr_input.partitions[0].state_points = [state_point1, state_point2]
    
    seekr_input.partitions[0].anchors = []
    for i, (value, position) in enumerate(zip(values_list, positions_list)):
        input_anchor = preparer.Anchor_input()
        input_anchor.value = value
        #input_anchor.positions = [position]
        input_anchor.positions = ""

        seekr_input.partitions[0].anchors.append(input_anchor)

    seekr_input.toy_settings = preparer.Toy_settings_input()
    seekr_input.toy_settings.potential_energy_expression \
        = "5*(x1^6+y1^6+exp(-(10*y1)^2)*(1-exp(-(10*x1)^2)))"
    seekr_input.toy_settings.num_particles = 1
    seekr_input.toy_settings.masses = [10.0]

    return seekr_input

def create_host_guest_example_input(
        root_dir: str, 
        bd: bool = True, 
        ff: str = "amber",
        use_rmsd_cv: bool = False,
        use_sda: bool = False,
        use_smd: bool = False
        ) -> preparer.Seekr_input:
    """
    Create an example input object for loading and testing in SEEKR.
    """
    amber_prmtop_filename = os.path.join(HOSTGUEST_DIRECTORY, "hostguest.parm7")
    forcefield_built_in_ff_list = ["amber14/tip3pfb.xml"]
    forcefield_custom_ff_list = ["hostguest.xml"]
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
    ligand_noh_indices_md = [147, 150, 152, 155, 158]
    receptor_indices_bd = list(range(147))
    ligand_indices_bd = list(range(15))
    o5_group_indices = [15, 36, 57, 78, 99, 120, 141]
    single_group_index = [15]
    
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
    
    receptor_selection_md = engines_structures.Selection()
    receptor_selection_md.name = "host"
    receptor_selection_md.atom_indices = receptor_indices_md
    ligand_selection_md = engines_structures.Selection()
    ligand_selection_md.name = "guest"
    ligand_selection_md.atom_indices = ligand_indices_md
    ligand_noh_selection_md = engines_structures.Selection()
    ligand_noh_selection_md.name = "guest_noH"
    ligand_noh_selection_md.atom_indices = ligand_noh_indices_md
    solute_selection_md = engines_structures.Selection()
    solute_selection_md.name = "solute"
    solute_selection_md.atom_indices = receptor_indices_md + ligand_indices_md
    o5_group = engines_structures.Selection()
    o5_group.name = "o5_group"
    o5_group.atom_indices = o5_group_indices
    single_group = engines_structures.Selection()
    single_group.name = "single_index"
    single_group.atom_indices = single_group_index
    md_settings.systemwide_settings.selections\
        = [receptor_selection_md, ligand_selection_md, ligand_noh_selection_md, 
           solute_selection_md, o5_group, single_group]
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
        if use_sda:
            bd_settings = sda_preparer.SDA_settings_input()
            bd_settings.binary_directory = "~/Downloads/sda_flex-7.4.0/bin"
            bd_settings.aux_directory = "~/Downloads/sda_flex-7.4.0/auxi"
            bd_settings.hydropro_directory = "~/Downloads/hydropro10"
            bd_settings.apbs_grid_spacing = 2.0
            bd_settings.ions = [ion1, ion2]
            receptor_molecule = sda_preparer.SDA_molecule_input()
            receptor_molecule.name = "beta-cyclodextrin"
            receptor_molecule.type = "non_protein"
            receptor_molecule.pqr_filename = os.path.join(
                HOSTGUEST_DIRECTORY, "hostguest_receptor.pqr")
            receptor_molecule.selections = [receptor_selection_bd]
            ligand_molecule = sda_preparer.SDA_molecule_input()
            ligand_molecule.name = "1-butanol"
            ligand_molecule.type = "non_protein"
            ligand_molecule.pqr_filename = os.path.join(
                HOSTGUEST_DIRECTORY, "hostguest_ligand.pqr")
            ligand_molecule.selections = [ligand_selection_bd]
            bd_settings.molecules = [receptor_molecule, ligand_molecule]
        else:
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
    else:
        pass

    stage1 = md_preparer.Input_stage_md()
    stage1.name = "equilibration"
    stage1.run_minimization = True
    stage1.checkpoint_interval = 1000
    stage1.ensemble = base.NVT_ensemble()
    stage1.sampling_method = sampling_preparer.Conventional_md_input()
    stage1.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage1.completion_criteria.number_of_steps = 100000
    pairwise_restraint_input = restraints_preparer.Pairwise_restraint_input()
    pairwise_restraint_input.selection_name = "o5_group"
    stage1.restraints = [pairwise_restraint_input]
    stage1_state_reporter = reporters_preparer.State_reporter_input()
    stage1_state_reporter.to_stdout = True
    stage1_state_reporter.interval = 1000
    stage1_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage1_dcd_reporter.interval = 1000
    stage1_dcd_reporter.atom_subset = None
    stage1.reporters = [stage1_state_reporter, stage1_dcd_reporter]
    
    stage2 = md_preparer.Input_stage_md()
    if use_smd:
        stage2.name = "steered"
        stage2.sampling_method = sampling_preparer.Steered_md_input()
        stage2.sampling_method.velocity = 10.0
        stage2.sampling_method.force_constant = 100000.0
        if use_rmsd_cv:
            stage2.sampling_method.cv_names = ["rmsd_host_guest"]
        else:
            stage2.sampling_method.cv_names = ["distance_host_guest"]
    else:
        stage2.name = "metadynamics"
        stage2.sampling_method = sampling_preparer.Metadynamics_md_input()
        stage2.sampling_method.gaussian_height = 1.0
        if use_rmsd_cv:
            stage2.sampling_method.cv_names = ["rmsd_host_guest"]
        else:
            stage2.sampling_method.cv_names = ["distance_host_guest"]

    stage2.run_minimization = False
    stage2.checkpoint_interval = 1000
    stage2.ensemble = base.NVT_ensemble()
    #stage2.completion_criteria = compl_crit_struct.Anchors_filled_completion_criteria()
    stage2.completion_criteria = compl_crit_struct.Number_of_steps_completion_criteria()
    stage2.completion_criteria.number_of_steps = 10000000
    
    proximity_restraint_input = restraints_preparer.Proximity_restraint_input()
    proximity_restraint_input.selection1_name = "host"
    proximity_restraint_input.selection2_name = "guest"
    proximity_restraint_input.radius = 1.05
    cylinder_restraint_input = restraints_preparer.Cylinder_restraint_input()
    cylinder_restraint_input.selection1_name = "guest"
    cylinder_restraint_input.selection2_name = "host"
    cylinder_restraint_input.selection3_name = "single_index"
    cylinder_restraint_input.radius = 0.4
    stage2.restraints = [proximity_restraint_input, cylinder_restraint_input]
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
    positional_restraint = restraints_preparer.Positional_restraint_input()
    positional_restraint.selection_name = "host"
    stage3.restraints = [positional_restraint]
    stage3_state_reporter = reporters_preparer.State_reporter_input()
    stage3_state_reporter.to_stdout = True
    stage3_state_reporter.interval = 250
    stage3_dcd_reporter = reporters_preparer.DCD_reporter_input()
    stage3_dcd_reporter.interval = 250
    stage3_dcd_reporter.atom_subset = "solute"
    stage3.reporters = [stage3_state_reporter, stage3_dcd_reporter]

    seekr_input.stages = [stage1, stage2, stage3]

    if bd:
        stage4 = bd_preparer.Input_stage_bd()
        stage4.name = "bd"
        stage4.n_trajectories_per_output = 1000
        stage4.n_steps_per_output = 100000
        stage4.sampling_method = sampling_preparer.NAM_bd_input()
        #stage4.completion_criteria = compl_crit_struct.Number_of_trajectories_completion_criteria()
        #stage4.completion_criteria.number_of_trajectories = 10000
        stage4.completion_criteria = compl_crit_struct.Number_of_encounters_completion_criteria()
        stage4.completion_criteria.number_of_encounters = 1000
        seekr_input.stages.append(stage4)
    
    seekr_input.partitions = [preparer.Partition_input()]
    if use_rmsd_cv:
        rmsd_cv_input = rmsd_cv.RMSD_cv_input()
        rmsd_cv_input.name = "rmsd_host_guest"
        rmsd_cv_input.group_rmsd_name = "guest_noH"
        rmsd_cv_input.group_align_name = "o5_group"
        rmsd_cv_input.ref_structure = os.path.join(
            HOSTGUEST_DIRECTORY, "hostguest_at0.5.pdb")
        seekr_input.partitions[0].cvs = [rmsd_cv_input]
    else:
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
    
    seekr_input.partitions[0].anchors = []
    for i, (value, pdb_filename) in enumerate(zip(values_list, pdb_filenames)):
        input_anchor = preparer.Anchor_input()
        input_anchor.value = [value]
        if pdb_filename == "":
            pdb_filename_full = ""
        else:
            pdb_filename_full = os.path.join(HOSTGUEST_DIRECTORY, pdb_filename)
        input_anchor.positions = ""
        seekr_input.partitions[0].anchors.append(input_anchor)
    
    if bd:
        seekr_input.partitions[0].anchors[-1].bulk_anchor = True

    return seekr_input
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    #argparser.add_argument(
    #    "input_file", metavar="INPUT_FILE", type=str, 
    #    help="The name of the input JSON file to generate.")
    argparser.add_argument(
        "-t", "--toy_mode", dest="toy_mode", default=False, 
        help="Toggle toy mode: whether a sample toy system is generated.", 
        action="store_true")
    argparser.add_argument(
        "-r", "--rmsd_cv", dest="rmsd_cv", default=False, 
        help="Use an RMSD CV instead of a distance CV.", 
        action="store_true")
    argparser.add_argument(
        "-s", "--smd", dest="smd", default=False, 
        help="Use steered MD instead of metadynamics.", 
        action="store_true")
    argparser.add_argument(
        "-S", "--sda", dest="sda", default=False, 
        help="Use SDA for BD instead of Browndye.", 
        action="store_true")
    argparser.add_argument(
        "-2", "--two_cvs", dest="two_cvs", default=False, 
        help="Toggle 2-CVs mode: whether to use two CVs.", 
        action="store_true")
    argspace = argparser.parse_args()
    args = vars(argspace)
    #json_filename = args["input_file"]
    toy_mode = args["toy_mode"]
    use_rmsd_cv = args["rmsd_cv"]
    use_sda = args["sda"]
    use_smd = args["smd"]
    use_2cvs = args["two_cvs"]

    if use_rmsd_cv:
        bd = False
    else:
        bd = True

    if toy_mode:
        if use_smd:
            sampling_suffix = "smd"
        else:
            sampling_suffix = "metad"
        if use_2cvs:
            cvs_suffix = "2cvs_"
        else:
            cvs_suffix = ""
        root_dir = "~/tmp/seekr3_test_toy_" + cvs_suffix + sampling_suffix
        json_filename = "toy_input_" + cvs_suffix + sampling_suffix + ".json"
        seekr_input = create_toy_model_input(root_dir, use_smd, use_2cvs=use_2cvs)
    else:
        assert not use_2cvs, "2-CV mode not implemented for real molecular system."
        if bd:
            if use_sda:
                bd_suffix = "sda"
            else:
                bd_suffix = "browndye"
        else:
            bd_suffix = "noBD"
            
        if use_smd:
            sampling_suffix = "smd"
        else:
            sampling_suffix = "metad"
        
        if use_rmsd_cv:
            cv_suffix = "rmsdCV"
        else:
            cv_suffix = "distCV"
            
        root_dir = "~/tmp/seekr3_test_" + bd_suffix + "_" + sampling_suffix \
            + "_" + cv_suffix
        json_filename = "hostguest_input_" + bd_suffix + "_" + sampling_suffix \
            + "_" + cv_suffix + ".json"
        seekr_input = create_host_guest_example_input(root_dir, bd=bd, 
                                                      use_rmsd_cv=use_rmsd_cv,
                                                      use_sda=use_sda, use_smd=use_smd)
    
    seekr_input_dict = unstructure(seekr_input)
    seekr_input_dict = base.wrap_noindent_integers(seekr_input_dict)
    json_dump = json.dumps(seekr_input_dict, indent=4, cls=base.NoIndentListEncoder)
    print("Writing input file:", json_filename)
    with open(json_filename, "w") as file:
        file.write(json_dump)
