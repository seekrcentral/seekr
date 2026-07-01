"""
tools/convert_from_seekr2.py

Take an existing SEEKR2 model and calculation, and convert it to a
SEEKR3 model and calculation. The model.xml file will be converted to
an equivalent model.json, and the data and file structure will also be
converted over to work within the SEEKR3 framework.
"""

import os
import sys
import glob
import shutil
import logging
import argparse

import seekr2.modules.common_base as base2
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base2
import seekr2.modules.mmvt_cvs.mmvt_spherical_cv as mmvt_spherical_cv2
import seekr2.modules.mmvt_cvs.mmvt_rmsd_cv as mmvt_rmsd_cv2
import seekr2.modules.mmvt_cvs.mmvt_external_cv as mmvt_external_cv2
import seekr2.modules.mmvt_cvs.mmvt_voronoi_cv as mmvt_voronoi_cv2
import seekr2.modules.mmvt_analyze as mmvt_analyze2
import seekr2.modules.common_converge as common_converge2

import seekr.modules.base as base3
import seekr.modules.structures as structures3
import seekr.modules.engines.structures as engine_structures3
import seekr.modules.engines.openmm.structures as openmm_structures3
import seekr.modules.engines.browndye.structures as browndye_structures3
import seekr.modules.scales.molecular_dynamics.structures as md_scale3
import seekr.modules.scales.brownian_dynamics.structures as bd_scale3
import seekr.modules.sampling_methods.structures as sampl_meth_struct3
import seekr.modules.collective_variables.distance_cv as distance_cv3
import seekr.modules.collective_variables.external_cv as external_cv3
import seekr.modules.collective_variables.rmsd_cv as rmsd_cv3
import seekr.modules.completion_criteria.structures as compl_crit_struct3
import seekr.modules.reporters.structures as reporters_struct3
import seekr.modules.scales.logistic.anchor_starting_structures_from_reporter_logistic_stage \
    as logistic_struct3
import seekr.modules.preparer as preparer3

logger = logging.getLogger(__name__)

NUM_WINDOWS = 100

def convert_mmvt_output_files(
        anchor2,
        input_glob: str,
        output_file: str,
        timestep: float,
        max_step_list: list[int],
        time_this_anchor: float
    ) -> None:
    input_files = base2.order_files_numerically(glob.glob(input_glob))
    alpha = anchor2.index
    anchor_stats = mmvt_analyze2.MMVT_anchor_statistics(alpha)
    min_time = None
    max_time = None
    #print("input_files", input_files)
    anchor_stats.read_output_file_list(
        "openmm", input_files, min_time, max_time, anchor2,
        timestep)
    #print("alpha:", alpha)
    #print("anchor_stats.N_i_j_alpha", anchor_stats.N_i_j_alpha)
    #print("anchor_stats.N_alpha_beta", anchor_stats.N_alpha_beta)
    #print("anchor_stats.R_i_alpha_total", anchor_stats.R_i_alpha_total)
    #print("anchor_stats.T_alpha_total", anchor_stats.T_alpha_total)
    #exit()
    
    last_max_time = 0.0
    #attr_dict = {}
    step_dict = {}

    for window in range(NUM_WINDOWS):
        list_index = int((window + 1) / NUM_WINDOWS * len(max_step_list)) - 1
        #print("list_index", list_index, "of ", len(max_step_list) - 1)
        max_step = max_step_list[list_index]
        max_time = max_step * timestep
        anchor_stats.read_output_file_list(
            "openmm", input_files, min_time, max_time, anchor2,
            timestep)
        aliases = list(anchor_stats.R_i_alpha_total.keys())
        # All ordered (alias_i, alias_j) pairs
        all_pairs: list[tuple[int, int]] = [
            (ai, aj) for ai in aliases for aj in aliases
        ]
        
        counts = {(ai, aj): anchor_stats.N_i_j_alpha.get((ai, aj), 0) for (ai, aj) in all_pairs}
        sum_counts = {ai: sum(counts[(ai, aj)] for aj in aliases if ai != aj) for ai in aliases}
        for ai in aliases:
            counts[(ai, ai)] = anchor_stats.N_alpha_beta.get(ai, 0) - sum_counts[ai]
        times = anchor_stats.R_i_alpha_total
        last_alias = 1
        last_time = anchor_stats.T_alpha_total
        output_block = f"""step_count: {max_step}
time_count: {max_time}
first_encounter_time: 0.0
last_milestone_alias_touched: {last_alias}
last_milestone_time_touched: {last_time}
"""
        for (ai, aj) in all_pairs:
            output_block += f"milestone_{ai}_{aj}_count: {counts[(ai, aj)]}\n"
        for ai in aliases:
            output_block += f"milestone_{ai}_time: {times[ai]}\n"
        output_block += "####################\n"

        for key in list(step_dict):
            if key >= max_step:
                del step_dict[key]

        step_dict[max_step] = output_block
                
        """
        # Remove any previously stored entry whose step, time, or any
        # cumulative milestone count/time exceeds the current window's
        # values. Cumulative MMVT stats must be monotonically
        # non-decreasing across windows, so an existing entry with a
        # value larger than the current one is inconsistent and the
        # whole entry should be discarded.
        for key in list(attr_dict):
            entry = attr_dict[key]
            should_remove = False
            if key >= max_step:
                should_remove = True
            elif entry["time_count"] > max_time:
                should_remove = True
            else:
                for (ai, aj) in all_pairs:
                    if f"milestone_{ai}_{aj}_count" in entry:
                        if entry[f"milestone_{ai}_{aj}_count"] > counts[(ai, aj)]:
                            should_remove = True
                            break
                if not should_remove:
                    for ai in aliases:
                        if f"milestone_{ai}_time" in entry:
                            if entry[f"milestone_{ai}_time"] > times[ai]:
                                should_remove = True
                                break
            if should_remove:
                del attr_dict[key]
                continue
        
        attr_dict[max_step] = {}
        attr_dict[max_step]["output_block"] = output_block
        attr_dict[max_step]["time_count"] = max_time
        for (ai, aj) in all_pairs:
            attr_dict[max_step][f"milestone_{ai}_{aj}_count"] \
                = counts[(ai, aj)]
        for ai in aliases:
            attr_dict[max_step][f"milestone_{ai}_time"] = times[ai]
        """
    
    first_window = True
    for key in sorted(step_dict.keys()):
        if first_window:
            symbol = "w"
        else:
            symbol = "a"
        with open(output_file, symbol) as f:
            f.write(step_dict[key])

        first_window = False
        
    return

def convert_from_seekr2(
        seekr2_model: base2.Model, 
        seekr3_output_dir: str,
        pdb_file: str | None = None,
        metadyn_height: float = 1.0,
        skip_bd: bool = False,
        #assign_from_all_anchors: bool = False
        ) -> None:
    """Convert a SEEKR2 model and calculation to SEEKR3 format.

    Parameters
    ----------
    seekr2_model : base2.Model
        The SEEKR2 model object.
    seekr3_output_dir : str
        Path to the output directory where the SEEKR3 model and data will be written.
    metadyn_height : float, optional
        The initial heights of the metadynamics Gaussians. Default is 1.0 kJ/mol.
    assign_from_all_anchors : bool, optional
        Whether to assign parameter files from all SEEKR2 anchors to all SEEKR3 anchors.
        Default is False.
    """
    # Make the filetree
    os.makedirs(seekr3_output_dir, exist_ok=True)
    unpartitioned_scope_dir = os.path.join(seekr3_output_dir, base3.UNPARTITIONED_DIRECTORY)
    os.makedirs(unpartitioned_scope_dir, exist_ok=True)

    model2 = seekr2_model
    model2_full_dir = os.path.abspath(model2.anchor_rootdir)
    model3 = structures3.Seekr_model()
    model3.directory = os.path.abspath(seekr3_output_dir)
    model3.temperature = model2.temperature
    # Model engines
    if isinstance(model2.calculation_settings, mmvt_cv_base2.MMVT_settings):
        scale3 = md_scale3.Scale_md()
        scale3.engine = "openmm"
        model3.scales.append(scale3)
        engine3 = openmm_structures3.Openmm_md_engine()
        engine3.nonbonded_method = model2.openmm_settings.nonbonded_method
        engine3.nonbonded_cutoff = model2.openmm_settings.nonbonded_cutoff
        engine3.constraints = model2.openmm_settings.constraints
        engine3.hydrogen_mass = model2.openmm_settings.hydrogenMass
        engine3.initial_temperature = model2.openmm_settings.initial_temperature
        if model2.openmm_settings.langevin_integrator.integrator_type == "langevin":
            engine3.integrator_settings = \
                engine_structures3.Langevin_integrator_settings()
        if model2.openmm_settings.langevin_integrator.integrator_type == "langevinMiddle":
            engine3.integrator_settings = \
                engine_structures3.Langevin_middle_integrator_settings()
            
        engine3.integrator_settings.timestep \
            = model2.openmm_settings.langevin_integrator.timestep
        engine3.integrator_settings.friction \
            = model2.openmm_settings.langevin_integrator.friction_coefficient
        engine3.integrator_settings.target_temperature \
            = model2.openmm_settings.langevin_integrator.target_temperature
        engine3.integrator_settings.random_seed \
            = model2.openmm_settings.langevin_integrator.random_seed
        engine3.integrator_settings.rigid_tolerance \
            = model2.openmm_settings.langevin_integrator.rigid_tolerance
        
        if model2.openmm_settings.cuda_platform_settings is not None:
            engine3.platform_settings = engine_structures3.Cuda_platform_settings()
            engine3.platform_settings.device_index \
                = model2.openmm_settings.cuda_platform_settings.cuda_device_index
            engine3.platform_settings.precision \
                = model2.openmm_settings.cuda_platform_settings.cuda_precision
        else: # Reference platform
            engine3.platform_settings = None

        if model2.toy_settings is not None:
            engine3.toy_settings = engine_structures3.Toy_settings()
            engine3.toy_settings.potential_energy_expression \
                = model2.toy_settings.potential_energy_expression
            engine3.toy_settings.num_particles = model2.toy_settings.num_particles
            engine3.toy_settings.masses = model2.toy_settings.masses

        metadyn_glob = os.path.join(
            model2_full_dir, "anchor_*", "building", "hidr_traj_metadyn.*")
        smd_glob = os.path.join(
            model2_full_dir, "anchor_*", "building", "hidr_traj_smd.*")
        
        
        if len(glob.glob(metadyn_glob)) > 0:
            hidr_method = "metad"
        elif len(glob.glob(smd_glob)) > 0:
            hidr_method = "smd"
        else:
            hidr_method = None

        if hidr_method == "metad":
            hidr_stage_name = "metadynamics"
            hidr_stage_params = {
                "index": 1,
                "directory": "stage_1_metadynamics",
                "positions": "metadynamics"
            }
            anchor_2_metadyn_traj_glob = os.path.join(
                model2_full_dir, "anchor_*", "building", "hidr_traj_metadyn.*")
            anchor_2_hidr_traj_files = glob.glob(anchor_2_metadyn_traj_glob)

        elif hidr_method == "smd":
            hidr_stage_name = "steered"
            hidr_stage_params = {
                "index": 1,
                "directory": "stage_1_steered",
                "positions": "steered"
            }
            anchor_2_smd_traj_glob = os.path.join(
                model2_full_dir, "anchor_*", "building", "hidr_traj_smd.*")
            anchor_2_hidr_traj_files = glob.glob(anchor_2_smd_traj_glob)
            assert len(anchor_2_hidr_traj_files) > 0, f"No SMD trajectory files found matching glob: {anchor_2_smd_traj_glob}"

        engine3.unpartitioned_settings = engine_structures3.MD_unpartitioned_settings()
        engine3.unpartitioned_settings.stage_parameters = {
            "initial": {
                "index": 0,
                "directory": "stage_0_initial",
                "positions": "positions"
            },
        }
        if hidr_method is not None:
            engine3.unpartitioned_settings.stage_parameters[hidr_stage_name] \
                = hidr_stage_params

        for stage_param, stage_attrs in engine3.unpartitioned_settings.stage_parameters.items():
            stage_dir = os.path.join(unpartitioned_scope_dir, stage_attrs["directory"])
            os.makedirs(stage_dir, exist_ok=True)

        initial_stage_dir = os.path.join(
            unpartitioned_scope_dir, 
            engine3.unpartitioned_settings.stage_parameters["initial"]["directory"])
        anchor_0_building_full_dir = os.path.join(
            model2_full_dir, model2.anchors[0].directory, "building")

        if model2.anchors[0].amber_params is not None:
            engine3.unpartitioned_settings.md_parameters_topology \
                = engine_structures3.Amber_parameters_topology()
            engine3.unpartitioned_settings.md_parameters_topology.prmtop_filename \
                = model2.anchors[0].amber_params.prmtop_filename
            
            # Copy over the prmtop file from model2.anchors[0] directory to initial_stage_dir
            prmtop_basename = os.path.basename(
                engine3.unpartitioned_settings.md_parameters_topology.prmtop_filename)
            prmtop_src = os.path.join(anchor_0_building_full_dir, prmtop_basename)
            prmtop_dest = os.path.join(initial_stage_dir, prmtop_basename)
            shutil.copy(prmtop_src, prmtop_dest)

        elif model2.anchors[0].charmm_params is not None:
            engine3.unpartitioned_settings.md_parameters_topology \
                = engine_structures3.Charmm_parameters_topology()
            engine3.unpartitioned_settings.md_parameters_topology.psf_filename \
                = model2.anchors[0].charmm_params.psf_filename
            engine3.unpartitioned_settings.md_parameters_topology.param_filename_list \
                = model2.anchors[0].charmm_params.charmm_ff_files

            # Copy over the psf and parameter files from model2.anchors[0] directory 
            # to initial_stage_dir
            psf_basename = os.path.basename(
                engine3.unpartitioned_settings.md_parameters_topology.psf_filename)
            psf_src = os.path.join(anchor_0_building_full_dir, psf_basename)
            psf_dest = os.path.join(initial_stage_dir, psf_basename)
            shutil.copy(psf_src, psf_dest)

            for param_filename in engine3.unpartitioned_settings.md_parameters_topology.param_filename_list:
                param_basename = os.path.basename(param_filename)
                param_src = os.path.join(anchor_0_building_full_dir, param_basename)
                param_dest = os.path.join(initial_stage_dir, param_basename)
                shutil.copy(param_src, param_dest)
        
        elif model2.anchors[0].forcefield_params is not None:
            if model2.anchors[0].forcefield_params.system_filename == "":
                engine3.unpartitioned_settings.md_parameters_topology \
                    = engine_structures3.Forcefield_parameters()
                engine3.unpartitioned_settings.md_parameters_topology\
                    .built_in_forcefield_filenames = model2.anchors[0]\
                    .forcefield_params.built_in_forcefield_filenames
                engine3.unpartitioned_settings.md_parameters_topology\
                    .custom_forcefield_filenames = model2.anchors[0]\
                    .forcefield_params.custom_forcefield_filenames
                
                # Copy over the custom forcefield files from model2.anchors[0] directory to initial_stage_dir
                for ff_filename in engine3.unpartitioned_settings\
                        .md_parameters_topology.custom_forcefield_filenames:
                    ff_basename = os.path.basename(ff_filename)
                    ff_src = os.path.join(anchor_0_building_full_dir, ff_basename)
                    ff_dest = os.path.join(initial_stage_dir, ff_basename)
                    shutil.copy(ff_src, ff_dest)
            else:
                engine3.unpartitioned_settings.md_parameters_topology \
                    = engine_structures3.Openmm_system()
                engine3.unpartitioned_settings.md_parameters_topology.system_filename \
                    = model2.anchors[0].forcefield_params.system_filename
                
                # Copy over the system file from model2.anchors[0] directory to initial_stage_dir
                system_basename = os.path.basename(
                    engine3.unpartitioned_settings.md_parameters_topology\
                        .system_filename)
                system_src = os.path.join(anchor_0_building_full_dir, system_basename)
                system_dest = os.path.join(initial_stage_dir, system_basename)
                shutil.copy(system_src, system_dest)
                
        else:
            raise NotImplementedError(
                "Currently only Amber, Charmm, and OpenMM System parameters are supported for conversion.")

        # Copy over the PDB file, if provided, to the initial_stage_dir
        if pdb_file is not None:
            pdb_basename = "positions.pdb"
            pdb_src = pdb_file
            pdb_dest = os.path.join(initial_stage_dir, pdb_basename)
            if os.path.exists(pdb_src):
                shutil.copy(pdb_src, pdb_dest)

        if hidr_method is not None:
            # Copy over the hidr traj files from model2 to the metadynamics stage directory
            for hidr_traj_file in anchor_2_hidr_traj_files:
                #hidr_traj_basename = os.path.basename(hidr_traj_file)
                hidr_traj_extension = os.path.splitext(hidr_traj_file)[1]
                hidr_traj_basename = f"{hidr_stage_name}1.swarm_0.position_reporter_1.{hidr_traj_extension}"
                hidr_traj_src = hidr_traj_file
                hidr_traj_dest = os.path.join(
                    unpartitioned_scope_dir, 
                    engine3.unpartitioned_settings.stage_parameters[hidr_stage_name]["directory"],
                    hidr_traj_basename)
                shutil.copy(hidr_traj_src, hidr_traj_dest)

        model3.engines.append(engine3)

        doing_bd = False
        if (model2.browndye_settings is not None) and (not skip_bd):
            assert model2.k_on_info is not None, "k_on_info must be provided for BrownDye settings."
            bd_scale3_obj = bd_scale3.Scale_bd()
            bd_scale3_obj.engine = "browndye"
            model3.scales.append(bd_scale3_obj)
            
            bd_engine3 = browndye_structures3.Browndye_bd_engine()
            bd_engine3.binary_directory = model2.browndye_settings.browndye_bin_dir
            bd_engine3.apbs_grid_spacing = model2.browndye_settings.apbs_grid_spacing
            bd_engine3.ions = []
            if model2.k_on_info.ions is not None:
                for i, ion2 in enumerate(model2.k_on_info.ions):
                    ion3 = engine_structures3.Ion()
                    ion3.name = f"ion_{i}"
                    ion3.charge = ion2.charge
                    ion3.concentration = ion2.conc
                    ion3.radius = ion2.radius
                    bd_engine3.ions.append(ion3)
            bd_engine3.debye_length = float(model2.browndye_settings.debye_length)
            bd_engine3.random_seed = 0
            bd_engine3.b_surface_directory = model2.k_on_info.b_surface_directory
            bd_engine3.results_glob = model2.k_on_info.bd_output_glob
            bd_engine3.unpartitioned_settings \
                = browndye_structures3.Browndye_unpartitioned_settings()
            bd_engine3.unpartitioned_settings.molecules = []
            bd_mol_lig = browndye_structures3.Browndye_molecule()
            bd_mol_lig.name = "ligand"
            bd_mol_lig.pqr_filename = model2.browndye_settings.ligand_pqr_filename
            bd_mol_lig.pqrxml_filename = bd_mol_lig.pqr_filename.replace(".pqr", ".xml")
            bd_mol_lig.original_indices = {"lig":
                model2.k_on_info.bd_milestones[0].ligand_indices}
            bd_mol_lig.ghost_site_indices = {"lig": model2.browndye_settings.ghost_indices_lig[0]}
            bd_mol_rec = browndye_structures3.Browndye_molecule()
            bd_mol_rec.name = "receptor"
            bd_mol_rec.pqr_filename = model2.browndye_settings.receptor_pqr_filename
            bd_mol_rec.pqrxml_filename = bd_mol_rec.pqr_filename.replace(".pqr", ".xml")
            bd_mol_rec.original_indices = {"rec":
                model2.k_on_info.bd_milestones[0].receptor_indices}
            bd_mol_rec.ghost_site_indices = {"rec": model2.browndye_settings.ghost_indices_rec[0]}
            bd_engine3.unpartitioned_settings.molecules.append(bd_mol_lig)
            bd_engine3.unpartitioned_settings.molecules.append(bd_mol_rec)
            # Assume stage 4 is BD
            bd_engine3.unpartitioned_settings.stage_parameters["bd"] = {"index": 4}
            model3.engines.append(bd_engine3)
            doing_bd = True
        
        # Figure out CVs
        cv_names = []
        for cv_index, cv2 in enumerate(model2.collective_variables):
            if isinstance(cv2, mmvt_spherical_cv2.MMVT_spherical_CV):
                cv3 = distance_cv3.Distance_CV()
                cv3.name = cv2.name
                cv3.index = cv_index
                cv3.group1_name = "rec"
                cv3.group2_name = "lig"

                engine3.unpartitioned_settings.selections = []
                rec_selection = engine_structures3.Selection()
                rec_selection.name = "rec"
                rec_selection.atom_indices = cv2.group1
                lig_selection = engine_structures3.Selection()
                lig_selection.name = "lig"
                lig_selection.atom_indices = cv2.group2
                engine3.unpartitioned_settings.selections.append(rec_selection)
                engine3.unpartitioned_settings.selections.append(lig_selection)
                cv_names.append(cv3.name)
                model3.collective_variables.append(cv3)

            elif isinstance(cv2, mmvt_rmsd_cv2.MMVT_RMSD_CV):
                cv3 = rmsd_cv3.RMSD_CV()
                cv3.name = cv2.name
                cv3.index = cv_index
                cv3.group_rmsd_name = "rmsd_group"
                cv3.group_align_name = "align_group"
                cv3.ref_structure = cv2.ref_structure

                engine3.unpartitioned_settings.selections = []
                rmsd_selection = engine_structures3.Selection()
                rmsd_selection.name = "rmsd_group"
                rmsd_selection.atom_indices = cv2.group
                align_selection = engine_structures3.Selection()
                align_selection.name = "align_group"
                align_selection.atom_indices = cv2.align_group
                engine3.unpartitioned_settings.selections.append(rmsd_selection)
                engine3.unpartitioned_settings.selections.append(align_selection)
                cv_names.append(cv3.name)
                model3.collective_variables.append(cv3)

            elif isinstance(cv2, mmvt_external_cv2.MMVT_external_CV):
                cv3 = external_cv3.External_CV()
                cv3.name = cv2.name
                cv3.index = cv_index
                engine3.unpartitioned_settings.selections = []
                for i, group in enumerate(cv2.groups):
                    selection = engine_structures3.Selection()
                    selection.name = f"group_{i}"
                    selection.atom_indices = group
                    engine3.unpartitioned_settings.selections.append(selection)

                cv_names.append(cv3.name)
                model3.collective_variables.append(cv3)
            
            elif isinstance(cv2, mmvt_voronoi_cv2.MMVT_Voronoi_CV):
                for child_index, child_cv in enumerate(cv2.child_cvs):
                    if isinstance(child_cv, mmvt_spherical_cv2.MMVT_spherical_CV):
                        cv3 = distance_cv3.Distance_CV()
                        cv3.name = child_cv.name
                        cv3.index = child_index
                        cv3.group1_name = "rec"
                        cv3.group2_name = "lig"

                        engine3.unpartitioned_settings.selections = []
                        rec_selection = engine_structures3.Selection()
                        rec_selection.name = "rec"
                        rec_selection.atom_indices = child_cv.group1
                        lig_selection = engine_structures3.Selection()
                        lig_selection.name = "lig"
                        lig_selection.atom_indices = child_cv.group2
                        engine3.unpartitioned_settings.selections.append(rec_selection)
                        engine3.unpartitioned_settings.selections.append(lig_selection)
                        cv_names.append(cv3.name)

                    elif isinstance(child_cv, mmvt_rmsd_cv2.MMVT_RMSD_CV):
                        cv3 = rmsd_cv3.RMSD_CV()
                        cv3.name = child_cv.name
                        cv3.index = child_index
                        cv3.group_rmsd_name = "rmsd_group"
                        cv3.group_align_name = "align_group"
                        cv3.ref_structure = child_cv.ref_structure

                        engine3.unpartitioned_settings.selections = []
                        rmsd_selection = engine_structures3.Selection()
                        rmsd_selection.name = "rmsd_group"
                        rmsd_selection.atom_indices = child_cv.group
                        align_selection = engine_structures3.Selection()
                        align_selection.name = "align_group"
                        align_selection.atom_indices = child_cv.align_group
                        engine3.unpartitioned_settings.selections.append(rmsd_selection)
                        engine3.unpartitioned_settings.selections.append(align_selection)
                        cv_names.append(cv3.name)

                    elif isinstance(child_cv, mmvt_external_cv2.MMVT_external_CV):
                        cv3 = external_cv3.External_CV()
                        cv3.name = child_cv.name
                        cv3.index = child_index
                        engine3.unpartitioned_settings.selections = []
                        for i, group in enumerate(child_cv.groups):
                            selection = engine_structures3.Selection()
                            selection.name = f"group_{i}"
                            selection.atom_indices = group
                            engine3.unpartitioned_settings.selections.append(selection)

                    cv_names.append(cv3.name)
                    model3.collective_variables.append(cv3)
            else:
                raise NotImplementedError(
                    f"CV type '{cv2.name}' not currently supported for conversion.")
            
        # Assigned to individual stages?
        if model2.openmm_settings.barostat is not None:
            ensemble = base3.NPT_ensemble()
        else:
            ensemble = base3.NVT_ensemble()

        # HIDR stage
        model3.stages = []
        
        if hidr_method is None:
            logistic_stage_index = 1
            mmvt_stage_index = 2
        else:
            hidr_stage = md_scale3.Stage_md()
            hidr_stage.index = 1
            hidr_stage.name = hidr_stage_name
            hidr_stage.run_minimization = False
            hidr_stage.md_restart_checkpoint_interval = 250
            hidr_stage.ensemble = ensemble
            if hidr_stage_name == "metadynamics":
                hidr_stage.sampling_method \
                    = sampl_meth_struct3.Metadynamics_md_sampling_method()
                hidr_stage.sampling_method.cv_names.append(cv_names[0])
                hidr_stage.sampling_method.number_of_points = [101]
                hidr_stage.sampling_method.bias_factor = 10.0
                hidr_stage.sampling_method.gaussian_height = metadyn_height
                hidr_stage.sampling_method.gaussian_width = [0.1]
                hidr_stage.sampling_method.steps_per_update = 250
                hidr_stage.sampling_method.write_biases = False
                
            elif hidr_stage_name == "steered":
                hidr_stage.sampling_method \
                    = sampl_meth_struct3.Steered_md_sampling_method()
                hidr_stage.sampling_method.cv_names.append(cv_names[0])
                hidr_stage.sampling_method.force_constant = 10000.0
                hidr_stage.sampling_method.velocity = 0.01
                
            else:
                raise NotImplementedError(
                    f"Sampling method '{hidr_stage_name}' not currently supported for conversion.")

            hidr_stage.completion_criteria \
                = compl_crit_struct3.Reporter_progress_completion_criteria()
            hidr_stage.completion_criteria.reporter_name = "my_anchor_reporter"
            hidr_stage.completion_criteria.reporter_progress_threshold = 1.0
            hidr_stage.completion_criteria.interval = 250
            
            anchor_reporter = reporters_struct3.Anchor_reporter()
            anchor_reporter.name = "my_anchor_reporter"
            anchor_reporter.output_interval = 250

            state_reporter = reporters_struct3.State_reporter()
            state_reporter.name = "state_reporter1"

            dcd_reporter = reporters_struct3.Position_reporter()
            if hidr_stage_name == "metadynamics":
                dcd_reporter.name = "hidr_traj_metadyn"
            dcd_reporter.output_interval = 250

            hidr_stage.reporters.append(anchor_reporter)
            hidr_stage.reporters.append(state_reporter)
            hidr_stage.reporters.append(dcd_reporter)
            hidr_stage.input_stage_index = 0
            hidr_stage.scope = "unpartitioned"

            model3.stages.append(hidr_stage)

            logistic_stage_index = 2
            mmvt_stage_index = 3

        # Assign positions to anchors logistic stage
        logistic_stage = logistic_struct3.Stage_anchor_starting_structures_from_reporter()
        logistic_stage.index = logistic_stage_index
        logistic_stage.name = "logistic_anchor_starting_structures"
        logistic_stage.input_stage_index = logistic_stage_index - 1
        logistic_stage.reporter_name = "my_anchor_reporter"
        logistic_stage.scope = "partitioned"
        logistic_stage_dir = f"stage_{logistic_stage_index}_logistic_anchor_starting_structures"
        logistic_stage_positions_name \
            = logistic_stage.output_positions_basename_noext + ".swarm_0.pdb"
        model3.stages.append(logistic_stage)

        # MMVT stage
        mmvt_stage = md_scale3.Stage_md()
        mmvt_stage.index = mmvt_stage_index
        mmvt_stage.name = "MMVT"
        mmvt_stage.run_minimization = model2.openmm_settings.run_minimization
        mmvt_stage.md_restart_checkpoint_interval \
            = model2.calculation_settings.restart_checkpoint_interval
        mmvt_stage.ensemble = ensemble
        mmvt_stage.sampling_method = sampl_meth_struct3.MMVT_md_sampling_method()
        mmvt_stage.completion_criteria \
            = compl_crit_struct3.Number_of_steps_completion_criteria()
        mmvt_stage.completion_criteria.number_of_steps \
            = model2.calculation_settings.num_production_steps
        
        state_reporter_mmvt = reporters_struct3.State_reporter()
        state_reporter_mmvt.name = "state_reporter_mmvt"
        state_reporter_mmvt.output_interval \
            = model2.calculation_settings.energy_reporter_interval
        
        position_reporter_mmvt = reporters_struct3.Position_reporter()
        position_reporter_mmvt.name = "mmvt"
        position_reporter_mmvt.output_interval \
             = model2.calculation_settings.trajectory_reporter_interval
        
        mmvt_stage.reporters.append(state_reporter_mmvt)
        mmvt_stage.reporters.append(position_reporter_mmvt)
        mmvt_stage.input_stage_index = mmvt_stage_index - 1
        mmvt_stage.scope = "partitioned"
        model3.stages.append(mmvt_stage)

        if doing_bd:
            # BD stage
            bd_stage = bd_scale3.Stage_bd()
            bd_stage.index = mmvt_stage_index + 1
            bd_stage.name = "bd"
            bd_stage.n_trajectories_per_output \
                = min(1000, model2.k_on_info.b_surface_num_trajectories)
            bd_stage.n_steps_per_output = 100000
            bd_stage.sampling_method = sampl_meth_struct3.NAM_bd_sampling_method()
            bd_stage.completion_criteria \
                = compl_crit_struct3.Number_of_trajectories_completion_criteria()
            bd_stage.completion_criteria.number_of_trajectories \
                = model2.k_on_info.b_surface_num_trajectories
            bd_stage.input_stage_index = 0
            bd_stage.scope = "unpartitioned"
            model3.stages.append(bd_stage)

        # Anchors and milestones
        highest_milestone_index = 0
        anchors_dir = os.path.join(model3.directory, base3.ANCHOR_SETS_DIR,
                                   base3.PREPARE_SET_NAME)
        
        max_step_list, times_dict = common_converge2.get_mmvt_max_steps(model2)
        
        for alpha, anchor2 in enumerate(model2.anchors):
            anchor3 = structures3.Anchor()
            anchor3.index = alpha
            anchor3.name = anchor2.name
            anchor3.directory = f"anchor_{alpha:04d}"
            anchor_full_dir = os.path.join(anchors_dir, anchor3.directory)
            os.makedirs(anchor_full_dir, exist_ok=True)

            
            
            anchor3.scale_types = ["molecular_dynamics"]
            anchor3.scale_parameters_topology = {}
            anchor3.stage_parameters = {}
            anchor3.stage_parameters["anchor_structures_from_metaD"] = {
                "index": logistic_stage_index,
                "directory": f"stage_{logistic_stage_index}_logistic_anchor_starting_structures",
                "positions": "output_positions"
            }
            anchor3.stage_parameters["MMVT"] = {
                "index": mmvt_stage_index,
                "directory": f"stage_{mmvt_stage_index}_MMVT",
                "positions": "mmvt"
            }
            for stage_param, stage_attrs in anchor3.stage_parameters.items():
                stage_dir = os.path.join(anchor_full_dir, stage_attrs["directory"])
                os.makedirs(stage_dir, exist_ok=True)
                
            anchor3.end_anchor = anchor2.endstate
            if anchor2.bulkstate:
                anchor3.bulk_anchor = True
                anchor3.name = "bulk"
                anchor3.end_anchor = True
                if doing_bd:
                    if "brownian_dynamics" not in anchor3.scale_types:
                        anchor3.scale_types.append("brownian_dynamics")

            anchor3.neighbor_anchor_indices = []

            for i, milestone2 in enumerate(anchor2.milestones):
                milestone3 = structures3.Milestone()
                milestone3.index = milestone2.index
                if milestone3.index > highest_milestone_index:
                    highest_milestone_index = milestone3.index
                milestone3.boundary = True
                milestone3.neighbor_anchor_index = milestone2.neighbor_anchor_index
                anchor3.neighbor_anchor_indices.append(milestone2.neighbor_anchor_index)
                neighbor_anchor2 = model2.anchors[milestone2.neighbor_anchor_index]
                if neighbor_anchor2.bulkstate and doing_bd:
                    if "brownian_dynamics" not in anchor3.scale_types:
                        anchor3.scale_types.append("brownian_dynamics")

                milestone3.alias_index = milestone2.alias_index
                milestone3.variables = {}
                
                if "me_0" in milestone2.variables:
                    # Then it's a Voronoi milestone.
                    for key, value in milestone2.variables.items():
                        milestone3.variables[key] = value
                else:
                    # Then it's an ordinary 1D milestone.
                    milestone3.variables["k"] = milestone2.variables["k"]
                    # There should only be one entry in anchor2.variables
                    milestone3.variables["me_0"] = list(anchor2.variables.values())[0]
                    milestone3.variables["neighbor_0"] \
                        = list(neighbor_anchor2.variables.values())[0]
                    if "radius" in milestone2.variables:
                        milestone3.variables["avg_value"] \
                            = milestone2.variables["radius"]
                    else:
                        milestone3.variables["avg_value"] \
                            = milestone2.variables["value"]

                anchor3.milestones.append(milestone3)

            # Go through the key-sorted variables in the anchor, and add them
            #  to the seekr3 anchor.
            for i, variable in enumerate(sorted(anchor2.variables.keys())):
                new_var = f"value_{i}"
                anchor3.variables[new_var] = anchor2.variables[variable]

            # Copy structures for logistic stage
            pdb_filename = None
            logistic_stage_full_dir = os.path.join(
                anchor_full_dir, logistic_stage_dir)
            if anchor2.amber_params is not None:
                pdb_filename = anchor2.amber_params.pdb_coordinates_filename
            elif anchor2.charmm_params is not None:
                pdb_filename = anchor2.charmm_params.pdb_coordinates_filename
            elif anchor2.forcefield_params is not None:
                pdb_filename = anchor2.forcefield_params.pdb_coordinates_filename
            
            if pdb_filename is not None:
                src_pdb = os.path.join(
                    model2_full_dir, anchor2.directory, "building", pdb_filename)
                dest_pdb = os.path.join(logistic_stage_full_dir, logistic_stage_positions_name)
                #print("Copying logistic stage starting structure from "f"{src_pdb} to {dest_pdb}")
                if os.path.exists(src_pdb):
                    shutil.copy(src_pdb, dest_pdb)

            # Copy structures for MMVT stage
            mmvt_stage_full_dir = os.path.join(
                anchor_full_dir, f"stage_{mmvt_stage_index}_MMVT")
            mmvt_stage_dcds_glob = os.path.join(
                model2_full_dir, anchor2.directory, "prod", "*.dcd")
            mmvt_stage_dcds = glob.glob(mmvt_stage_dcds_glob)
            for i, mmvt_dcd in enumerate(mmvt_stage_dcds):
                mmvt_dcd_basename = f"mmvt{i}.swarm_0.position_reporter_1.dcd"
                mmvt_dcd_dest = os.path.join(mmvt_stage_full_dir, mmvt_dcd_basename)
                shutil.copy(mmvt_dcd, mmvt_dcd_dest)
            
            mmvt_stage_output_glob = os.path.join(
                model2_full_dir, anchor2.directory, "prod", "*.out")
            mmvt_new_output_file_name = "milestoning_restart1_swarm_0.out"
            convert_mmvt_output_files(
                anchor2,
                mmvt_stage_output_glob, 
                os.path.join(mmvt_stage_full_dir, mmvt_new_output_file_name),
                engine3.integrator_settings.timestep,
                max_step_list,
                times_dict[alpha])

            model3.anchors.append(anchor3)

        model3.num_milestones = highest_milestone_index + 1
        model3.num_anchors = len(model3.anchors)

        if doing_bd:
            bd_stage_dir = os.path.join(
                anchors_dir,
                bd_engine3.b_surface_directory)
            # make bd_stage_dir if it doesn't exist
            os.makedirs(bd_stage_dir, exist_ok=True)
            old_bd_dir = os.path.join(
                model2_full_dir, model2.k_on_info.b_surface_directory)
            # Copy every file from old_bd_dir to bd_stage_dir
            for filename in os.listdir(old_bd_dir):
                src_file = os.path.join(old_bd_dir, filename)
                dest_file = os.path.join(bd_stage_dir, filename)
                if os.path.exists(src_file):
                    shutil.copy(src_file, dest_file)

    else:
        raise NotImplementedError(
            "Only MMVT SEEKR2 models are currently supported for conversion.")
    
    seekr3_output_model_filename = os.path.join(model3.directory, "model.json")
    model3.save(seekr3_output_model_filename)
    model3.save_cvs(model3.directory, base3.PREPARE_SET_NAME)
    model3.save_selections(model3.directory, base3.PREPARE_SET_NAME)
    model3.save_anchor_set(model3.directory, base3.PREPARE_SET_NAME)
    log_directory = model3.get_log_directory(model3.directory)
    preparer3.generate_directory(log_directory)
    synopsis_filename = model3.get_synopsis_filename(model3.directory)
    base3.configure_logging(synopsis_filename, console=False)
    full_command = " ".join(sys.argv)
    logger.info(f"Converting from SEEKR2 with command: {full_command}")

    return


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SEEKR2 model and calculation to SEEKR3 format.")
    parser.add_argument(
        "seekr2_model_xml", help="Path to the SEEKR2 model.xml file.")
    parser.add_argument(
        "seekr3_output_dir", help="Path to the output directory where the SEEKR3 "\
        "model and data will be written.")
    parser.add_argument(
        "-p", "--pdb_file", dest="pdb_file", type=str, default=None,
        metavar="FILE", help="One PDB file which will be "\
        "placed into the correct anchor for hidr stage.")
    parser.add_argument(
        "-H", "--metadyn_height", dest="metadyn_height", type=float, 
        default=1.0, help="The initial heights of the metadynamics Gaussians. "\
        "Default: 1.0 kJ/mol.")
    parser.add_argument(
        "-b", "--skip_bd", dest="skip_bd", action="store_true",
        help="Whether to skip conversion of BrownDye settings and data. "\
            "By default, conversion of BrownDye settings and data is "\
            "performed.")
    # NOTE: not ever really done
    #parser.add_argument(
    #    "-a", "--assign_from_all_anchors", dest="assign_from_all_anchors", 
    #    action="store_true",
    #    help="Assign the parameter files from all SEEKR2 anchors to all SEEKR3 "\
    #    "anchors. If left at default False, it is assumed that all anchors have the "\
    #    "same parameter files, and only the first anchor's parameter files will be "\
    #    "assigned to all SEEKR3 anchors."
    #)
    
    # TODO: argument to spread out the prmtop files across the anchors vs. 
    # keeping anchor0's prmtop for the unpartitioned scope. Needed for qmrebind.
    args = parser.parse_args()
    seekr2_model_xml = args.seekr2_model_xml
    seekr3_output_dir = args.seekr3_output_dir
    pdb_file = args.pdb_file
    metadyn_height = args.metadyn_height
    skip_bd = args.skip_bd
    #assign_from_all_anchors = args.assign_from_all_anchors

    model2 = base2.load_model(seekr2_model_xml)
    convert_from_seekr2(model2, seekr3_output_dir, pdb_file=pdb_file,
                        metadyn_height=metadyn_height, skip_bd=skip_bd)
                        #assign_from_all_anchors=assign_from_all_anchors)

if __name__ == "__main__":
    main()