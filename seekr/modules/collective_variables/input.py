"""
collective_variables/input.py

Collective variable, state, and anchor objects for the prepare
stage of SEEKR.
"""

import os
import shutil
import typing
import logging
from collections import defaultdict

import seekr.modules.base as base
import seekr.modules.data_structures as data_structures
import seekr.modules.collective_variables.distance_cv as distance_cv

if typing.TYPE_CHECKING:
    import seekr.modules.prepare_base as prepare_base

CV_input = typing.Union[distance_cv.Distance_cv_input]

Collective_variable = typing.Union[distance_cv.Distance_CV]

def cleanse_anchor_outputs(
        model: data_structures.Seekr_model,
        anchor: data_structures.Anchor) -> None:
    """
    Delete all simulation outputs from an existing anchor.
    """
    # TODO: this will be engine-dependent
    logging.warning("Warning: cleanse_anchor_outputs not yet implemented.")
    return

def modify_model(
        old_model: data_structures.Seekr_model, 
        new_model: data_structures.Seekr_model, 
        root_directory: str, 
        force_overwrite=False) -> list[int]:
    """
    If someone runs the prepare stage on a model with existing
    directories and simulation results, examine the changes to delete
    and rename directories properly, and only overwrite simulation files
    if forced to do so.
    """
    
    old_model.directory = root_directory
    anchor_pairs: list[tuple[int, int]] = []
    old_anchors_to_delete: list[int] = []
    new_anchors_to_create: list[int] = []
    # Look through the new and old milestones and pair them up by the value
    #  of their variables
    for alpha, anchor1 in enumerate(new_model.anchors):
        #if anchor1.bulkstate:
        #    continue
        alpha_paired: bool = False
        for beta, anchor2 in enumerate(old_model.anchors):
            #if anchor2.bulkstate:
            #    continue
            if anchor1.variables == anchor2.variables:
                pair = (alpha, beta)
                anchor_pairs.append(pair)
                alpha_paired = True
                break
        if not alpha_paired:
            new_anchors_to_create.append(alpha)
        
    for beta, anchor2 in enumerate(old_model.anchors):
        #if anchor2.bulkstate:
        #    continue
        beta_paired = False
        for pair in anchor_pairs:
            if beta == pair[1]:
                beta_paired = True
                break
        if not beta_paired:
            old_anchors_to_delete.append(beta)
        
    # Now check all the paired anchors to see if anyone's milestones
    # have changed
    old_anchors_with_changed_milestones = []
    old_anchors_with_starting_pdbs_to_keep = []
    for pair in anchor_pairs:
        (alpha, beta) = pair
        anchor1: data_structures.Anchor = new_model.anchors[alpha]
        anchor2: data_structures.Anchor = old_model.anchors[beta]
        milestones_changed = False
        milestone_pairs = []
        for i, milestone1 in enumerate(anchor1.milestones):
            milestone1_paired = False
            for j, milestone2 in enumerate(anchor2.milestones):
                if milestone1.variables == milestone2.variables:
                    milestone_pairs.append((i,j))
                    milestone1_paired = True
                    break
            if not milestone1_paired:
                milestones_changed = True
        for j, milestone2 in enumerate(anchor2.milestones):
            milestone2_paired = False
            for milestone_pair in milestone_pairs:
                if j == milestone_pair[1]:
                    milestone2_paired = True
                    break
            if not milestone2_paired:
                milestones_changed = True
        if milestones_changed:
            old_anchors_with_changed_milestones.append(beta)
            
        # See whether the starting structures have changed
        if not new_model.using_toy():
            # TODO: this originally looked to see whether hidr or string
            #  PDB output files were present, but that was a hacky solution.
            #  This time, examine whether the anchor has a stage with a PDB
            #  file defined.
            pass
            #old_anchor_starting = base.get_anchor_pdb_filename(anchor2)
            #if re.match(KEEP_ANCHOR_RE, old_anchor_starting):
            #    old_anchors_with_starting_pdbs_to_keep.append(beta)
    
    warning_msg = ""
    # check whether any anchors have been given new starting structures
    if len(old_anchors_with_starting_pdbs_to_keep) > 0:
        warning_msg+=f"The anchor(s) {old_anchors_with_starting_pdbs_to_keep}"\
            +"have been given new starting structures (from a simulation stage) "\
            "and the entered command mismatches the new structure file names. "\
            "If you desire to ignore this message, then use the "\
            "--force_overwrite (-f) option, and the existing structures from the stage "\
            "will be kept in the new model. If you wish to overwrite these, you "\
            "will need to change the model.xml file or write the root directory "\
            "to a new location.\n\n"
    
    # check whether deleting directories contain simulation files
    deleting_directories_with_files = []
    for del_anchor_index in old_anchors_to_delete:
        del_anchor: data_structures.Anchor = old_model.anchors[del_anchor_index]
        if del_anchor.has_files(root_directory):
            deleting_directories_with_files.append(del_anchor.directory)
    
    if len(deleting_directories_with_files) > 0 :
        warning_msg+=f"The anchor(s) {deleting_directories_with_files} "\
            +"already have existing output files and the entered command "\
            "would overwrite them. If you desire to overwrite the existing "\
            "files, then use the --force_overwrite (-f) option, and these "\
            f"anchors will be deleted in directory: {root_directory}\n\n"
    
    # Check whether directories with changed milestones already contain
    # simulation files
    cleansing_anchors: list[data_structures.Anchor] = []
    cleansing_directories: list[str] = []
    for cleanse_anchor_index in old_anchors_with_changed_milestones:
        cleanse_anchor: data_structures.Anchor = old_model.anchors[cleanse_anchor_index]
        if cleanse_anchor.has_files():
            cleansing_anchors.append(cleanse_anchor)
            cleansing_directories.append(cleanse_anchor.directory)
    
    if len(cleansing_directories) > 0 :
        warning_msg+=f"The anchor(s) {cleansing_directories} "\
            +"already have existing output files yet the milestone locations "\
            "will be changed if this command proceeds. If you desire "\
            "to overwrite the existing files, then use the "\
            "--force_overwrite (-f) option, and these outputs will be "\
            f"deleted in directory: {root_directory}\n\n"
    
    if warning_msg and not force_overwrite:
        print(warning_msg)
        raise Exception("Cannot overwrite existing files.")
    
    for del_anchor_index in old_anchors_to_delete:
        del_anchor = old_model.anchors[del_anchor_index]
        full_path = os.path.join(root_directory, del_anchor.directory)
        print("removing directory: "+del_anchor.directory)
        shutil.rmtree(full_path)
    
    for cleansing_anchor in cleansing_anchors:
        print("removing output files from anchor:", cleansing_anchor.name)
        cleanse_anchor_outputs(old_model, cleansing_anchor)
    
    # Now that the equivalent anchors are known, we know which ones will
    # be deleted, and which ones are being changed, with or without
    # simulation data, 
    
    new_anchors_with_starting_pdbs_to_keep = []
    for pair in anchor_pairs:
        (alpha, beta) = pair
        anchor1 = old_model.anchors[beta]
        anchor2 = new_model.anchors[alpha]
        full_path1 = os.path.join(root_directory, anchor1.directory)
        full_path2 = os.path.join(root_directory, anchor2.directory)
        if full_path1 != full_path2:
            if os.path.exists(full_path2):
                # then we cannot rename to this without giving this one
                # a temporary directory
                for problem_anchor in old_model.anchors:
                    if problem_anchor.directory == "":
                        continue
                    if problem_anchor.directory == anchor2.directory:
                        temporary_name = problem_anchor.directory+"temp"
                        temp_full_path1 = os.path.join(root_directory, 
                                                  problem_anchor.directory)
                        temp_full_path2 = os.path.join(root_directory, 
                                                  temporary_name)
                        print("moving directory {} to {}".format(
                            problem_anchor.directory, temporary_name))
                        os.rename(temp_full_path1, temp_full_path2)
                        problem_anchor.directory = temporary_name
                        break
            
            print("moving directory {} to {}".format(anchor1.directory, 
                                                 anchor2.directory))
            # If it gets here, it's because a new directory was made that
            # has no data in it. It can be safely deleted
            if os.path.exists(full_path2):
                shutil.rmtree(full_path2)
                
            os.rename(full_path1, full_path2)
        
        anchor2.md_parameters_topology = anchor1.md_parameters_topology
        anchor2.positions = anchor1.positions
            
        if anchor1.index in old_anchors_with_starting_pdbs_to_keep:
            new_anchors_with_starting_pdbs_to_keep.append(anchor2.index)
    
    for new_anchor_index_to_create in new_anchors_to_create:
        print("creating new anchor {}".format(new_anchor_index_to_create))
    
    return new_anchors_with_starting_pdbs_to_keep

def resolve_connections(
        connection_flag_dict: dict, 
        model: data_structures.Seekr_model, 
        associated_input_anchor: dict[int, data_structures.Anchor_input],
        anchor_connections: typing.Set[typing.Tuple[int]]
        ) -> list[data_structures.Anchor]:
    """
    Given connection information between anchors, merge anchors
    that are connected.
    """
    anchors: list[data_structures.Anchor] = model.anchors
    anchor_indices_to_remove = []
    index_reference = {}
    bulk_anchor: data_structures.Anchor | None = None
    for anchor in anchors:
        index_reference[anchor.index] = anchor.index
    
    for key in connection_flag_dict:
        anchor_connection_list: list[data_structures.Anchor] \
            = connection_flag_dict[key]
        bulk_discarded_indices = []
        if key == "bulk":
            bulk_anchor = anchor_connection_list[0]
            for anchor_discarded in anchor_connection_list:
                anchor_indices_to_remove.append(anchor_discarded.index)
                bulk_discarded_indices.append(anchor_discarded.index)
                
        else:
            assert len(anchor_connection_list) > 1, \
                f"Connection flag {key} needs to have endpoints "\
                +"in at least two anchors."
        
        anchor_kept = anchor_connection_list[0]
        for anchor_discarded in anchor_connection_list[1:]:
            input_anchor_kept = associated_input_anchor[anchor_kept.index]
            input_anchor_discarded = associated_input_anchor[anchor_discarded.index]
            assert anchor_kept.end_anchor == anchor_discarded.end_anchor, \
                "The anchors connected by connection flag {} ".format(key) \
                +"must have the same value for 'end_anchor'."
            if anchor_kept.name == "bulk":
                assert anchor_kept.name == anchor_discarded.name, \
                    "The anchors connected by connection flag {} ".format(key) \
                    +"must have the same name 'bulk'."   
            # Ensure that both anchors don't have starting structures 
            # defined
            if (input_anchor_kept.positions is None) \
                    or (input_anchor_kept.positions == ""):
                input_anchor_kept.positions \
                    = input_anchor_discarded.positions
            
            if (input_anchor_discarded.positions is None) \
                    or (input_anchor_discarded.positions == ""):
                input_anchor_discarded.positions \
                    = input_anchor_kept.positions
            assert input_anchor_kept.positions \
                    == input_anchor_discarded.positions, \
                "The anchors connected by connection flag {} ".format(key) \
                +"have inconsistent positions defined."
            assert input_anchor_kept.same_md_parameters(
                input_anchor_discarded.md_parameters_topology), \
                "The anchors connected by connection flag {} ".format(key) \
                +"have inconsistent MD parameters defined."
            anchor_indices_to_remove.append(anchor_discarded.index)
            anchor_kept.milestones += anchor_discarded.milestones
            anchor_kept.variables.update(anchor_discarded.variables)
            index_reference[anchor_discarded.index] = anchor_kept.index
            
    # sort and remove all redundant anchors
    anchor_indices_to_remove = sorted(set(anchor_indices_to_remove), 
                                      reverse=True)
    for anchor_index_to_remove in anchor_indices_to_remove:
        if len(anchors) > anchor_index_to_remove+1:
            for lower_anchor in anchors[anchor_index_to_remove+1:]:
                index_reference[lower_anchor.index] -= 1
                
    for anchor_index_to_remove in anchor_indices_to_remove:
        anchors.pop(anchor_index_to_remove)
    
    if bulk_anchor is not None:
        # make sure the bulk anchor is the last anchor always
        anchors.append(bulk_anchor)
        index_reference[bulk_anchor.index] = len(anchors) - 1
        for bulk_discarded in bulk_discarded_indices:
            index_reference[bulk_discarded] = len(anchors) - 1
    
    visited_new_indices = []
    
    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            anchors[new_index].index = new_index
            anchors[new_index].directory = f"anchor_{new_index}"
            if anchors[new_index].name.startswith("anchor_"):
                anchors[new_index].name = anchors[new_index].directory
            
            for pair in anchor_connections:
                pair_tuple = set(pair)
                if old_index in pair:
                    pair_tuple.remove(old_index)
                    old_other_index = pair_tuple.pop()
                    new_other_index = index_reference[old_other_index]
                    assert new_other_index != new_index, "An anchor cannot "\
                        "be a neighbor to itself."
                    anchors[new_index].neighbor_anchor_indices.append(new_other_index)

            # TODO: this shouldn't be necessary if the sampling method
            # subsequently adds milestones after this is all resolved
            """
            alias_index = 1
            milestones_to_remove = []
            for i, milestone in enumerate(anchors[new_index].milestones):
                if (model.get_type() == "mmvt") \
                        and (new_index not in visited_new_indices):
                    # don't renumber the same anchor twice
                    neighbor_id = milestone.neighbor_anchor_index
                    milestone.neighbor_anchor_index = index_reference[neighbor_id]
                    milestone.alias_index = alias_index
                    if milestone.neighbor_anchor_index \
                            == anchors[new_index].index:
                        milestones_to_remove.append(i)
                    alias_index += 1
                    
            for milestone_to_remove in sorted(milestones_to_remove, reverse=True):
                anchors[new_index].milestones.pop(milestone_to_remove)
            """
            
        visited_new_indices.append(new_index)
    
    if bulk_anchor is None:
        assert not model.using_bd(), "A bulk anchor must be defined "\
            "if using BD. This can be done by naming an input_anchor "\
            "with the name 'bulk', or by setting the 'bulk_anchor' flag "\
            "to True."
    else:
        bulk_anchor.md = False
        if model.using_bd():
            bulk_anchor.bd = True
            bulk_anchor.directory = model.bd_engine.b_surface_directory

            for neighbor_index in bulk_anchor.neighbor_anchor_indices:
                neighbor_anchor = anchors[neighbor_index]
                neighbor_anchor.bd = True


    return anchors, index_reference

def resolve_modifications_and_filetree(
        model: data_structures.Seekr_model, 
        anchors: list[data_structures.Anchor], 
        index_reference: dict[int, int],
        associated_input_anchor: dict[int, data_structures.Anchor_input],
        root_directory: str, 
        force_overwrite: bool
        ) -> None:
    """
    Resolve modifications to the model and the filetree.
    """
    json_path = os.path.join(root_directory, "model.json")
    if os.path.exists(json_path):
        # then a model file already exists at this location: update
        # the anchor directories.
        old_model = data_structures.load_model(json_path)
        new_anchors_with_starting_pdbs_to_keep = modify_model(
            old_model, model, root_directory, force_overwrite)
    else:
        new_anchors_with_starting_pdbs_to_keep = None
    
    for stage in model.stages:
        assert stage.input_from_systemwide_settings is not None, \
            "The input_from_systemwide_settings flag must be defined "\
            "for each stage."
        assert stage.output_to_systemwide_settings is not None, \
            "The output_to_systemwide_settings flag must be defined "\
            "for each stage."
        if not stage.input_from_systemwide_settings:
            stage.add_stage_directory_to_anchors(model)
        elif stage.type == "md_stage":
            stage.add_stage_directory_to_systemwide_settings(model)

    for old_index in index_reference:
        new_index = index_reference[old_index]
        if new_index < len(anchors):
            input_anchor = associated_input_anchor[old_index]
            anchors[new_index].generate_filetree(root_directory)
            if (new_anchors_with_starting_pdbs_to_keep is not None) \
                    and (new_index in new_anchors_with_starting_pdbs_to_keep):
                logging.info("Keeping starting structure for anchor:", new_index)
            else:
                input_anchor.copy_initial_files(anchors[new_index], root_directory, 
                                                model.using_toy())

    return
