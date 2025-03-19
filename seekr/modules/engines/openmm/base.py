"""

"""

import typing
from attrs import define, field, validators, Factory
from attrs.validators import instance_of, optional, in_, gt, and_

import numpy as np
import openmm.unit as unit
import openmm.app as openmm_app

import seekr.modules.engines.base as engine_base
import seekr.modules.engines.openmm.sim as sim
import seekr.modules.engines.openmm.runner as runner

if typing.TYPE_CHECKING:
    import seekr.modules.data_structures as data_structures

@define
class Openmm_md_engine:
    type: typing.Literal["openmm"] = "openmm"
    nonbonded_method: str | None = field(
        default="pme", converter=engine_base.string_lower, validator=
            optional(and_(instance_of(str), in_([
                "nocutoff", "cutoffperiodic", "cutoffnonperiodic", 
                "pme", "ljpme", "ewald"]))))
    nonbonded_cutoff: float | None = field(
        default=0.9, validator=optional(and_(instance_of(float), gt(0))))
    constraints: str | None = field(
        default="hbonds", converter=engine_base.string_lower, 
        validator=optional(and_(instance_of(str), in_([
            "none", "hbonds", "allbonds", "hangles"]))))
    hydrogen_mass: float | None = field(
        default=None, validator=optional(and_(instance_of(float), gt(0))))
    initial_temperature: float = field(
        default=298.15, validator=[instance_of(float), gt(0)])
    integrator_settings: engine_base.Integrator_settings | None = None
    # If platform_settings is None, then use Reference platform
    platform_settings: engine_base.Platform_settings | None = None
    toy_settings: engine_base.Toy_settings | None = None
    systemwide_settings: engine_base.MD_systemwide_settings | None = None

    def get_pdb_object(
            self, 
            pdb_filename: str) -> openmm_app.PDBFile:
        """
        Using the OpenMM engine, get a PDB object from a PDB file.
        """
        pdb_obj = openmm_app.PDBFile(pdb_filename)
        return pdb_obj

    def get_pdb_num_frames(
            self, 
            pdb_filename: str) -> int:
        """
        Using the OpenMM engine, get the number of frames in a PDB file.
        """
        pdb_obj = self.get_pdb_object(pdb_filename)
        num_frames = pdb_obj.getNumFrames()
        return num_frames
    
    def get_random_seed(
            self
            ) -> int:
        """
        Get the random seed for the OpenMM engine.
        """
        if self.integrator_settings is not None:
            if hasattr(self.integrator_settings, "random_seed"):
                return self.integrator_settings.random_seed
            
        return 0

    def set_random_seed(
            self, 
            seed: int
            ) -> None:
        """
        Set the random seed for the OpenMM engine.
        """
        if self.integrator_settings is not None:
            if hasattr(self.integrator_settings, "random_seed"):
                self.integrator_settings.random_seed = seed
        return

    def run_simulation(
            self, 
            model: "data_structures.Seekr_model", 
            stage_index: int, 
            anchor_index: int, 
            restart: bool, 
            total_simulation_length: int, 
            swarm_index: int | None = None, 
            device_index: str | None = None,
            force_overwrite: bool = False):
        """
        Run an OpenMM simulation.
        """
        if anchor_index is not None:
            assert anchor_index >= 0, "Invalid anchor index provided. "\
                "Anchor index must be non-negative."
            assert anchor_index < len(model.anchors), \
                "Invalid anchor index provided. Anchor index must be less "\
                "than the number of anchors."
        if device_index is not None:
            assert self.platform_settings is not None, \
                "A device_index is provided, yet a platform is not set."
            self.platform_settings.device_index = device_index

        # NOTE: does old_num_production_steps need to be saved and reset?
        runner = runner.Runner_openmm(model, stage_index, anchor_index)
        default_output_file, restart_index = runner.prepare(
            restart, force_overwrite, swarm_index)
        if swarm_index is None:
            frame = 0
        else:
            frame = swarm_index
        # TODO: get info from enhanced sampling object to pass to next func
        sim_openmm_obj = self.sampling_method.create_sim_openmm()
        if sim_openmm_obj is None:
            sim_openmm_obj = sim.create_sim_openmm(
                model, runner.input_directory, runner.input_positions, 
                runner.md_parameters_topology, frame, runner.ensemble,
                runner.sampling_engine_interface)
                # TODO: change self.sampling_method argument above to a
                # sampling_method_OpenMM_intersection object
        runner.run(sim_openmm_obj, restart, restart_index)

def get_box_vectors_from_pdb(pdb_filename: str) -> unit.Quantity:
    """
    Extract the box_vectors from the CRYST line in a pdb file.
    """
    pdb_obj = openmm_app.PDBFile(pdb_filename)
    box_vectors = pdb_obj.topology.getPeriodicBoxVectors()
    assert box_vectors is not None, "No box vectors "\
    "found in {}. ".format(pdb_filename) \
    + "Box vectors for an anchor must be defined with a CRYST "\
    "line within the PDB file, or explicitly set in the model "\
    "input XML file."
    length1 = np.linalg.norm(box_vectors[0].value_in_unit(unit.nanometers))
    length2 = np.linalg.norm(box_vectors[1].value_in_unit(unit.nanometers))
    length3 = np.linalg.norm(box_vectors[2].value_in_unit(unit.nanometers))
    assert (length1 > 0.0) and (length2 > 0.0) and (length3 > 0.0), \
        "Box vector length(s) of zero detected. Please check box vectors in " \
        "input PDB."
    return box_vectors