from __future__ import annotations
"""
modules/engines/openmm/sim.py

Objects needed for OpenMM Simulation and System objects.
"""

import os
import typing
import logging

import numpy as np
import openmm.unit as unit
import openmm
import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.openmm.base as openmm_base
import seekr.modules.structures as structures
import seekr.modules.sampling_methods.openmm_interface as sampling_openmm_interface
import seekr.modules.completion_criteria.openmm_interface as completion_openmm_interface
import seekr.modules.restraints.openmm_interface as restraints_openmm_interface

DEFAULT_BAROSTAT_FREQUENCY = 25
SWARM_ID_RANDOM_SEED_FACTOR = 10000
RANDOM_VEL_SEED_FACTOR = 1234

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

logger = logging.getLogger(__name__)

# Particle masses from md_parameters_topology, keyed by
# (directory, topology type, filename, hydrogen_mass).
_OPENMM_PARTICLE_MASSES_CACHE: dict[
    tuple[str, str, str, float | None], np.ndarray
] = {}

def make_topology_system_from_amber(
        amber_parameters_topology: engines_structures.Amber_parameters_topology,
        directory: str,
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> typing.Tuple[openmm.System, openmm_app.Topology]:
    """
    Load the Amber parameters and topology from the provided object, and
    return OpenMM system and topology.
    """
    prmtop_filename = os.path.join(
        directory, amber_parameters_topology.prmtop_filename)
    prmtop = openmm_app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        hydrogenMass=hydrogenMass)
    return system, prmtop.topology

def make_topology_system_from_gromacs(
        gromacs_parameters_topology: engines_structures.Gromacs_parameters_topology,
        directory: str,
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> typing.Tuple[openmm.System, openmm_app.Topology]:
    """
    Load the Gromacs parameters and topology from the provided object, and
    return OpenMM system and topology.
    """
    top_filename = os.path.join(
        directory, gromacs_parameters_topology.top_filename)
    gro_filename = os.path.join(
        directory, gromacs_parameters_topology.gro_filename)
    top_file = openmm_app.GromacsTopFile(top_filename)
    gro_file = openmm_app.GromacsGroFile(gro_filename)
    system = top_file.createSystem(
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        hydrogenMass=hydrogenMass)
    return system, top_file.topology

def make_topology_system_from_charmm(
        charmm_parameters_topology: engines_structures.Charmm_parameters_topology,
        directory: str,
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> typing.Tuple[openmm.System, openmm_app.Topology]:
    """
    Load the CHARMM parameters and topology from the provided object, and
    return OpenMM system and topology.
    """
    psf_filename = os.path.join(
        directory, charmm_parameters_topology.psf_filename)
    param_filenames_list = []
    for param_basename in charmm_parameters_topology.param_filename_list:
        param_filename = os.path.join(directory, param_basename)
        param_filenames_list.append(param_filename)
    psf = openmm_app.CharmmPsfFile(psf_filename)
    # TODO: hacky - will this always work?
    psf.setBox(4.0*unit.nanometers, 4.0*unit.nanometers, 4.0*unit.nanometers)
    params = openmm_app.CharmmParameterSet(*param_filenames_list)
    system = psf.createSystem(
        params, nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        hydrogenMass=hydrogenMass)
    return system, psf.topology

def make_topology_system_from_forcefield(
        forcefield_parameters_topology: engines_structures.Forcefield_parameters,
        topology: openmm_app.Topology,
        directory: str,
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> openmm.System:
    """
    Load the OpenMM forcefield parameters and topology from the 
    provided object, and return OpenMM system.
    """
    forcefield_filenames = []
    for forcefield_filename in \
            forcefield_parameters_topology.built_in_forcefield_filenames:
        forcefield_filenames.append(forcefield_filename)
    
    for forcefield_basename in \
            forcefield_parameters_topology.custom_forcefield_filenames:
        forcefield_filenames.append(os.path.join(
            directory, forcefield_basename))

    forcefield = openmm_app.ForceField(*forcefield_filenames)
    system = forcefield.createSystem(
        topology, nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        hydrogenMass=hydrogenMass)
    return system

def make_topology_system_from_xml(
        openmm_system_obj: engines_structures.Openmm_system,
        directory: str, 
        ) -> openmm.System:
    """
    Load the OpenMM system XML from the 
    provided object, and return OpenMM system.
    """
    system_xml_filename = os.path.join(
        directory, openmm_system_obj.system_filename)
    with open(system_xml_filename) as input:
        system = openmm.XmlSerializer.deserialize(input.read())
    return system

def _md_parameters_directory(
        model: structures.Seekr_model,
        md_engine: typing.Any,
        ) -> str | None:
    """
    Directory that holds copied MD parameter/topology files
    (unpartitioned_scope/stage_0_initial), matching Openmm_runner.
    """
    settings = getattr(md_engine, "unpartitioned_settings", None)
    if settings is None:
        return None
    stage_params = getattr(settings, "stage_parameters", None)
    if not isinstance(stage_params, dict):
        return None
    initial = stage_params.get("initial")
    if not isinstance(initial, dict) or "directory" not in initial:
        return None
    unpart_dir = getattr(settings, "directory", None)
    model_dir = getattr(model, "directory", None)
    if not isinstance(unpart_dir, str) or not isinstance(model_dir, str):
        return None
    return os.path.join(model_dir, unpart_dir, initial["directory"])


def _topology_cache_filename(topo: typing.Any) -> str:
    topo_type = getattr(topo, "type", None)
    if topo_type == "Amber":
        return str(getattr(topo, "prmtop_filename", "") or "")
    if topo_type == "Charmm":
        return str(getattr(topo, "psf_filename", "") or "")
    if topo_type == "Gromacs":
        return str(getattr(topo, "top_filename", "") or "")
    if topo_type == "OpenMM_system":
        return str(getattr(topo, "system_filename", "") or "")
    if topo_type == "OpenMM_forcefield":
        return str(getattr(topo, "topology_filename", "") or "")
    return ""


def particle_masses_from_openmm_parameters(
        model: structures.Seekr_model,
        ) -> np.ndarray | None:
    """
    Particle masses (amu) from the OpenMM System used for MD.

    Looks up files from unpartitioned_settings.md_parameters_topology in
    the initial-stage parameters directory (no glob). Amber/Charmm/Gromacs
    use createSystem(NoCutoff, hydrogenMass=...); OpenMM_system deserializes
    the XML (masses already include HMR). Returns None for toy systems or
    when parameters are unavailable.
    """
    using_toy = getattr(model, "using_toy", None)
    if callable(using_toy) and using_toy():
        return None
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    if md_engine is None:
        return None
    settings = getattr(md_engine, "unpartitioned_settings", None)
    topo = getattr(settings, "md_parameters_topology", None) \
        if settings is not None else None
    if topo is None:
        return None
    topo_type = getattr(topo, "type", None)
    if not isinstance(topo_type, str):
        return None
    directory = _md_parameters_directory(model, md_engine)
    if directory is None or not os.path.isdir(directory):
        return None
    hydrogen_mass = getattr(md_engine, "hydrogen_mass", None)
    if not isinstance(hydrogen_mass, (int, float)):
        hydrogen_mass = None
    cache_name = _topology_cache_filename(topo)
    cache_key = (directory, topo_type, cache_name, hydrogen_mass)
    cached = _OPENMM_PARTICLE_MASSES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    hydrogenMass = None if hydrogen_mass is None \
        else float(hydrogen_mass) * unit.amu
    # Masses do not depend on electrostatics; pass a dummy cutoff because
    # AmberPrmtopFile.createSystem(NoCutoff, nonbondedCutoff=None) can raise
    # UnboundLocalError (boxX) on periodic prmtops.
    dummy_cutoff = 0.9 * unit.nanometer
    try:
        if topo_type == "Amber":
            system, _topology = make_topology_system_from_amber(
                topo, directory, openmm_app.NoCutoff, dummy_cutoff, None,
                hydrogenMass)
        elif topo_type == "Charmm":
            system, _topology = make_topology_system_from_charmm(
                topo, directory, openmm_app.NoCutoff, dummy_cutoff, None,
                hydrogenMass)
        elif topo_type == "Gromacs":
            system, _topology = make_topology_system_from_gromacs(
                topo, directory, openmm_app.NoCutoff, dummy_cutoff, None,
                hydrogenMass)
        elif topo_type == "OpenMM_system":
            system = make_topology_system_from_xml(topo, directory)
        elif topo_type == "OpenMM_forcefield":
            topology = _load_topology_pdb(
                directory, topo.topology_filename)
            system = make_topology_system_from_forcefield(
                topo, topology, directory, openmm_app.NoCutoff, dummy_cutoff,
                None, hydrogenMass)
        else:
            return None
    except Exception as exc:
        logger.warning(
            "Could not load OpenMM particle masses from %s (%s): %s. "
            "Falling back to mdtraj topology masses.",
            directory, topo_type, exc)
        return None
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton)
         for i in range(system.getNumParticles())],
        dtype=float)
    _OPENMM_PARTICLE_MASSES_CACHE[cache_key] = masses
    return masses


def _load_topology_pdb(
        md_parameters_directory: str,
        topology_filename: str
        ) -> openmm_app.Topology:
    """
    Load an OpenMM Topology from a topology PDB in the parameters directory.
    """
    assert topology_filename != "", \
        "topology_filename must be set on the parameters/topology object."
    topology_full_path = os.path.join(md_parameters_directory, topology_filename)
    return openmm_app.PDBFile(topology_full_path).topology

def make_toy_system_object(
        model: structures.Seekr_model
        ) -> typing.Tuple[openmm.System, openmm_app.Topology]:
    """
    Make openmm system and topology files for use with a toy system.
    """
    system = openmm.System()
    topology = openmm_app.Topology()
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    assert md_engine is not None
    toy_settings = md_engine.toy_settings
    assert toy_settings is not None
    num_particles = toy_settings.num_particles
    assert num_particles == len(toy_settings.masses)
    new_expression = "k*("+toy_settings.potential_energy_expression+")"
    force = openmm.CustomCentroidBondForce(num_particles, new_expression)
    force.addGlobalParameter("k", 1.0*openmm.unit.kilocalories_per_mole)
    groups_list = []
    for i in range(num_particles):
        mass = toy_settings.masses[i] * openmm.unit.amu
        system.addParticle(mass)
        atom_name = "X{}".format(i)
        if not atom_name in openmm_app.element.Element._elements_by_symbol:
            element = openmm_app.element.Element(0, atom_name, atom_name, mass)
        else:
            element = openmm_app.element.Element._elements_by_symbol[atom_name]
        chain = topology.addChain()
        residue = topology.addResidue("UNK", chain)
        topology.addAtom(atom_name, element, residue)
        groups_list.append(force.addGroup([i]))
    
    force.addBond(groups_list, [])
    system.addForce(force)
    return system, topology

def create_openmm_system(
        model: structures.Seekr_model,
        positions_directory: str,
        positions_basename: str,
        md_parameters_directory: str,
        md_parameters_topology: engines_structures.Parameters_topology,
        ) -> openmm.System:
    """
    Create an OpenMM system object for the given frame.
    """
    # TODO: This isn't right - how do we get the correct stage parameters
    # Given a stage index? The list here will not be indexed correctly...
    # Solution: add Nones to stage_parameters
    system = None
    md_engine = model.get_engine_by_scale_type("molecular_dynamics")
    assert md_engine.type == "openmm"
    nonbonded_method = md_engine.nonbonded_method
    if nonbonded_method is not None:
        nonbonded_method = nonbonded_method.lower()

    if nonbonded_method == "pme":
        nonbondedMethod = openmm_app.PME
        
    elif nonbonded_method == "nocutoff":
        nonbondedMethod = openmm_app.NoCutoff
        
    elif nonbonded_method == "cutoffnonperiodic":
        nonbondedMethod = openmm_app.CutoffNonPeriodic
        
    elif nonbonded_method == "cutoffperiodic":
        nonbondedMethod = openmm_app.CutoffPeriodic
        
    elif nonbonded_method == "ewald":
        nonbondedMethod = openmm_app.Ewald
    
    elif nonbonded_method is None:
        nonbondedMethod = None
        
    else:
        raise Exception("nonbonded method not found: "\
                        f"{nonbonded_method}")

    if md_engine.constraints is None:
        constraints_str = None
    else:
        constraints_str = md_engine.constraints

    if constraints_str is None:
        constraints = None
    
    elif constraints_str.lower() == "none":
        constraints = None
    
    elif constraints_str.lower() == "hbonds":
        constraints = openmm_app.HBonds
        
    elif constraints_str.lower() == "allbonds":
        constraints = openmm_app.AllBonds
        
    elif constraints_str.lower() == "hangles":
        constraints = openmm_app.HAngles
        
    else:
        raise Exception("constraints not found: "\
                        f"{constraints_str}")

    if md_engine.hydrogen_mass is not None:
        hydrogenMass = md_engine.hydrogen_mass*openmm.unit.amu
    else:
        hydrogenMass = None

    velocities = None
    if model.using_toy():
        if (positions_basename is not None) and \
                positions_basename != "":
            toy_full_path = os.path.join(positions_directory, positions_basename)
            ext = os.path.splitext(toy_full_path)[-1]
            if ext == ".txt":
                positions = np.loadtxt(toy_full_path, ndmin=2)
            elif ext == ".state":
                with open(toy_full_path, "r") as f:
                    xml = f.read()

                state = openmm.XmlSerializer.deserialize(xml)
                positions = state.getPositions(asNumpy=True)
                velocities = state.getVelocities(asNumpy=True)
            else:
                raise Exception("For toy systems, positions file must be .txt or .state")

            num_frames = 1 #len(positions_frames)
        else:
            num_frames = 0
        system, topology = make_toy_system_object(model)
        box_vectors = None
    else:
        if (positions_basename is not None) and \
                positions_basename != "":
            positions_full_path = os.path.join(positions_directory, positions_basename)
            ext = os.path.splitext(positions_full_path)[-1]
            pdb_obj = None
            if ext == ".pdb":
                pdb_obj = openmm_app.PDBFile(positions_full_path)
                box_vectors = openmm_base.get_box_vectors_from_pdb(positions_full_path)
                positions = pdb_obj.getPositions()
                num_frames = pdb_obj.getNumFrames()
            elif ext == ".state":
                with open(positions_full_path, "r") as f:
                    xml = f.read()

                state = openmm.XmlSerializer.deserialize(xml)
                box_vectors = state.getPeriodicBoxVectors()
                positions = state.getPositions(asNumpy=True)
                velocities = state.getVelocities(asNumpy=True)
                num_frames = 1
            else:
                raise Exception("For non-toy systems, positions file must be .pdb or .state")
            
        else:
            pdb_obj = None
            box_vectors = None
            positions = None
            num_frames = 0

        if md_parameters_topology.type == "Amber":
            system, topology = make_topology_system_from_amber(
                md_parameters_topology, md_parameters_directory,
                nonbondedMethod, md_engine.nonbonded_cutoff,
                constraints, hydrogenMass)
        elif md_parameters_topology.type == "Gromacs":
            system, topology = make_topology_system_from_gromacs(
                md_parameters_topology, md_parameters_directory,
                nonbondedMethod, md_engine.nonbonded_cutoff,
                constraints, hydrogenMass)
        elif md_parameters_topology.type == "Charmm":
            system, topology = make_topology_system_from_charmm(
                md_parameters_topology, md_parameters_directory,
                nonbondedMethod, md_engine.nonbonded_cutoff,
                constraints, hydrogenMass)
        elif md_parameters_topology.type == "OpenMM_forcefield":
            topology = _load_topology_pdb(
                md_parameters_directory, md_parameters_topology.topology_filename)
            system = make_topology_system_from_forcefield(
                md_parameters_topology, topology,
                md_parameters_directory, nonbondedMethod,
                md_engine.nonbonded_cutoff, constraints, hydrogenMass)
        elif md_parameters_topology.type == "OpenMM_system":
            system = make_topology_system_from_xml(
                md_parameters_topology, md_parameters_directory)
            topology = _load_topology_pdb(
                md_parameters_directory, md_parameters_topology.topology_filename)

    assert system is not None, \
        "md_parameters_topology.type not found: "\
        f"{md_parameters_topology.type}"
    _assert_hydrogen_mass(system, topology, hydrogenMass)
    # set all non-milestoning force groups to zero
    for force in system.getForces():
        force.setForceGroup(0)

    return system, topology, positions, box_vectors, num_frames, velocities

def _assert_hydrogen_mass(
        system: openmm.System,
        topology: openmm_app.Topology,
        expected_hydrogen_mass: unit.Quantity | None,
        tolerance_amu: float = 1e-3,
        ) -> None:
    """
    Assert that the first hydrogen in the topology has the expected mass
    in the OpenMM System (e.g. after hydrogen-mass repartitioning).
    If no expected mass is provided, or the topology has no hydrogens,
    this is a no-op.
    """
    if expected_hydrogen_mass is None:
        return
    expected = expected_hydrogen_mass.value_in_unit(unit.amu)
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            actual = system.getParticleMass(atom.index).value_in_unit(unit.amu)
            assert abs(actual - expected) <= tolerance_amu, (
                f"Hydrogen mass mismatch: particle {atom.index} has "
                f"{actual} amu, expected {expected} amu"
            )
            return

def _compute_full_system_run_dofs(system: openmm.System):
    """
    Compute the number of degrees of freedom for a OpenMM system, particularly
    used in the Nose-Hoover thermostat/integrator.
    """
    massive_particles = set()
    for particle in range(system.getNumParticles()):
        mass = system.getParticleMass(particle)
        if mass.value_in_unit(unit.dalton) > 0:
            massive_particles.add(particle)

    num_dofs = 3*len(massive_particles)
    for constraint_index in range(system.getNumConstraints()):
        particle1, particle2, distance = system.getConstraintParameters(
            constraint_index)
        if particle1 in massive_particles and particle2 in massive_particles:
            num_dofs -= 1

    for force_index in range(system.getNumForces()):
        if isinstance(system.getForce(force_index), openmm.CMMotionRemover):
            num_dofs -= 3
    return num_dofs

def _get_yoshida_suzuki_weights(num_yoshida_suzuki):
    if num_yoshida_suzuki == 1:
        return [1.0]
    if num_yoshida_suzuki == 3:
        return [0.828981543588751, -0.657963087177502,
                0.828981543588751]
    if num_yoshida_suzuki == 5:
        return [0.2967324292201065, 0.2967324292201065,
                -0.186929716880426, 0.2967324292201065,
                0.2967324292201065]
    if num_yoshida_suzuki == 7:
        return [0.784513610477560, 0.235573213359357,
                -1.17767998417887, 1.31518632068391,
                -1.17767998417887, 0.235573213359357,
                0.784513610477560]
    raise ValueError("num_yoshida_suzuki must be 1, 3, 5, or 7")

class Sim_openmm:
    """
    Contain superclass information necessary to run a SEEKR2 
    calculation in OpenMM.
    
    platform : openmm.Platform()
        The platform to use for the simulation. The CUDA platform
        is recommended, but Reference is also available.
        
    properties : dict
        A dictionary of properties passed to the Simulation() object
        that apply to the platform.
        
    timestep : float
        The timestep (in picoseconds) which was passed to the 
        integrator, but is also useful to know in other places within 
        the program.
        
    output_filename : str
        The file that the plugin will write SEEKR output to, and which
        will be parsed by the analysis software.

    system : The OpenMM system object for this simulation.
    
    integrator : the OpenMM integrator object for this simulation.
    
    simulation : the OpenMM simulation object.
    
    traj_reporter : openmm.DCDReporter
        The OpenMM Reporter object to which the trajectory will be
        written.
        
    energy_reporter : openmm.StateDataReporter
        The OpenMM StateDataReporter to which the energies and other
        state data will be reported.
        
    """
    def __init__(self, anchor_index, swarm_id):
        self.platform = None
        self.properties = None
        self.timestep = None
        self.box_vectors = None
        self.system = None
        self.topology = None
        self.integrator = None
        self.simulation = None
        self.traj_reporter = openmm_app.DCDReporter
        self.energy_reporter = openmm_app.StateDataReporter
        if anchor_index is None:
            self.random_seed = 0
        else:
            self.random_seed = anchor_index + 1
        self.random_seed += swarm_id * SWARM_ID_RANDOM_SEED_FACTOR
        return
    
    def add_restraints(
            self,
            model: structures.Seekr_model,
            anchor_index: int | None,
            restraints_interfaces: list[restraints_openmm_interface.Base_restraint_openmm_interface],
            positions: unit.Quantity,
            periodic_system: bool
            ) -> None:
        """
        Add restraints to the OpenMM system.
        """
        for restraint_interface in restraints_interfaces:
            restraint_interface.add_forces_to_system(
                model=model,
                anchor_index=anchor_index,
                system=self.system, 
                positions=positions,
                periodic_system=periodic_system)
            
        return

    def add_integrator(
            self, 
            model: structures.Seekr_model,
            anchor_index: int | None,
            ensemble: base.Ensemble,
            sampling_method_interface: sampling_openmm_interface.Sampling_md_openmm_interface,
            completion_criteria_interface: completion_openmm_interface\
                .Completion_criteria_openmm_interface,
            restart: bool,
            system: openmm.System,
            timestep: float | None = None,
            ) -> None:
        """
        Assign the proper integrator to this OpenMM simulation.
        """
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.type == "openmm"
        assert md_engine.integrator_settings is not None
        anchor: structures.Anchor | engines_structures.MD_unpartitioned_settings
        if anchor_index is None:
            assert not sampling_method_interface.sampling_method.needs_to_be_in_anchor(), \
                "This sampling method must be run within an anchor, not unpartitioned."
            assert md_engine.unpartitioned_settings is not None
            anchor = md_engine.unpartitioned_settings
        else:
            anchor = model.anchors[anchor_index]
        integrator_settings = md_engine.integrator_settings
        if timestep is None:
            timestep = model.get_timestep_by_type("molecular_dynamics")
        rigid_constraint_tolerance = integrator_settings.rigid_tolerance
        self.timestep = timestep
        integrator = openmm.CustomIntegrator(timestep * unit.picoseconds)
        integrator_type = integrator_settings.type
        if integrator_type == "langevinMiddle":
            if ensemble is not None:
                assert ensemble.type in ["NVT", "NPT", "NPT_membrane"], \
                    "Ensemble type not allowed or recognized for Langevin "\
                    f"middle integrator: {ensemble.type}"
            temperature = integrator_settings.target_temperature * unit.kelvin
            friction = integrator_settings.friction
            integrator.addGlobalVariable("a", np.exp(-friction*timestep))
            integrator.addGlobalVariable("b", np.sqrt(1-np.exp(-2*friction*timestep)))
            integrator.addGlobalVariable("thermal_energy", kB*temperature)
            sampling_method_interface.add_global_variables_integrator(
                integrator=integrator, anchor=anchor)
            integrator.addPerDofVariable("x1", 0)
            integrator.addPerDofVariable("old_x", 0)
            integrator.addPerDofVariable("old_v", 0)
            sampling_method_interface.add_per_DOF_variables_integrator(
                integrator=integrator)
            sampling_method_interface.pre_integration_instructions(
                integrator=integrator)
            integrator.addComputePerDof("old_x", "x")
            integrator.addUpdateContextState()
            integrator.addComputePerDof("v", "v + dt*f/m")
            integrator.addConstrainVelocities()
            integrator.addComputePerDof("old_v", "v")
            integrator.addComputePerDof("x", "x + 0.5*dt*v")
            integrator.addComputePerDof("v", "a*v + b*sqrt(thermal_energy/m)*gaussian")
            integrator.addComputePerDof("x", "x + 0.5*dt*v")
            integrator.addComputePerDof("x1", "x")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "v + (x-x1)/dt")
            sampling_method_interface.post_integration_instructions(
                integrator=integrator, anchor=anchor)
            #completion_criteria_interface.post_integration_instructions(
            #    integrator=integrator)
            random_seed = integrator_settings.random_seed + self.random_seed
            integrator.setRandomNumberSeed(random_seed)

        elif integrator_type == "langevin":
            if ensemble is not None:
                assert ensemble.type in ["NVT", "NPT", "NPT_membrane"], \
                    "Ensemble type not allowed or recognized for Langevin "\
                    f"integrator: {ensemble.type}"
            temperature = integrator_settings.target_temperature * unit.kelvin
            friction = integrator_settings.friction
            integrator.addGlobalVariable("thermal_energy", kB*temperature)
            integrator.addGlobalVariable("friction_coeff", friction)
            integrator.addGlobalVariable("vscale", 0.0)
            integrator.addGlobalVariable("fscale", 0.0)
            integrator.addGlobalVariable("noisescale", 0.0)
            sampling_method_interface.add_global_variables_integrator(
                integrator=integrator, anchor=anchor, is_restart=restart)
            integrator.addPerDofVariable("old_x", 0)
            integrator.addPerDofVariable("old_v", 0)
            sampling_method_interface.add_per_DOF_variables_integrator(
                integrator=integrator)
            sampling_method_interface.pre_integration_instructions(
                integrator=integrator)
            integrator.addComputeGlobal("vscale", "exp(-dt*friction_coeff)")
            integrator.addComputeGlobal("fscale", "(1-vscale)/friction_coeff")
            integrator.addComputeGlobal("noisescale", 
                                        "sqrt(thermal_energy*(1-vscale*vscale))")
            integrator.addUpdateContextState()
            integrator.addComputePerDof("old_x", "x")
            integrator.addComputePerDof("v", 
                              "vscale*v + fscale*f/m "\
                              "+ noisescale*gaussian/sqrt(m)")
            integrator.addComputePerDof("x", "x + dt*v")
            integrator.addComputePerDof("old_v", "v")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "(x-old_x)/dt")
            sampling_method_interface.post_integration_instructions(
                integrator=integrator, anchor=anchor)
            random_seed = integrator_settings.random_seed + self.random_seed
            integrator.setRandomNumberSeed(random_seed)

        elif integrator_type == "brownian":
            if ensemble is not None:
                assert ensemble.type in ["NVT", "NPT", "NPT_membrane"], \
                    "Ensemble type not allowed or recognized for Brownian "\
                    f"integrator: {ensemble.type}"
            temperature = integrator_settings.target_temperature * unit.kelvin
            friction = integrator_settings.friction
            integrator.addGlobalVariable("thermal_energy", kB*temperature)
            integrator.addGlobalVariable("friction_coeff", friction)
            integrator.addGlobalVariable("fscale", 0.0)
            integrator.addGlobalVariable("noiseAmplitude", 0.0)
            sampling_method_interface.add_global_variables_integrator(
                integrator=integrator, anchor=anchor)
            integrator.addPerDofVariable("old_x", 0)
            integrator.addPerDofVariable("old_v", 0)
            sampling_method_interface.add_per_DOF_variables_integrator(
                integrator=integrator)
            sampling_method_interface.pre_integration_instructions(
                integrator=integrator)
            integrator.addUpdateContextState()
            integrator.addComputeGlobal("fscale", "dt/friction_coeff")
            integrator.addComputeGlobal("noiseAmplitude", 
                                        "sqrt(2*thermal_energy*dt/friction_coeff)")
            integrator.addComputePerDof("old_x", "x")
            integrator.addComputePerDof("x", "x + fscale*f/m + noiseAmplitude*gaussian/sqrt(m)")
            integrator.addComputePerDof("old_v", "v")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "(x-old_x)/dt")
            sampling_method_interface.post_integration_instructions(
                integrator=integrator, anchor=anchor)
            random_seed = integrator_settings.random_seed + self.random_seed
            integrator.setRandomNumberSeed(random_seed)
        elif integrator_type == "verlet":
            #integrator_object = openmm.VerletIntegrator
            if ensemble is not None:
                assert ensemble.type in ["NVE"], \
                    "Ensemble type not allowed or recognized for Verlet "\
                    f"integrator: {ensemble.type}"
            sampling_method_interface.add_global_variables_integrator(
                integrator=integrator, anchor=anchor)
            integrator.addPerDofVariable("old_x", 0)
            integrator.addPerDofVariable("old_v", 0)
            sampling_method_interface.add_per_DOF_variables_integrator(
                integrator=integrator)
            sampling_method_interface.pre_integration_instructions(
                integrator=integrator)
            integrator.addUpdateContextState()
            integrator.addComputePerDof("old_x", "x")
            integrator.addComputePerDof("v", "v+dt*f/m")
            integrator.addComputePerDof("old_v", "v")
            integrator.addComputePerDof("x", "x+dt*v")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "(x-old_x)/dt")
            sampling_method_interface.post_integration_instructions(
                integrator=integrator, anchor=anchor)
        elif integrator_type == "nose_hoover":
            if ensemble is not None:
                assert ensemble.type in ["NVT", "NPT", "NPT_membrane"], \
                    "Ensemble type not allowed or recognized for Langevin "\
                    f"middle integrator: {ensemble.type}"
            temperature = integrator_settings.target_temperature * unit.kelvin
            collision_frequency = integrator_settings.collision_frequency
            chain_length = integrator_settings.chain_length
            num_mts = integrator_settings.num_mts
            num_yoshida_suzuki = integrator_settings.num_yoshida_suzuki
            num_dofs = _compute_full_system_run_dofs(system)
            
            # Thermostat parameters shared by the full Nose-Hoover chain.
            integrator.addGlobalVariable("thermal_energy", kB*temperature)
            integrator.addGlobalVariable("collision_frequency", collision_frequency)
            integrator.addGlobalVariable("num_dofs", num_dofs)
            
            # Work variables for particle kinetic energy and the accumulated
            # velocity scale returned by the chain propagation.
            integrator.addGlobalVariable("kinetic_energy", 0)
            integrator.addGlobalVariable("twice_kinetic_energy", 0)
            integrator.addGlobalVariable("scale", 1)
            
            # Temporary scalars used inside the Yoshida-Suzuki/MTS loop, plus the
            # reported total heat bath energy.
            integrator.addGlobalVariable("wdt", 0)
            integrator.addGlobalVariable("aa", 0)
            integrator.addGlobalVariable("heat_bath_energy", 0)
            
            # Persistent state for each chain bead: position, velocity, mass, and
            # force.  These correspond to chainData, chainMasses, and chainForces
            # in OpenMM's native kernels.
            for bead in range(chain_length):
                integrator.addGlobalVariable(f"chain_pos_{bead}", 0)
                integrator.addGlobalVariable(f"chain_vel_{bead}", 0)
                integrator.addGlobalVariable(f"chain_mass_{bead}", 0)
                integrator.addGlobalVariable(f"chain_force_{bead}", 0)
                
            sampling_method_interface.add_global_variables_integrator(
                integrator=integrator, anchor=anchor)
            integrator.addPerDofVariable("x1", 0)
            integrator.addPerDofVariable("old_x", 0)
            integrator.addPerDofVariable("old_v", 0)
            sampling_method_interface.add_per_DOF_variables_integrator(
                integrator=integrator)
                
            sampling_method_interface.pre_integration_instructions(
                integrator=integrator)
                
            # Update chain masses
            frequency_sq = "collision_frequency*collision_frequency"
            integrator.addComputeGlobal(
                "chain_mass_0",
                f"num_dofs*thermal_energy/({frequency_sq})")
            for bead in range(1, chain_length):
                integrator.addComputeGlobal(
                    f"chain_mass_{bead}",
                    f"thermal_energy/({frequency_sq})")
                
            integrator.addComputePerDof("old_x", "x")
            integrator.addUpdateContextState()
            integrator.addComputePerDof("v", "v + dt*f/m")
            integrator.addConstrainVelocities()
            integrator.addComputePerDof("old_v", "v")
            integrator.addComputePerDof("x", "x + 0.5*dt*v")
            integrator.addComputeSum("kinetic_energy", "0.5*m*v*v")
            integrator.addComputeGlobal("twice_kinetic_energy", "2*kinetic_energy")
            integrator.addComputeGlobal("scale", "1")
            integrator.beginIfBlock("kinetic_energy >= 1e-8")
            # Add initial chain forces
            integrator.addComputeGlobal(
                "chain_force_0",
                "(twice_kinetic_energy - num_dofs*thermal_energy)"
                "/chain_mass_0")
            for bead in range(chain_length - 1):
                integrator.addComputeGlobal(
                    f"chain_force_{bead+1}", f"(chain_mass_{bead}*chain_vel_{bead}"
                    f"*chain_vel_{bead} - thermal_energy)/chain_mass_{bead+1}")
            # Propagate chain
            last_bead = chain_length - 1
            for mts in range(num_mts):
                for weight in _get_yoshida_suzuki_weights(num_yoshida_suzuki):
                    integrator.addComputeGlobal(
                        "wdt", f"{weight}*dt/{num_mts}")
                    integrator.addComputeGlobal(
                        f"chain_vel_{last_bead}",
                        f"chain_vel_{last_bead} + 0.5*wdt*chain_force_{last_bead}")
                    for bead in range(chain_length - 2, -1, -1):
                        integrator.addComputeGlobal(
                            "aa", f"exp(-0.25*wdt*chain_vel_{bead+1})")
                        integrator.addComputeGlobal(
                            f"chain_vel_{bead}",
                            f"aa*(chain_vel_{bead}*aa + 0.5*wdt*chain_force_{bead})")
                    integrator.addComputeGlobal(
                        "scale", "scale*exp(-wdt*chain_vel_0)")
                    for bead in range(chain_length):
                        integrator.addComputeGlobal(
                            f"chain_pos_{bead}",
                            f"chain_pos_{bead} + chain_vel_{bead}*wdt")
                    integrator.addComputeGlobal(
                        "chain_force_0",
                        "(scale*scale*twice_kinetic_energy"
                        " - num_dofs*thermal_energy)/chain_mass_0")
                    for bead in range(chain_length - 1):
                        integrator.addComputeGlobal(
                            "aa", f"exp(-0.25*wdt*chain_vel_{bead+1})")
                        integrator.addComputeGlobal(
                            f"chain_vel_{bead}",
                            f"aa*(aa*chain_vel_{bead}"
                            f" + 0.5*wdt*chain_force_{bead})")
                        integrator.addComputeGlobal(
                            f"chain_force_{bead+1}",
                            f"(chain_mass_{bead}*chain_vel_{bead}"
                            f"*chain_vel_{bead} - thermal_energy)"
                            f"/chain_mass_{bead+1}")
                    integrator.addComputeGlobal(
                        f"chain_vel_{last_bead}",
                        f"chain_vel_{last_bead}"
                        f" + 0.5*wdt*chain_force_{last_bead}")
            integrator.endBlock()
            integrator.addComputePerDof("v", "scale*v")
            
            integrator.addComputePerDof("x", "x + 0.5*dt*v")
            integrator.addComputePerDof("x1", "x")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "v + (x-x1)/dt")
            # Compute heat bath energies
            energy_terms = []
            for bead in range(chain_length):
                prefac = "num_dofs" if bead == 0 else "1"
                energy_terms.append(
                    f"0.5*chain_mass_{bead}*chain_vel_{bead}"
                    f"*chain_vel_{bead}"
                    f" + {prefac}*thermal_energy*chain_pos_{bead}")
            integrator.addComputeGlobal("heat_bath_energy", " + ".join(energy_terms))
            
            sampling_method_interface.post_integration_instructions(
                integrator=integrator, anchor=anchor)
            
        else:
            raise Exception("Integrator type not available: "\
                            f"{integrator_type}")
        
        if rigid_constraint_tolerance is not None:
            integrator.setConstraintTolerance(
                rigid_constraint_tolerance)
        self.integrator = integrator
        return
    
    def add_barostat(
            self, 
            model: structures.Seekr_model,
            ensemble: base.Ensemble,
            ) -> typing.Any:
        """
        Add a barostat to the simulation.
        """
        
        if ensemble.type in ["NVE", "NVT"]:
            return
        else:
            md_engine = model.get_engine_by_scale_type("molecular_dynamics")
            assert md_engine.type == "openmm"
            assert md_engine.integrator_settings is not None
            integrator_settings = md_engine.integrator_settings
            target_temperature = integrator_settings.target_temperature
            if ensemble.type == "NPT":
                barostat = openmm.MonteCarloBarostat(
                    ensemble.pressure*unit.bar,
                    target_temperature*unit.kelvin,
                    DEFAULT_BAROSTAT_FREQUENCY)
                self.system.addForce(barostat)
            elif ensemble.type == "NPT_membrane":
                barostat = openmm.MonteCarloMembraneBarostat(
                    ensemble.pressure*unit.atmospheres,
                    0.0*unit.bar*unit.nanometers,
                    target_temperature*unit.kelvin,
                    openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.MonteCarloMembraneBarostat.ZFixed)
                self.system.addForce(barostat)
            else:
                raise Exception("Ensemble type not recognized: "\
                                f"{ensemble.type}")
        return
    
    def add_platform(
            self, 
            model: structures.Seekr_model,
            ) -> None:
        """
        Add the platform to the simulation.
        """
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        assert md_engine.type == "openmm"
        platform_settings = md_engine.platform_settings
        if platform_settings is None:
            self.platform = openmm.Platform.getPlatformByName("Reference")
            self.properties = {}
            return
        elif platform_settings.type == "cuda":
            self.platform = openmm.Platform.getPlatformByName("CUDA")
            self.properties = {"CudaDeviceIndex": platform_settings.device_index, 
                               "CudaPrecision": platform_settings.precision,
                               "DeterministicForces": "true"}
        else:
            raise Exception("Platform not recognized: "\
                            f"{platform_settings.type}")
        return
    
    def add_simulation(
            self, 
            model: structures.Seekr_model,
            positions: unit.Quantity | None,
            velocities: unit.Quantity | None = None
            ) -> unit.Quantity:
        """
        Add the simulation object to the OpenMM simulation.
        """
        assert self.system is not None
        assert self.integrator is not None
        assert self.platform is not None
        assert self.properties is not None
        md_engine = model.get_engine_by_scale_type("molecular_dynamics")
        self.simulation = openmm_app.Simulation(
            self.topology, self.system, self.integrator, self.platform, 
            self.properties)
        if positions is not None:
            self.simulation.context.setPositions(positions)
            assert md_engine is not None
            if velocities is None:
                random_seed = self.random_seed + RANDOM_VEL_SEED_FACTOR 
                if hasattr(md_engine.integrator_settings, "random_seed"):
                        random_seed += md_engine.integrator_settings.random_seed
                self.simulation.context.setVelocitiesToTemperature(
                    md_engine.initial_temperature * unit.kelvin, random_seed)
                state = self.simulation.context.getState(getVelocities=True, getPositions=True)
                positions = state.getPositions(asNumpy=True)
                assert np.all(np.isfinite(positions)), "Positions contain non-finite values."
                velocities = state.getVelocities(asNumpy=True)
                assert np.all(np.isfinite(velocities)), "Velocities contain non-finite values."
            else:
                self.simulation.context.setVelocities(velocities)
        if self.box_vectors is not None:
            self.simulation.context.setPeriodicBoxVectors(
                *self.box_vectors)
        
        return
    
    def write_pdb_file(
            self, 
            filename: str,
            positions: unit.Quantity | None = None,
            box_vectors: unit.Quantity | None = None
            ) -> None:
        """
        Write the PDB file to the output directory.
        """
        if (positions is None) or (box_vectors is None):
            assert self.simulation is not None
            state = self.simulation.context.getState(getPositions=True,
                                                     enforcePeriodicBox=True)
        if positions is None:
            positions = state.getPositions()
        if box_vectors is None:
            box_vectors = state.getPeriodicBoxVectors()

        assert self.topology is not None
        self.topology.setPeriodicBoxVectors(box_vectors)
        with open(filename, "w") as f:
            openmm_app.PDBFile.writeFile(
                self.topology, positions, f, 
                keepIds=True)
        return

def write_toy_pdb_file(topology, positions, pdb_file_name):
    """
    For toy systems, write a PDB file to use as a topology file for
    mdtraj.
    """
    positions_nm = np.array(positions) * unit.nanometers
    writing_positions = positions_nm.value_in_unit(unit.angstroms)
    with open(pdb_file_name, "w") as out_file:
        openmm_app.PDBFile.writeHeader(topology, out_file)
        openmm_app.PDBFile.writeModel(topology, writing_positions, out_file)
        openmm_app.PDBFile.writeFooter(topology, out_file)
    return

def create_sim_openmm(
        model: structures.Seekr_model, 
        anchor_index: int | None,
        positions_directory: str,
        positions_basename: str,
        md_parameters_directory: str,
        md_parameters_topology: engines_structures.Parameters_topology,
        running_directory: str,
        ensemble: base.Ensemble,
        sampling_method_interface: sampling_openmm_interface.Sampling_md_openmm_interface,
        completion_criteria_interface: completion_openmm_interface.Completion_criteria_openmm_interface,
        restraints_interfaces: list[restraints_openmm_interface.Base_restraint_openmm_interface],
        swarm_id: int,
        restart: bool,
        timestep: float | None = None
        ) -> Sim_openmm:
    """
    Generate an OpenMM simulation and other objects for a given
    calculation.
    """
    sim_openmm = Sim_openmm(anchor_index, swarm_id)
    system, topology, positions, box_vectors, _, velocities \
        = create_openmm_system(
            model, positions_directory, positions_basename, 
            md_parameters_directory, md_parameters_topology)
    sim_openmm.system = system
    sim_openmm.topology = topology
    sim_openmm.box_vectors = box_vectors
    if model.using_toy():
        periodic_system = False
    else:
        if box_vectors is None:
            periodic_system = False
        else:
            periodic_system = True
    # TODO: replace with adding integrator from sampling method
    # A sampling method of None will be conventional MD.
    sim_openmm.add_integrator(
        model, anchor_index, ensemble, sampling_method_interface,
        completion_criteria_interface, restart, system, timestep)
    if ensemble is not None:
        sim_openmm.add_barostat(model, ensemble)
    sim_openmm.add_platform(model)
    # TODO: add forces here based on sampling method
    sim_openmm.add_restraints(
        model, anchor_index, restraints_interfaces, positions, 
        periodic_system)
    sim_openmm.add_simulation(model, positions, velocities)
    if model.using_toy():
        toy_pdb_filename = os.path.join(running_directory, "toy.pdb")
        write_toy_pdb_file(topology, positions, toy_pdb_filename)
    return sim_openmm
