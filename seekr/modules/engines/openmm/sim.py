from __future__ import annotations
"""
modules/engines/openmm/sim.py

Objects needed for OpenMM Simulation and System objects.
"""

import os
import typing

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
        topology_obj: openmm_app.PDBFile,
        directory: str,
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> openmm.System:
    """
    Load the OpenMM forcefield parameters and topology from the 
    provided object, and return OpenMM system and topology.
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
        topology_obj.topology, nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        hydrogenMass=hydrogenMass)
    return system

def make_topology_system_from_xml(
        openmm_system_obj: engines_structures.Openmm_system,
        directory: str, 
        nonbonded_method: typing.Any,
        nonbonded_cutoff: unit.Quantity | None,
        constraints: typing.Any,
        hydrogenMass: unit.Quantity | None
        ) -> openmm.System:
    """
    Load the OpenMM system XML from the 
    provided object, and return OpenMM system.
    """
    # TODO: enforce these assertions since the prepare stage.
    assert nonbonded_method is None, \
        "When an OpenMM system is loaded from XML, the nonbonded method "\
        "is already set within the XML file, so it should not be provided."
    assert nonbonded_cutoff is None, \
        "When an OpenMM system is loaded from XML, the nonbonded cutoff "\
        "is already set within the XML file, so it should not be provided."
    assert constraints is None, \
        "When an OpenMM system is loaded from XML, the constraints "\
        "are already set within the XML file, so it should not be provided."
    assert hydrogenMass is None, \
        "When an OpenMM system is loaded from XML, the hydrogen mass "\
        "is already set within the XML file, so it should not be provided."
    system_xml_filename = os.path.join(
        directory, openmm_system_obj.system_filename)
    with open(system_xml_filename) as input:
        system = openmm.XmlSerializer.deserialize(input.read())
    return system

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
        elif md_parameters_topology.type == "Forcefield":
            system = make_topology_system_from_forcefield(
                md_parameters_topology, pdb_obj,
                md_parameters_directory, nonbondedMethod,
                md_engine.nonbonded_cutoff, constraints, hydrogenMass)
            assert pdb_obj is not None
            topology = pdb_obj.topology

    # set all non-milestoning force groups to zero
    for force in system.getForces():
        force.setForceGroup(0)

    return system, topology, positions, box_vectors, num_frames, velocities

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
            timestep: float | None = None
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
        completion_criteria_interface, restart, timestep)
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