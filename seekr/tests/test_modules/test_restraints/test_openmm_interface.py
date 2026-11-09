from __future__ import annotations
"""
Tests for modules/restraints/openmm_interface.py (Z and torsion restraints).
"""

import openmm
import openmm.unit as unit

import seekr.modules.engines.openmm.structures as openmm_engine_structures
import seekr.modules.restraints.openmm_interface as restraints_openmm
import seekr.modules.restraints.structures as restraints_structures


def _system_with_n_particles(n: int) -> openmm.System:
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    return system


def _positions(n: int):
    coords = [[float(i), float(i) + 0.1, float(i) + 0.2] for i in range(n)]
    return unit.Quantity(coords, unit.nanometer)


def test_z_restraint_adds_custom_external_force_nonperiodic():
    restraint = restraints_structures.Z_restraint(
        atom_indices=[0, 2], force_constant=400.0)
    iface = restraints_openmm.Z_restraint_openmm_interface(restraint)
    system = _system_with_n_particles(4)
    n_before = system.getNumForces()
    iface.add_forces_to_system(
        system=system,
        positions=_positions(4),
        periodic_system=False,
    )
    assert system.getNumForces() == n_before + 1
    force = system.getForce(n_before)
    assert isinstance(force, openmm.CustomExternalForce)
    assert force.getNumPerParticleParameters() == 1
    assert force.getPerParticleParameterName(0) == "z0"
    assert force.getNumParticles() == 2


def test_z_restraint_adds_force_periodic():
    restraint = restraints_structures.Z_restraint(atom_indices=[1])
    iface = restraints_openmm.Z_restraint_openmm_interface(restraint)
    system = _system_with_n_particles(3)
    iface.add_forces_to_system(
        system=system,
        positions=_positions(3),
        periodic_system=True,
    )
    force = system.getForce(system.getNumForces() - 1)
    assert isinstance(force, openmm.CustomExternalForce)
    assert "periodicdistance" in force.getEnergyFunction()


def test_torsion_restraint_adds_custom_torsion_force():
    restraint = restraints_structures.Torsion_restraint(
        torsion_indices=[(0, 1, 2, 3)],
        torsion_values=[0.25],
        force_constant=50.0,
    )
    iface = restraints_openmm.Torsion_restraint_openmm_interface(restraint)
    system = _system_with_n_particles(4)
    n_before = system.getNumForces()
    iface.add_forces_to_system(system=system)
    assert system.getNumForces() == n_before + 1
    force = system.getForce(n_before)
    assert isinstance(force, openmm.CustomTorsionForce)
    assert force.getNumTorsions() == 1
    assert force.getNumPerTorsionParameters() == 1
    assert "min(dtheta" in force.getEnergyFunction()


def test_assign_restraints_interface_z_and_torsion():
    engine = openmm_engine_structures.Openmm_md_engine()
    z = restraints_structures.Z_restraint(atom_indices=[0])
    t = restraints_structures.Torsion_restraint(
        torsion_indices=[(0, 1, 2, 3)], torsion_values=[0.0])
    z_iface = restraints_openmm.assign_restraints_interface(z, engine, [])
    t_iface = restraints_openmm.assign_restraints_interface(t, engine, [])
    assert isinstance(z_iface, restraints_openmm.Z_restraint_openmm_interface)
    assert isinstance(t_iface, restraints_openmm.Torsion_restraint_openmm_interface)

def test_positional_restraint_uses_swarm_block(tmp_path):
    coords = tmp_path / "coords.txt"
    coords.write_text(
        "0 0 0\n"
        "1 1 1\n"
        "2 2 2\n"
        "3 3 3\n"
    )
    restraint = restraints_structures.Positional_restraint(
        atom_indices=[0, 1],
        force_constant=400.0,
        coordinates_filename=coords.name,
    )
    iface = restraints_openmm.Positional_restraint_openmm_interface(restraint)
    model = type("Model", (), {"directory": str(tmp_path)})()

    def targets(swarm_id):
        system = _system_with_n_particles(2)
        iface.add_forces_to_system(
            model=model, system=system, swarm_id=swarm_id, periodic_system=False)
        force = system.getForce(0)
        return [list(force.getParticleParameters(i)[1]) for i in range(2)]

    assert targets(0) == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert targets(1) == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]