from __future__ import annotations
"""
Tests for modules/engines/openmm/sim.py.
"""

# TODO: tests needed for Gromacs, CHARMM, OpenMM system inputs.
# This will need the inclusion of a input files of these types
# for host-guest?

import os
from unittest.mock import MagicMock

import pytest
import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import seekr.modules.base as base
import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.openmm.sim as sim_mod

_TEMPERATURE = 298.15
_TIMESTEP = 0.002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(anchor_index=0, swarm_id=0):
    return sim_mod.Sim_openmm(anchor_index, swarm_id)


def _make_sampling_mock():
    """MagicMock that satisfies the sampling_method_interface protocol."""
    mock = MagicMock()
    mock.sampling_method.needs_to_be_in_anchor.return_value = False
    return mock


def _set_integrator(model, integrator_type):
    """Replace the engine's integrator_settings with the requested type."""
    engine = model.get_engine_by_scale_type("molecular_dynamics")
    if integrator_type == "langevinmiddle":
        engine.integrator_settings = \
            engines_structures.Langevin_middle_integrator_settings(
                timestep=_TIMESTEP, target_temperature=_TEMPERATURE)
    elif integrator_type == "brownian":
        engine.integrator_settings = \
            engines_structures.Brownian_integrator_settings(
                timestep=_TIMESTEP, target_temperature=_TEMPERATURE)
    elif integrator_type == "verlet":
        engine.integrator_settings = \
            engines_structures.Verlet_integrator_settings(timestep=_TIMESTEP)
    elif integrator_type == "langevin":
        engine.integrator_settings = \
            engines_structures.Langevin_integrator_settings(
                timestep=_TIMESTEP, target_temperature=_TEMPERATURE)


def _run_add_integrator(model, integrator_type, ensemble=None):
    _set_integrator(model, integrator_type)
    s = _make_sim()
    s.add_integrator(
        model=model,
        anchor_index=0,
        ensemble=ensemble,
        sampling_method_interface=_make_sampling_mock(),
        completion_criteria_interface=MagicMock(),
        restart=False,
        system=None
    )
    return s


def _one_atom_topology():
    topology = openmm_app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("RES", chain)
    topology.addAtom("C", openmm_app.element.carbon, residue)
    return topology


# ---------------------------------------------------------------------------
# add_integrator — integrator type branches
# ---------------------------------------------------------------------------

def test_add_integrator_langevinMiddle(host_guest_mmvt_model):
    s = _run_add_integrator(host_guest_mmvt_model, "langevinmiddle",
                            ensemble=base.NVT_ensemble())
    assert isinstance(s.integrator, openmm.CustomIntegrator)
    assert s.timestep == _TIMESTEP


def test_add_integrator_brownian(host_guest_mmvt_model):
    s = _run_add_integrator(host_guest_mmvt_model, "brownian",
                            ensemble=base.NVT_ensemble())
    assert isinstance(s.integrator, openmm.CustomIntegrator)
    assert s.timestep == _TIMESTEP


def test_add_integrator_verlet(host_guest_mmvt_model):
    s = _run_add_integrator(host_guest_mmvt_model, "verlet",
                            ensemble=base.NVE_ensemble())
    assert isinstance(s.integrator, openmm.CustomIntegrator)
    assert s.timestep == _TIMESTEP


def test_add_integrator_sets_constraint_tolerance(host_guest_mmvt_model):
    s = _run_add_integrator(host_guest_mmvt_model, "langevin",
                            ensemble=base.NVT_ensemble())
    engine = host_guest_mmvt_model.get_engine_by_scale_type("molecular_dynamics")
    expected_tol = engine.integrator_settings.rigid_tolerance
    assert abs(s.integrator.getConstraintTolerance() - expected_tol) < 1e-10


# ---------------------------------------------------------------------------
# add_barostat
# ---------------------------------------------------------------------------

def test_add_barostat_NVT_is_noop(host_guest_mmvt_model):
    """NVT ensemble: no force is added to the system."""
    _set_integrator(host_guest_mmvt_model, "langevin")
    s = _make_sim()
    s.system = openmm.System()
    before = s.system.getNumForces()
    s.add_barostat(host_guest_mmvt_model, base.NVT_ensemble())
    assert s.system.getNumForces() == before


def test_add_barostat_NPT_adds_force(host_guest_mmvt_model):
    """NPT ensemble: a MonteCarloBarostat is added."""
    _set_integrator(host_guest_mmvt_model, "langevin")
    s = _make_sim()
    s.system = openmm.System()
    s.add_barostat(host_guest_mmvt_model, base.NPT_ensemble(pressure=1.0))
    assert s.system.getNumForces() == 1
    assert isinstance(s.system.getForce(0), openmm.MonteCarloBarostat)


def test_add_barostat_NVE_is_noop(host_guest_mmvt_model):
    """NVE ensemble: no force is added."""
    _set_integrator(host_guest_mmvt_model, "verlet")
    s = _make_sim()
    s.system = openmm.System()
    before = s.system.getNumForces()
    s.add_barostat(host_guest_mmvt_model, base.NVE_ensemble())
    assert s.system.getNumForces() == before


# ---------------------------------------------------------------------------
# write_pdb_file
# ---------------------------------------------------------------------------

def test_write_pdb_file(tmp_path):
    """write_pdb_file writes a readable PDB when positions/box_vectors supplied."""
    s = _make_sim()
    s.topology = _one_atom_topology()
    positions = [openmm.Vec3(0.1, 0.2, 0.3)] * unit.nanometers
    box_vectors = (
        openmm.Vec3(3.0, 0.0, 0.0),
        openmm.Vec3(0.0, 3.0, 0.0),
        openmm.Vec3(0.0, 0.0, 3.0),
    ) * unit.nanometers
    out_file = str(tmp_path / "out.pdb")
    s.write_pdb_file(out_file, positions=positions, box_vectors=box_vectors)
    assert os.path.exists(out_file)
    content = open(out_file).read()
    assert "ATOM" in content or "HETATM" in content


def test_particle_masses_from_openmm_parameters_toy():
    model = MagicMock()
    model.using_toy.return_value = True
    assert sim_mod.particle_masses_from_openmm_parameters(model) is None


def test_particle_masses_from_openmm_system_xml(tmp_path):
    """OpenMM_system XML particle masses are read without createSystem."""
    system = openmm.System()
    system.addParticle(6.034 * unit.dalton)
    system.addParticle(16.0 * unit.dalton)
    param_dir = tmp_path / "unpartitioned_scope" / "stage_0_initial"
    param_dir.mkdir(parents=True)
    xml_name = "system.xml"
    (param_dir / xml_name).write_text(openmm.XmlSerializer.serialize(system))

    topo = engines_structures.Openmm_system(
        system_filename=xml_name, topology_filename="dummy.pdb")
    settings = MagicMock()
    settings.directory = "unpartitioned_scope"
    settings.stage_parameters = {
        "initial": {"directory": "stage_0_initial"},
    }
    settings.md_parameters_topology = topo
    md_engine = MagicMock()
    md_engine.unpartitioned_settings = settings
    md_engine.hydrogen_mass = 3.0
    model = MagicMock()
    model.using_toy.return_value = False
    model.directory = str(tmp_path)
    model.get_engine_by_scale_type.return_value = md_engine

    sim_mod._OPENMM_PARTICLE_MASSES_CACHE.clear()
    masses = sim_mod.particle_masses_from_openmm_parameters(model)
    assert masses is not None
    assert list(masses) == pytest.approx([6.034, 16.0])
    cached = sim_mod.particle_masses_from_openmm_parameters(model)
    assert cached is masses

