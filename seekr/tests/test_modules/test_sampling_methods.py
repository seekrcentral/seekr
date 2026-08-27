from __future__ import annotations
"""
test_sampling_methods.py

Smoke tests for the Conventional sampling method's heat-ramp feature. The ramp
must run both for an integrator that exposes a "thermal_energy" global variable
(Langevin) and for one that does not (Verlet), where the temperature is only
applied via setVelocitiesToTemperature.
"""

import os
from unittest.mock import MagicMock

import openmm
import openmm.app as openmm_app
import openmm.unit as unit
import pytest

import seekr.prepare as prepare
import seekr.run as run
import seekr.modules.base as base
import seekr.tests.create_seekr_input as create_seekr_input
import seekr.modules.sampling_methods.structures as sampling_methods_structures
import seekr.modules.sampling_methods.openmm_interface as sampling_openmm_interface


@pytest.mark.parametrize("integrator_type, ensemble_cls", [
    ("langevin", base.NVT_ensemble),
    ("verlet", base.NVE_ensemble),
])
def test_heat_ramp_runs(tmp_path, integrator_type, ensemble_cls):
    """A Conventional heat ramp runs to completion for integrators with and
    without a 'thermal_energy' global variable."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path))
    seekr_input.engine_settings_list[0].integrator_type = integrator_type
    stage1 = seekr_input.stages[0]
    stage1.ensemble = ensemble_cls()
    stage1.sampling_method.heat_ramp_temperatures = [100.0, 200.0, 300.0]
    stage1.sampling_method.heat_ramp_step_interval = 100
    stage1.completion_criteria.number_of_steps = 500
    old_dir = os.getcwd()
    try:
        model, model_xml_path = prepare.prepare(
            seekr_input, force_overwrite=False)
        model.directory = os.path.abspath(os.path.dirname(model_xml_path))
        run.run(model, "equilibration")
    finally:
        os.chdir(old_dir)
    assert os.path.isdir(model.directory)


def _one_particle_topology():
    topology = openmm_app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("UNK", chain)
    topology.addAtom("Ar", openmm.app.element.argon, residue)
    topology.setPeriodicBoxVectors(
        (unit.Quantity((2.0, 0.0, 0.0), unit.nanometer),
         unit.Quantity((0.0, 2.0, 0.0), unit.nanometer),
         unit.Quantity((0.0, 0.0, 2.0), unit.nanometer)))
    return topology


def _make_conventional_iface(system, add_barostat=False):
    if add_barostat:
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(0.8 * unit.nanometer)
        nb.addParticle(0.0, 0.1, 0.0)
        system.addForce(nb)
        system.addForce(openmm.MonteCarloBarostat(
            1.0 * unit.bar, 300.0 * unit.kelvin))
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm_app.Simulation(
        _one_particle_topology(), system, integrator, platform)
    simulation.context.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
    iface = sampling_openmm_interface.Conventional_md_openmm_interface(
        stage_index=0,
        sampling_method=sampling_methods_structures.Conventional_md_sampling_method(),
    )
    iface.assign_system_openmm(system)
    iface.assign_sim_openmm(simulation)
    iface.pre_simulation_preparations(model=MagicMock())
    return iface


def test_heat_ramp_updates_barostat_context_temperature():
    """NPT: heat-ramp T is written to the live MonteCarloBarostat Context parameter."""
    system = openmm.System()
    system.addParticle(1.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(
        (2.0, 0.0, 0.0) * unit.nanometer,
        (0.0, 2.0, 0.0) * unit.nanometer,
        (0.0, 0.0, 2.0) * unit.nanometer)
    iface = _make_conventional_iface(system, add_barostat=True)
    assert isinstance(iface._barostat, openmm.MonteCarloBarostat)
    iface._apply_heat_ramp_temperature(400.0, 0)
    T = iface._simulation_obj.context.getParameter(
        openmm.MonteCarloBarostat.Temperature())
    assert T == pytest.approx(400.0)


def test_heat_ramp_skips_barostat_when_absent():
    """NVT: no barostat is stored, and setParameter is not called."""
    system = openmm.System()
    system.addParticle(1.0 * unit.amu)
    iface = _make_conventional_iface(system, add_barostat=False)
    assert iface._barostat is None
    iface._simulation_obj.context.setParameter = MagicMock()
    iface._apply_heat_ramp_temperature(400.0, 0)
    iface._simulation_obj.context.setParameter.assert_not_called()



@pytest.mark.parametrize("integrator_type, ensemble_cls", [
    ("langevin", base.NVT_ensemble),
    ("verlet", base.NVE_ensemble),
])
def test_heat_ramp_runs(tmp_path, integrator_type, ensemble_cls):
    """A Conventional heat ramp runs to completion for integrators with and
    without a 'thermal_energy' global variable."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path))
    seekr_input.engine_settings_list[0].integrator_type = integrator_type
    stage1 = seekr_input.stages[0]
    stage1.ensemble = ensemble_cls()
    stage1.sampling_method.heat_ramp_temperatures = [100.0, 200.0, 300.0]
    stage1.sampling_method.heat_ramp_step_interval = 100
    stage1.completion_criteria.number_of_steps = 500
    old_dir = os.getcwd()
    try:
        model, model_xml_path = prepare.prepare(
            seekr_input, force_overwrite=False)
        model.directory = os.path.abspath(os.path.dirname(model_xml_path))
        run.run(model, "equilibration")
    finally:
        os.chdir(old_dir)
    assert os.path.isdir(model.directory)
